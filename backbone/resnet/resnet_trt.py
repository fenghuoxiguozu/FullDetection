import numpy as np
import tensorrt as trt
from backbone.build_trt import TRT_BACKBONE_REGISTRY


def addBatchNorm2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)
    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input=input, mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale)


def bottleneck(network, weight_map, input, in_channels, out_channels, stride, layer_name, EPS=1e-5):
    # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)
    conv1 = network.add_convolution(input=input, num_output_maps=out_channels, kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name + "conv1.weight"], bias=trt.Weights())
    # self.bn1 = nn.BatchNorm2d(width)
    bn1 = addBatchNorm2d(network, weight_map,
                         conv1.get_output(0), layer_name + "bn1", EPS)
    # self.relu = nn.ReLU(inplace=True)
    relu1 = network.add_activation(
        bn1.get_output(0), type=trt.ActivationType.RELU)

    # self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
    conv2 = network.add_convolution(input=relu1.get_output(0), num_output_maps=out_channels, kernel_shape=(3, 3),
                                    kernel=weight_map[layer_name + "conv2.weight"], bias=trt.Weights())
    conv2.stride = (stride, stride)
    conv2.padding = (1, 1)
    # self.bn2 = nn.BatchNorm2d(width)
    bn2 = addBatchNorm2d(network, weight_map,
                         conv2.get_output(0), layer_name + "bn2", EPS)
    # self.relu = nn.ReLU(inplace=True)
    relu2 = network.add_activation(
        bn2.get_output(0), type=trt.ActivationType.RELU)

    # self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,kernel_size=1, stride=1, bias=False)
    conv3 = network.add_convolution(input=relu2.get_output(0), num_output_maps=out_channels * 4, kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name + "conv3.weight"], bias=trt.Weights())
    # self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
    bn3 = addBatchNorm2d(network, weight_map,
                         conv3.get_output(0), layer_name + "bn3", EPS)
    # if stride != 1 or self.in_channel != channel * block.expansion:
    if stride != 1 or in_channels != 4 * out_channels:
        # nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
        conv4 = network.add_convolution(input=input, num_output_maps=out_channels * 4, kernel_shape=(1, 1),
                                        kernel=weight_map[layer_name + "downsample.0.weight"], bias=trt.Weights())
        conv4.stride = (stride, stride)
        # nn.BatchNorm2d(channel * block.expansion))
        bn4 = addBatchNorm2d(network, weight_map, conv4.get_output(
            0), layer_name + "downsample.1", EPS)

        ew1 = network.add_elementwise(bn4.get_output(
            0), bn3.get_output(0), trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(
            input, bn3.get_output(0), trt.ElementWiseOperation.SUM)
    # self.relu = nn.ReLU(inplace=True)
    relu3 = network.add_activation(
        ew1.get_output(0), type=trt.ActivationType.RELU)
    return relu3


class TRTResnet:
    def __init__(self, weight_map, network, cfg, EPS=1e-5):
        self.wm = weight_map
        self.cfg = cfg
        self.network = network
        self.EPS = EPS

    def forward(self):
        data = self.network.add_input(self.cfg["INPUT_NAME"], eval(
            self.cfg["DTYPE"]), (self.cfg["INPUT_C"], self.cfg["INPUT_H"], self.cfg["INPUT_W"]))
        # self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        conv1 = self.network.add_convolution(input=data, num_output_maps=self.cfg["out_channel"], kernel_shape=(7, 7),
                                             kernel=self.wm["conv1.weight"], bias=trt.Weights())
        conv1.stride = (2, 2)
        conv1.padding = (3, 3)
        # self.bn1 = nn.BatchNorm2d(self.in_channel)
        bn1 = addBatchNorm2d(self.network, self.wm,
                             conv1.get_output(0), "bn1", self.EPS)
        # self.relu = nn.ReLU(inplace=True)
        relu1 = self.network.add_activation(
            bn1.get_output(0), type=trt.ActivationType.RELU)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        pool1 = self.network.add_pooling(input=relu1.get_output(
            0), window_size=trt.DimsHW(3, 3), type=trt.PoolingType.MAX)
        pool1.stride = (2, 2)
        pool1.padding = (1, 1)

        x = bottleneck(self.network, self.wm, pool1.get_output(0),
                       64, 64, 1, "layer1.0.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       256, 64, 1, "layer1.1.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       256, 64, 1, "layer1.2.")

        x = bottleneck(self.network, self.wm, x.get_output(0),
                       256, 128, 2, "layer2.0.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       512, 128, 1, "layer2.1.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       512, 128, 1, "layer2.2.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       512, 128, 1, "layer2.3.")

        x = bottleneck(self.network, self.wm, x.get_output(0),
                       512, 256, 2, "layer3.0.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       1024, 256, 1, "layer3.1.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       1024, 256, 1, "layer3.2.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       1024, 256, 1, "layer3.3.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       1024, 256, 1, "layer3.4.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       1024, 256, 1, "layer3.5.")

        x = bottleneck(self.network, self.wm, x.get_output(0),
                       1024, 512, 2, "layer4.0.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       2048, 512, 1, "layer4.1.")
        x = bottleneck(self.network, self.wm, x.get_output(0),
                       2048, 512, 1, "layer4.2.")

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        pool2 = self.network.add_pooling(x.get_output(
            0), window_size=trt.DimsHW(7, 7), type=trt.PoolingType.AVERAGE)
        pool2.stride = (1, 1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        fc1 = self.network.add_fully_connected(input=pool2.get_output(0),
                                               num_outputs=self.cfg["NUM_CLASSES"],
                                               kernel=self.wm['fc.weight'],
                                               bias=self.wm['fc.bias'])
        fc1.get_output(0).name = self.cfg["OUTPUT_NAME"]
        self.network.mark_output(fc1.get_output(0))
        return self.network


@TRT_BACKBONE_REGISTRY.register()
def build_resnet50_trt_backbone(weight_map, builder, cfg):
    return TRTResnet(weight_map, builder, cfg)
