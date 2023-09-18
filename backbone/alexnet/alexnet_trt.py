import tensorrt as trt
import torch.nn as nn
from backbone.build_trt import TRT_BACKBONE_REGISTRY


@TRT_BACKBONE_REGISTRY.register()
class TRTAlexnet:
    def __init__(self, weight_map, network, cfg):
        self.wm = weight_map
        self.network = network
        self.cfg = cfg

    def forward(self):
        data = self.network.add_input(self.cfg["INPUT_NAME"], eval(
            self.cfg["DTYPE"]), (self.cfg["INPUT_C"], self.cfg["INPUT_H"], self.cfg["INPUT_W"]))

        # 0.nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2)
        conv1 = self.network.add_convolution(input=data, num_output_maps=64, kernel_shape=(11, 11),
                                             kernel=self.wm["features.0.weight"], bias=self.wm["features.0.bias"])
        conv1.stride = (4, 4)
        conv1.padding = (2, 2)
        # 1.nn.ReLU(inplace=True)
        relu1 = self.network.add_activation(
            conv1.get_output(0), type=trt.ActivationType.RELU)
        # 2.nn.MaxPool2d(kernel_size=3, stride=2)
        pool1 = self.network.add_pooling(input=relu1.get_output(0),
                                         type=trt.PoolingType.MAX,
                                         window_size=trt.DimsHW(3, 3))
        pool1.stride_nd = (2, 2)
        # 3.nn.Conv2d(64, 192, kernel_size=5, padding=2)
        conv2 = self.network.add_convolution(input=pool1.get_output(0), num_output_maps=192, kernel_shape=(5, 5),
                                             kernel=self.wm["features.3.weight"], bias=self.wm["features.3.bias"])
        conv2.padding = (2, 2)
        # 4.nn.ReLU(inplace=True)
        relu2 = self.network.add_activation(
            conv2.get_output(0), type=trt.ActivationType.RELU)
        # 5.nn.MaxPool2d(kernel_size=3, stride=2)
        pool2 = self.network.add_pooling(input=relu2.get_output(
            0), type=trt.PoolingType.MAX, window_size=trt.DimsHW(3, 3))
        pool2.stride_nd = (2, 2)
        # 6.nn.Conv2d(192, 384, kernel_size=3, padding=1)
        conv3 = self.network.add_convolution(input=pool2.get_output(0), num_output_maps=384, kernel_shape=(3, 3),
                                             kernel=self.wm["features.6.weight"], bias=self.wm["features.6.bias"])
        conv3.padding = (1, 1)
        # 7.nn.ReLU(inplace=True)
        relu3 = self.network.add_activation(
            conv3.get_output(0), type=trt.ActivationType.RELU)
        # 8.nn.Conv2d(384, 256, kernel_size=3, padding=1)
        conv4 = self.network.add_convolution(input=relu3.get_output(0), num_output_maps=256, kernel_shape=(3, 3),
                                             kernel=self.wm["features.8.weight"], bias=self.wm["features.8.bias"])
        conv4.padding = (1, 1)
        # 9.nn.ReLU(inplace=True)
        relu4 = self.network.add_activation(
            conv4.get_output(0), type=trt.ActivationType.RELU)
        # 10.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        conv5 = self.network.add_convolution(input=relu4.get_output(0), num_output_maps=256, kernel_shape=(3, 3),
                                             kernel=self.wm["features.10.weight"], bias=self.wm["features.10.bias"])
        conv5.padding = (1, 1)
        # 11.ReLU(inplace=True)
        relu5 = self.network.add_activation(
            conv5.get_output(0), type=trt.ActivationType.RELU)
        # 12.nn.MaxPool2d(kernel_size=3, stride=2)
        pool3 = self.network.add_pooling(input=relu5.get_output(
            0), type=trt.PoolingType.MAX, window_size=trt.DimsHW(3, 3))
        pool3.stride_nd = (2, 2)
        # nn.AdaptiveAvgPool2d((6, 6))
        # nn.Dropout(p=dropout)
        # nn.Linear(256 * 6 * 6, 4096),
        fc1 = self.network.add_fully_connected(input=pool3.get_output(0), num_outputs=4096,
                                               kernel=self.wm["classifier.1.weight"], bias=self.wm["classifier.1.bias"])
        # nn.ReLU(inplace=True),
        relu6 = self.network.add_activation(
            fc1.get_output(0), type=trt.ActivationType.RELU)
        # nn.Dropout(p=dropout)
        # nn.Linear(4096, 4096)
        fc2 = self.network.add_fully_connected(input=relu6.get_output(0), num_outputs=4096,
                                               kernel=self.wm["classifier.4.weight"], bias=self.wm["classifier.4.bias"])
        # nn.ReLU(inplace=True)
        relu7 = self.network.add_activation(
            fc2.get_output(0), type=trt.ActivationType.RELU)
        # nn.Linear(4096, num_classes)
        fc3 = self.network.add_fully_connected(input=relu7.get_output(0), num_outputs=self.cfg["NUM_CLASSES"],
                                               kernel=self.wm["classifier.6.weight"], bias=self.wm["classifier.6.bias"])

        fc3.get_output(0).name = self.cfg["OUTPUT_NAME"]
        self.network.mark_output(fc3.get_output(0))

        return self.network
