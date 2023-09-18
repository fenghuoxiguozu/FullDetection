import time
import torch
from data.process import pre_prpcess, post_process
from backbone import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight_path = "weights/alexnet.pth"
    img_path = "/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/code/mmpretrain/data/dog/test/n02089867/n0208986700000003.jpg"

    numpy_im = pre_prpcess(img_path)

    # # create model
    # model = build_resnet50_backbone(
    #     num_classes=1000, include_top=True).to(device)
    model = AlexNet(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        start = time.time()
        for i in range(1000):
            output = model(torch.from_numpy(numpy_im).to(device))
            label, prob = post_process(output)

        end = time.time()
        print("cost:{:.3}s pred class:{} pred prob:{:.3}".format(
            (end-start), label, prob))
        print("cost:{:.3}s".format((end-start)))


if __name__ == '__main__':
    main()
