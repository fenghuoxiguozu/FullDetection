import os
import torch
from PIL import Image
from torchvision import transforms


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406])])

    # load image
    img_path = "/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/code/mmpretrain/data/dog/test/n02089867/n0208986700000003.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img) 
    img = torch.unsqueeze(img, dim=0) # [N, C, H, W]

    # create model
    model = create_model(num_classes=8, has_logits=False).to(device)
    # load model weights
    weight_path = "weights/resnet50-0676ba61.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

        print("pred class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy()))



if __name__ == '__main__':
    main()
