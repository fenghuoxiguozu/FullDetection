import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def pre_prpcess(img_path):
    img = Image.open(img_path)
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406])])
    img = data_transform(img)
    data = torch.unsqueeze(img, dim=0)  # [N, C, H, W]
    data = data.numpy().astype(np.float32)
    return data


def post_process(output):
    if isinstance(output, np.ndarray):
        result = softmax(output)
        label_index = np.argmax(result)
        return label_index, result[label_index]
    elif isinstance(output, torch.Tensor):
        result = torch.softmax(torch.squeeze(output).cpu(), dim=0)
        label_index = torch.argmax(result.cpu()).numpy()
        return label_index.item(), result[label_index].item()
    else:
        raise ValueError(f"不支持的数据格式{type(output)}")
