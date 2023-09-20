import torch
import numpy as np
import cv2


class Instances:

    def __init__(self, image, bboxes, labels, masks=None, normalized=False):
        self._image = image
        self._bboxes = bboxes
        self._labels = labels
        self._masks = masks
        self._normalized = normalized

    def normalize(self):
        pass


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class Bbox:
    """
    Args:
        data: numpy | tensor  N*4*2
        format: ["xyxy", "xywh"]
    """

    def __init__(self, data, format='xyxy'):
        self.data = data
        self._format = format

    def convert(self, format):
        assert format in ["xyxy", "xywh"]
        if self._format == format:
            return
        elif self._format == "xyxy":
            bboxes = xyxy2xywh(self.data)
        else:
            bboxes = xywh2xyxy(self.data)
        self.data = bboxes
        self._format = format

    @property
    def area(self):
        self.convert("xyxy")
        return (self.data[:, 2]-self.data[:, 0])*(self.data[:, 3]-self.data[:, 1])

    def normalized(self, w, h):
        self.bboxes[:, 0::2] /= w
        self.bboxes[:, 1::2] /= h


class Mask:
    """
    Args: 
        mask: xyxy [(x,y),(x,y),...]
              xxyy [(x,x,x...),(y,y,y)]
    """

    def __init__(self, data, format='xyxy'):
        self.data = data
        assert format in ["xyxy", "xxyy"]
        self._format = format


class ImageX:
    def __init__(self, path):
        self.data = cv2.imread(path)
        self.shape = self.data.shape
        self.file = path

    def _to_numpy(self, im):
        return np.array(im, dtype=np.float32)

    def _to_tensor(self, im):
        return torch.from_numpy(self._to_numpy(im))


class Label:
    def __init__(self, data, mapper_dict=None):
        self.data = data
        self.mapper_dict = mapper_dict

    def mapper(self):
        if self.mapper_dict:
            return [self.mapper_dict(item) for item in self.labels]
