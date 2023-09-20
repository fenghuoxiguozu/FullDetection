import random
import cv2
import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseTransform(metaclass=ABCMeta):
    def _set_attributes(self, params):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_im(self, im):
        return

    @abstractmethod
    def apply_bbox(self, bbox):
        """
        Args:
            bbox: N*4
            array([[0.09199998, 0.05120483, 0.81200004, 0.9578314],
                    [0.566, 0., 0.676, 0.02710843]], dtype=float32)
        """
        return

    def apply_polygons(self, polygons):
        if len(polygons) == 0:
            return []
        return [self.apply_polygon(np.asarray(p, dtype=int)) for p in polygons]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = f'{self.__class__.__name__}('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class ToTensor(BaseTransform):

    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def apply_im(self, im):
        # HWC to CHW -> BGR to RGB -> contiguous
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im

    def apply_bbox(self, bbox):
        return

    def apply_labels(self, labels):
        return torch.from_numpy(labels)

    def __call__(self, instance):
        tensor_im = self.apply_im(instance._image.data)
        instance._image.data = tensor_im
        tensor_label = self.apply_labels(instance._labels.data)
        instance._labels.data = tensor_label
        return instance


class HFilp(BaseTransform):

    def __init__(self, p=0.5):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        return np.flip(im, axis=1)

    def apply_bbox(self, w, polygon):
        polygon[:, 0] = w - polygon[:, 0]
        polygon[:, 2] = w - polygon[:, 2]
        return polygon

    def __call__(self, instance):
        if random.random() < self.p:
            instance._image.data = self.apply_im(instance._image.data)
            instance._bboxes.data = self.apply_bbox(1, instance._bboxes.data)
            return instance


class VFilp(BaseTransform):

    def __init__(self, p=0.5):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        return np.flip(im, axis=0)

    def apply_bbox(self, w, polygon):
        polygon[:, 1] = w - polygon[:, 1]
        polygon[:, 3] = w - polygon[:, 3]
        return polygon

    def __call__(self, instance):
        if random.random() < self.p:
            instance._image.data = self.apply_im(instance._image.data)
            instance._bboxes.data = self.apply_bbox(1, instance._bboxes.data)
            return instance


class Resize(BaseTransform):

    def __init__(self, imgsz=448):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        self.oh, self.ow = im.shape[:2]
        new_im = cv2.resize(im, (self.imgsz, self.imgsz))
        self.rh, self.rw = self.imgsz/self.oh, self.imgsz/self.ow
        cv2.imwrite("new_im.jpg", new_im)
        return new_im

    def apply_bbox(self, bbox):
        return bbox

    def __call__(self, instance):
        new_im = self.apply_im(instance._image.data)
        instance._image.data = new_im
        instance._image.shape = new_im.shape


class letterBox(BaseTransform):

    def __init__(self, imgsz=448):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        oh, ow = im.shape[:2]
        r = min(self.imgsz / oh, self.imgsz / ow)
        self.resize_h, self.resize_w = round(oh*r), round(ow*r)
        self.pad_h, self.pad_w = self.imgsz-self.resize_h, self.imgsz-self.resize_w
        new_im = cv2.resize(im, (self.resize_w, self.resize_h))
        pad_im = cv2.copyMakeBorder(
            new_im, 0, self.pad_h, 0, self.pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return pad_im

    def apply_bbox(self, bbox):
        bbox[:, 0] = (bbox[:, 0] * self.resize_w)/(self.resize_w+self.pad_w)
        bbox[:, 1] = (bbox[:, 1] * self.resize_h)/(self.resize_h+self.pad_h)
        bbox[:, 2] = (bbox[:, 2] * self.resize_w)/(self.resize_w+self.pad_w)
        bbox[:, 3] = (bbox[:, 3] * self.resize_h)/(self.resize_h+self.pad_h)
        return bbox

    def __call__(self, instance):
        new_im = self.apply_im(instance._image.data)
        instance._image.data = new_im
        instance._image.shape = new_im.shape
        new_bbox = self.apply_bbox(instance._bboxes.data)
        instance._bboxes.data = new_bbox
        return instance


# class letterBox(BaseTransform):

#     def __init__(self, p=0.5):
#         super().__init__()
#         self._set_attributes(locals())

#     def apply_im(self, im):
#         return

#     def apply_bbox(self, bbox):
#         return
