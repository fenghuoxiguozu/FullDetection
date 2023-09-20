import cv2
import os


COLOR_MAP = ["#2F4F4F", "#008080", "#20B2AA", "#7FFFAA", "#F5FFFA", "#3CB371", "#2E8B57", "#F0FFF0", "#90EE90", "#98FB98", "#8FBC8F", "#32CD32", "#008000", "#ADFF2F", "#556B2F", "#F5F5DC", "#BDB76B", "#FFFACD",
             "#FFD700", "#DAA520", "#F5DEB3", "#FFE4B5", "#D2B48C", "#DEB887", "#CD853F", "#FFDAB9", "#D2691E", "#FFA07A", "#FF7F50", "#FF6347", "#FA8072", "#F08080", "#CD5C5C", "#A52A2A", "#B22222", "#8B0000", "#DCDCDC", "#D3D3D3"]


class Vis:
    def __init__(self, dst):
        self.dst = dst

    def draw(self):
        pass

    def draw_rectangle(self, instance):
        instance._bboxes.convert("xyxy")
        draw_im = instance._image.data.copy()
        h, w = instance._image.shape[:2]
        filename = os.path.basename(instance._image.file)
        bbox = instance._bboxes.data
        for item in bbox:
            x1 = int(item[0]*w)
            y1 = int(item[1]*h)
            x2 = int(item[2]*w)
            y2 = int(item[3]*h)
            print("BBOX:", x1, y1, x2, y2)
            cv2.rectangle(draw_im, (x1, y1), (x2, y2), 2)
        if not os.path.exists(self.dst):
            os.makedirs(self.dst)
        cv2.imwrite(os.path.join(self.dst, filename), draw_im)

    def draw_text(self):
        pass
