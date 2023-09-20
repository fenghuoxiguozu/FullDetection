from data.transforms import *
from structures.instance import *
from visualize.instance_draw import Vis


img_path = r"/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/code/yolov5/datasets/VOC/images/train/000244.jpg"
label_path = img_path.replace("images", "labels").replace("jpg", "txt")

with open(label_path, 'r') as f:
    txts = f.readlines()

bboxes, labels = [], []
for item in txts:
    result = item.replace('\n', '').split(' ')
    label, masks = result[0], result[1:]
    bboxes.append([float(mask) for mask in masks])
    labels.append(int(label))


instance = Instances(image=ImageX(path=img_path), bboxes=Bbox(data=np.array(
    bboxes, dtype=np.float32), format='xywh'), labels=Label(data=np.array(labels, dtype=np.float32)))

Vis("out").draw_rectangle(instance)
print(instance._bboxes.data)
print(instance._image.data.shape)

tf = VFilp(p=1)
tf(instance)

Vis("out").draw_rectangle(instance)

print(instance._bboxes.data)
print(instance._image.data.shape)
