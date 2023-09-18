## deep learning
> contains(classify model,object detection,tensorrt)

### models
- 已完成pytorch模型

```
from backbone.build_trt import TRT_BACKBONE_REGISTRY
print(TRT_BACKBONE_REGISTRY)
```

- 已完成tensorrt模型
```
from backbone.build_trt import TRT_BACKBONE_REGISTRY
print(TRT_BACKBONE_REGISTRY)
```


### speed compare
| BACKBONE  | API  | torch time(ms) | trt time(ms) | map  |
| --------- | :--: | :------------: | ------------ | ---- |
| alexnet   |  √   |      4.54      |      1.66    |      |
| resnet18  |  √   |                |              |      |
| resnet34  |  √   |                |              |      |
| resnet50  |  √   |      12.7      |      3.78    |      |
| resnet100 |  √   |                |              |      |
|           |      |                |              |      |
|           |      |                |              |      |
|           |      |                |              |      |
|           |      |                |              |      |

