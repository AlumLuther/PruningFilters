# Pruning Filters For Efficient ConvNets

### 使用说明：

### Train the origin VGG16_bn model:

```
--train-flag --save-path ./trained_models --epoch 150 --lr 0.1 --lr-milestone 50 100
```

搭建VGG_16bn模型，并基于CIFAR10进行训练。

### Prune the model:

```
--prune-flag --load-path ./trained_models/model.pth --save-path ./prunned_models --prune-layers conv1 conv8 conv9 conv10 conv11 conv12 conv13 --prune-channels 32 256 256 256 256 256 256
```

显式指出需要剪枝的卷积层和通道数，可以加入 --independent-prune-flag 以执行独立策略剪枝。以上是论文作者基于VGG16_bn给出的剪枝参数，也即第一和后六个卷积层剪掉一半的filter。

### Retrain the prunned model:

```
--train-flag --load-path ./prunned_models/model.pth --save-path ./trained_prunned_models --epoch 20 --lr 0.001
```

剪枝后再训练应当显式给出适当的学习率。

### Test the model:

```
--test-flag --load-path ./trained_models/model.pth --save-path ./trained_models
```

在不训练的情况下直接对模型进行测试。

### 参考：

https://github.com/tyui592/Pruning_filters_for_efficient_convnets

基于该框架修改了网络结构、参数等，使得剪枝后的模型结构得以保存和再处理。保留了剪枝的大部分实现。

https://arxiv.org/abs/1608.08710

原论文链接。