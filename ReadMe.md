# Pruning Filters For Efficient ConvNets & Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks

### 目前工作：

实现了VGG16_bn、ResNet的搭建、训练和保存等功能，数据集为CIFAR10。

实现了VGG16_bn的网络剪枝，算法基于两篇论文中提出的hard filter pruning(greedy/independent)和soft filter pruning。剪枝后的网络模型得以保存，支持后续读取再训练、进一步剪枝等操作。

实现了ResNet的hard filter pruning，不涉及shortcut的剪枝。

### 有待完成：

ResNet的soft filter pruning；bottleneck模块的ResNet搭建、剪枝；跨shortcut的剪枝。

剪枝参数的进一步细化调整，如为soft filter pruning添加各卷积层独立的剪枝参数。

### 使用说明：

### Train the origin VGG16_bn model

```
--train-flag --save-path ./trained_models/vgg.pth --epoch 300 --lr 0.1 --lr-milestone 100 200
```

无读取路径时默认新搭建一个VGG_16bn模型，并基于CIFAR10进行训练。

### Prune the model ( hard filter pruning )

```
--hard-prune-flag --load-path ./trained_models/vgg.pth --save-path ./prunned_models/vgg.pth --prune-layers conv1 conv8 conv9 conv10 conv11 conv12 conv13 --prune-channels 32 256 256 256 256 256 256
```

显式指出需要剪枝的卷积层和通道数，可以加入 --independent-prune-flag 以执行独立策略剪枝。以上是论文作者基于VGG16_bn给出的剪枝参数，也即第一和后六个卷积层剪掉一半的filter。

### Retrain the pruned model

```
--train-flag --load-path ./prunned_models/vgg.pth --save-path ./trained_prunned_models/vgg.pth --epoch 20 --lr 0.001
```

剪枝后再训练应当显式给出适当的学习率。

### Test the model

```
--test-flag --load-path ./trained_models/vgg.pth --save-path ./trained_models/vgg.pth
```

在不训练的情况下直接对模型进行测试。

### Prune the model ( soft filter pruning )

```
--soft-prune-flag --load-path ./trained_models/vgg.pth --save-path ./soft_prunned_models/vgg.pth --epoch 50 --lr 0.001 --independent-prune-flag --prune-rate 0.125
```

由于soft filter pruning采用的是全局参数prune_rate，故无需显示指出剪枝的层数和通道数。然而，soft filter pruning需要在训练过程中实现。基于已训练的模型剪枝的结果更好，并且不依赖后续的fine tuning。

### 结果：

### VGG16_bn

|        Model Type         | Average Test Accuracy (Top 1) | Average Test Accuracy (Top 5) | Model Size / Kb |
| :-----------------------: | :---------------------------: | :---------------------------: | :-------------: |
|      Origin VGG16_bn      |            93.9200            |            99.7500            |     131482      |
| After hard filter pruning |            93.9300            |            99.7600            |      90416      |
| After soft filter pruning |            93.9100            |            99.7200            |     116980      |

### ResNet32

|                  Model Type                   | Average Test Accuracy (Top 1) | Average Test Accuracy (Top 5) | Model Size / Kb |
| :-------------------------------------------: | :---------------------------: | :---------------------------: | :-------------: |
|                Origin resnet32                |            93.3400            |            99.7900            |      1855       |
| After hard filter pruning(avoiding shortcut)  |                               |                               |                 |
| After hard filter pruning(including shortcut) |                               |                               |                 |
| After soft filter pruning(avoiding shortcut)  |                               |                               |                 |
| After soft filter pruning(including shortcut) |                               |                               |                 |

### 参考：

https://github.com/tyui592/Pruning_filters_for_efficient_convnets

保留了hard filter pruning的大部分实现。

https://arxiv.org/abs/1608.08710

https://arxiv.org/abs/1808.06866