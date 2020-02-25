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

### Retrain the prunned model

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

### Before Pruning

```
Epoch 295. [Train] Top1: 99.9800, Top5: 100.0000, Loss: 0.0013. [Test] Top1: 93.8800, Top5: 99.7500, Loss: 0.3060. Time: 00:00:36
Epoch 296. [Train] Top1: 99.9860, Top5: 100.0000, Loss: 0.0012. [Test] Top1: 93.8300, Top5: 99.7300, Loss: 0.3074. Time: 00:00:36
Epoch 297. [Train] Top1: 99.9700, Top5: 100.0000, Loss: 0.0015. [Test] Top1: 94.0100, Top5: 99.7700, Loss: 0.3062. Time: 00:00:36
Epoch 298. [Train] Top1: 99.9840, Top5: 100.0000, Loss: 0.0012. [Test] Top1: 93.9500, Top5: 99.7400, Loss: 0.3037. Time: 00:00:36
Epoch 299. [Train] Top1: 99.9760, Top5: 100.0000, Loss: 0.0014. [Test] Top1: 93.9300, Top5: 99.7500, Loss: 0.3040. Time: 00:00:36

Average test accuracy: 93.9200
Model size: 131482 Kb
```

### After hard filter pruning

```
Epoch 5. [Train] Top1: 99.9820, Top5: 100.0000, Loss: 0.0019. [Test] Top1: 93.8900, Top5: 99.7500, Loss: 0.2969. Time: 00:00:29
Epoch 6. [Train] Top1: 99.9800, Top5: 100.0000, Loss: 0.0020. [Test] Top1: 93.9700, Top5: 99.7500, Loss: 0.2961. Time: 00:00:29
Epoch 7. [Train] Top1: 99.9740, Top5: 100.0000, Loss: 0.0021. [Test] Top1: 93.9600, Top5: 99.7600, Loss: 0.3008. Time: 00:00:30
Epoch 8. [Train] Top1: 99.9620, Top5: 100.0000, Loss: 0.0024. [Test] Top1: 93.9200, Top5: 99.7500, Loss: 0.3005. Time: 00:00:29
Epoch 9. [Train] Top1: 99.9800, Top5: 100.0000, Loss: 0.0017. [Test] Top1: 93.9100, Top5: 99.7800, Loss: 0.3010. Time: 00:00:29

Average test accuracy: 93.9300
Model size: 90416 Kb
```

模型的空间占用减少了31.23%，浮点运算减少了34%，并且完全没有影响到准确率。

### After soft filter pruning

```
Epoch 45. [Train] Top1: 99.9320, Top5: 100.0000, Loss: 0.0027. [Test] Top1: 93.3100, Top5: 99.7500, Loss: 0.3381. Time: 00:00:37
Epoch 46. [Train] Top1: 99.9300, Top5: 100.0000, Loss: 0.0028. [Test] Top1: 93.4000, Top5: 99.7400, Loss: 0.3395. Time: 00:00:37
Epoch 47. [Train] Top1: 99.9320, Top5: 100.0000, Loss: 0.0030. [Test] Top1: 93.4600, Top5: 99.7300, Loss: 0.3364. Time: 00:00:37
Epoch 48. [Train] Top1: 99.9500, Top5: 100.0000, Loss: 0.0024. [Test] Top1: 93.6200, Top5: 99.7100, Loss: 0.3294. Time: 00:00:37
Epoch 49. [Train] Top1: 99.9480, Top5: 100.0000, Loss: 0.0027. [Test] Top1: 93.3500, Top5: 99.6600, Loss: 0.3411. Time: 00:00:37

Average test accuracy: 93.4280
Model size: 116980 Kb
```

模型的空间占用减少了11.03%，准确率仅有略微下降。将全局剪枝率进行逐层划分可能会得到更好的表现，有待进一步优化。



### 参考：

https://github.com/tyui592/Pruning_filters_for_efficient_convnets

保留了hard filter pruning的大部分实现。

https://arxiv.org/abs/1608.08710

https://arxiv.org/abs/1808.06866