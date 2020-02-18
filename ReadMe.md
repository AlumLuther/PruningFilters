# Pruning Filters For Efficient ConvNets

### 使用说明：

### Train the origin VGG16_bn model

```
--train-flag --save-path ./trained_models --epoch 150 --lr 0.1 --lr-milestone 50 100
```

搭建VGG_16bn模型，并基于CIFAR10进行训练。

### Prune the model

```
--prune-flag --load-path ./trained_models/model.pth --save-path ./prunned_models --prune-layers conv1 conv8 conv9 conv10 conv11 conv12 conv13 --prune-channels 32 256 256 256 256 256 256
```

显式指出需要剪枝的卷积层和通道数，可以加入 --independent-prune-flag 以执行独立策略剪枝。以上是论文作者基于VGG16_bn给出的剪枝参数，也即第一和后六个卷积层剪掉一半的filter。

### Retrain the prunned model

```
--train-flag --load-path ./prunned_models/model.pth --save-path ./trained_prunned_models --epoch 20 --lr 0.001
```

剪枝后再训练应当显式给出适当的学习率。

### Test the model

```
--test-flag --load-path ./trained_models/model.pth --save-path ./trained_models
```

在不训练的情况下直接对模型进行测试。

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

### After pruning

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

### 参考：

https://github.com/tyui592/Pruning_filters_for_efficient_convnets

基于该框架修改了网络结构、参数等，使得剪枝后的模型结构得以保存和再处理。保留了剪枝的大部分实现。

https://arxiv.org/abs/1608.08710

原论文链接。