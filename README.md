# PyTorch卷积神经网络（CNN）训练与评估

该仓库包含了一个使用PyTorch在CIFAR-10数据集上训练卷积神经网络（CNN）的Python脚本。脚本定义了一个简单的CNN架构，对其进行训练，并在测试集上评估其性能。此外，使用scikit-learn计算了准确率、精确度、召回率、F1分数和混淆矩阵等指标。

## 设置

要运行该脚本，请确保已安装所需的依赖项。您可以使用以下命令进行安装：

```bash
pip install torch torchvision scikit-learn
```

## 代码概述

1. **数据加载与预处理**

脚本加载CIFAR-10数据集并使用PyTorch的transforms.Compose应用标准图像变换。将训练集和测试集加载到相应的DataLoader实例中。

2. **CNN模型定义**

使用PyTorch的nn.Module类定义了CNN模型。该架构包括两个卷积层，随后是最大池化，以及两个全连接层。使用交叉熵损失和随机梯度下降（SGD）作为优化器来训练模型。

3. **训练CNN模型**

然后，脚本在训练集上对CNN模型进行了10个时期的训练。它打印每个时期的平均损失。

4. **测试CNN模型**

在训练后，脚本在测试集上评估模型并打印测试集上的准确率。

5. **额外指标**

脚本进一步使用scikit-learn计算额外的指标，包括精确度、召回率、F1分数和混淆矩阵。这些指标提供了对模型性能更详细的评估。

## 运行脚本

在Python环境中执行脚本以训练和评估CNN模型。根据需要调整超参数或模型架构。

```bash
python script_name.py
```

请随时修改代码以满足您的特定要求或将其集成到您的机器学习项目中。

## 超参数

在给定的代码中，以下是一些超参数和相关设置：

1. **学习率 (`lr=0.01`)：**
   ```python
   optimizer = optim.SGD(cnn_model.parameters(), lr=0.01)
   ```
   学习率是优化器在更新模型参数时的步长。在这里，学习率被设置为0.01。

2. **训练时期数 (`for epoch in range(10)`)：**
   ```python
   for epoch in range(10):
   ```
   训练时期数决定了模型在整个训练集上迭代的次数。

3. **批量大小 (`batch_size=64`)：**
   ```python
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
   testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
   ```
   批量大小定义了每次优化模型时所使用的样本数。

4. **神经网络架构：**
   ```python
   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(3, 32, 3)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.Conv2d(32, 64, 3)
           self.fc1 = nn.Linear(64 * 6 * 6, 128)
           self.fc2 = nn.Linear(128, 10)
   ```
   这是定义CNN模型的架构，包括卷积层、最大池化层和全连接层等。

这些超参数的调整可能会对模型的性能产生显著影响。详细输出见日志输出，1-3超参数结果分析见output.png
