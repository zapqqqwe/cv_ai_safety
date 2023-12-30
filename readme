# PyTorch卷积神经网络（CNN）训练与评估
这个仓库包含了一个使用PyTorch在CIFAR-10数据集上训练卷积神经网络（CNN）的Python脚本。该脚本定义了一个简单的CNN架构，对其进行训练，并在测试集上评估其性能。此外，使用scikit-learn计算了准确率、精确度、召回率、F1分数和混淆矩阵等指标。

## 设置
要运行该脚本，请确保已安装所需的依赖项。您可以使用以下命令进行安装：

bash
Copy code
pip install torch torchvision scikit-learn
代码概述
1. 数据加载与预处理
脚本加载CIFAR-10数据集并使用PyTorch的transforms.Compose应用标准图像变换。将训练集和测试集加载到相应的DataLoader实例中。

2. CNN模型定义
使用PyTorch的nn.Module类定义了CNN模型。该架构包括两个卷积层，随后是最大池化，以及两个全连接层。使用交叉熵损失和随机梯度下降（SGD）作为优化器来训练模型。

3. 训练CNN模型
然后，脚本在训练集上对CNN模型进行了10个时期的训练。它打印每个时期的平均损失。

4. 测试CNN模型
在训练后，脚本在测试集上评估模型并打印测试集上的准确率。

5. 额外指标
脚本进一步使用scikit-learn计算额外的指标，包括精确度、召回率、F1分数和混淆矩阵。这些指标提供了对模型性能更详细的评估。

## 运行脚本
在Python环境中执行脚本以训练和评估CNN模型。根据需要调整超参数或模型架构。

bash
Copy code
python script_name.py
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

这些超参数的调整可能会对模型的性能产生显著影响。您可以尝试不同的值，观察模型的训练和测试性能，以找到最优的超参数组合。建议通过实验和验证集上的性能监控来确定最佳设置，以提高模型的泛化性。

Files already downloaded and verified
lr 0.01
Epoch 1, Loss: 2.138725309573171
Epoch 2, Loss: 1.7744912732287745
Epoch 3, Loss: 1.543716507189719
Epoch 4, Loss: 1.4201168763972913
Epoch 5, Loss: 1.3415962442412706
Epoch 6, Loss: 1.2761967685978737
Epoch 7, Loss: 1.218720068635843
Epoch 8, Loss: 1.1676750921684762
Epoch 9, Loss: 1.1210428968719814
Epoch 10, Loss: 1.0777256811976128
Finished Training
Files already downloaded and verified
Accuracy on the test set: 58.91%
Precision on the test set: 63.10%
Recall on the test set: 58.91%
F1 score on the test set: 58.91%
Confusion Matrix:
[[711   4 119  21   8   5  12   9  98  13]
 [ 97 549  56  20   5   3  14  21 111 124]
 [ 64   1 676  63  45  28  49  50  17   7]
 [ 20   2 224 498  41  56  66  65  21   7]
 [ 39   1 280  66 397  16  75 105  19   2]
 [ 14   1 241 263  28 299  30 106  14   4]
 [  9   2 155  72  28   4 698  20   9   3]
 [ 26   0  88  63  48  27   9 722   5  12]
 [133  17  30  25   5   6   5   7 757  15]
 [ 90  70  46  34   5   3  23  56  89 584]]


##分析

lr=0.1 bathsize=64

Files already downloaded and verified
Epoch 1, Loss: 1.6269771139640028
Epoch 2, Loss: 1.1924263021677657
Epoch 3, Loss: 0.9894488785425415
Epoch 4, Loss: 0.8528684525538588
Epoch 5, Loss: 0.7336303130592532
Epoch 6, Loss: 0.6295550381359847
Epoch 7, Loss: 0.5363457865841553
Epoch 8, Loss: 0.44856758852062933
Epoch 9, Loss: 0.366260273682187
Epoch 10, Loss: 0.28958724772610017
Finished Training
Files already downloaded and verified
Accuracy on the test set: 70.23%
Precision on the test set: 70.64%
Recall on the test set: 70.23%
F1 score on the test set: 69.86%
Confusion Matrix:
[[727  22  42  11   9  16  14   7 117  35]
 [ 19 831   4   1   2  11  11   1  28  92]
 [ 75   8 559  45  74 108  76  16  23  16]
 [ 29  10  56 393  71 286  90  28  18  19]
 [ 30   6  59  51 623  68  75  53  22  13]
 [ 19   6  37  74  40 737  35  28  18   6]
 [ 11   2  36  44  23  45 817   4  12   6]
 [ 21   9  27  31  43 124  11 695   7  32]
 [ 38  41   6   7   4  11   8   3 847  35]
 [ 38  71   5  12   7  16   7  10  40 794]]
 
 lr=0.05

 Epoch 1, Loss: 1.726372512405181
Epoch 2, Loss: 1.3194019224332727
Epoch 3, Loss: 1.1305484395197896
Epoch 4, Loss: 0.9987178768038445
Epoch 5, Loss: 0.8938771508386373
Epoch 6, Loss: 0.8062651825073125
Epoch 7, Loss: 0.7262052972908215
Epoch 8, Loss: 0.6530594522004847
Epoch 9, Loss: 0.5827864234900231
Epoch 10, Loss: 0.5138129132521122
Finished Training
Files already downloaded and verified
Accuracy on the test set: 67.89%
Precision on the test set: 71.88%
Recall on the test set: 67.89%
F1 score on the test set: 67.17%
Confusion Matrix:
[[707  13  45  38  12   0  13   5  87  80]
 [ 10 722   6  10   2   0  14   5  37 194]
 [ 56   3 615 120  51   6  73  35  21  20]
 [ 14   9  74 692  34  14  73  34  20  36]
 [ 16   1 139 102 556   3  93  56  20  14]
 [ 17   3  82 473  32 237  38  82  16  20]
 [  6   2  42  87  13   0 826   6  11   7]
 [  8   5  43  90  51   3  12 743   5  40]
 [ 54  24   5  18   1   0  10   6 817  65]
 [ 13  46  10  20   2   0  10   9  16 874]]
 
 lr=0.5

 Files already downloaded and verified
Epoch 1, Loss: 1.81102337084158
Epoch 2, Loss: 1.318377030932385
Epoch 3, Loss: 1.1105888274776967
Epoch 4, Loss: 0.9762614018014629
Epoch 5, Loss: 0.8879922033499574
Epoch 6, Loss: 0.7951730337289288
Epoch 7, Loss: 0.7610655362572512
Epoch 8, Loss: 0.7268951271882143
Epoch 9, Loss: 0.6717774431075891
Epoch 10, Loss: 0.6684638136983527
Finished Training
Files already downloaded and verified
Accuracy on the test set: 60.69%
Precision on the test set: 62.22%
Recall on the test set: 60.69%
F1 score on the test set: 60.66%
Confusion Matrix:
[[620  19  65  60  25  22  20  42  81  46]
 [ 26 716  12  36   5   8  32  12  38 115]
 [ 53   1 371 176 103  64 158  47  13  14]
 [ 23   5  39 481  71 136 165  50   8  22]
 [ 17   2  47 116 535  40 129  98  11   5]
 [ 16   2  37 240  59 425 128  75  10   8]
 [  3   0  26  52  54  12 835  11   4   3]
 [ 13   1  23  78  74  49  40 694   8  20]
 [ 85  22  19  44  22   8  22  16 708  54]
 [ 44  78  13  53  16  18  27  34  33 684]]


lr=0.1 ,epoch 5

Files already downloaded and verified
Epoch 1, Loss: 1.6582871728845874
Epoch 2, Loss: 1.1985968246179468
Epoch 3, Loss: 0.9989825166247385
Epoch 4, Loss: 0.8640268469405601
Epoch 5, Loss: 0.7418744219538501
Finished Training
Files already downloaded and verified
Accuracy on the test set: 68.78%
Precision on the test set: 70.61%
Recall on the test set: 68.78%
F1 score on the test set: 68.96%
Confusion Matrix:
[[797  10  44  23  14   4   3   4  69  32]
 [ 34 808  12  16   4   6   4   1  33  82]
 [ 80   5 502  73 200  76  33  12   8  11]
 [ 29  11  58 545 125 134  43  15  17  23]
 [ 28   4  38  45 792  33  21  25   9   5]
 [ 25   5  50 209  88 576  12  21   4  10]
 [ 14   9  44  76 127  31 676   8  10   5]
 [ 34   5  32  68 157  71   3 606   1  23]
 [ 82  33  13  26  11   6   1   3 793  32]
 [ 40  80  10  22  11   7   3   9  35 783]]



##lr=0.1 ,epoch 20



Files already downloaded and verified
Epoch 1, Loss: 1.6215357704235769
Epoch 2, Loss: 1.1773506382389751
Epoch 3, Loss: 0.9717653827441622
Epoch 4, Loss: 0.831327083432461
Epoch 5, Loss: 0.7209114431191588
Epoch 6, Loss: 0.617999032215999
Epoch 7, Loss: 0.5195568759956628
Epoch 8, Loss: 0.430397050917301
Epoch 9, Loss: 0.3562420044484956
Epoch 10, Loss: 0.2810650098011317
Epoch 11, Loss: 0.2246953872892329
Epoch 12, Loss: 0.17753198397967518
Epoch 13, Loss: 0.13648516936775515
Epoch 14, Loss: 0.12379957819500428
Epoch 15, Loss: 0.12120799542835835
Epoch 16, Loss: 0.10732370903810767
Epoch 17, Loss: 0.08287677331053464
Epoch 18, Loss: 0.06534558755424245
Epoch 19, Loss: 0.0516614479080076
Epoch 20, Loss: 0.03747949934706849
Finished Training
Files already downloaded and verified
Accuracy on the test set: 70.62%
Precision on the test set: 70.84%
Recall on the test set: 70.62%
F1 score on the test set: 70.54%
Confusion Matrix:
[[738  23  46  15  23   8   8  16  64  59]
 [ 19 805   8  10   5   5   7   4  22 115]
 [ 65  13 531  74 105  78  46  58  15  15]
 [ 29   9  48 519  79 188  40  54  14  20]
 [ 22   5  42  59 705  51  30  69   8   9]
 [ 13   5  41 149  54 619  18  75   5  21]
 [ 10   8  40  74  60  51 724  12   9  12]
 [ 11   3  17  29  77  51   2 782   6  22]
 [ 53  31  11  23   5  10   4  10 816  37]
 [ 24  64   6  11   8  10   4  21  29 823]]


##lr=0.1 ,epoch 50
Files already downloaded and verified
Epoch 1, Loss: 1.6162702201882286
Epoch 2, Loss: 1.1967372956788143
Epoch 3, Loss: 0.994237590171492
Epoch 4, Loss: 0.8568409489624945
Epoch 5, Loss: 0.742854357451734
Epoch 6, Loss: 0.6373279082119617
Epoch 7, Loss: 0.540562494674607
Epoch 8, Loss: 0.4500422532029469
Epoch 9, Loss: 0.370262400097097
Epoch 10, Loss: 0.29649449615260526
Epoch 11, Loss: 0.23738111635608136
Epoch 12, Loss: 0.1808962404699353
Epoch 13, Loss: 0.1639015533082435
Epoch 14, Loss: 0.13352835006402125
Epoch 15, Loss: 0.10997621556672522
Epoch 16, Loss: 0.09636374093387323
Epoch 17, Loss: 0.08509326830644474
Epoch 18, Loss: 0.07867398679904315
Epoch 19, Loss: 0.07419746693746541
Epoch 20, Loss: 0.07242641365096031
Epoch 21, Loss: 0.06293314654806444
Epoch 22, Loss: 0.04762741702738578
Epoch 23, Loss: 0.05620943638183477
Epoch 24, Loss: 0.0560364640288679
Epoch 25, Loss: 0.04017131559802648
Epoch 26, Loss: 0.036561378190229005
Epoch 27, Loss: 0.02623864750844835
Epoch 28, Loss: 0.03516135505355044
Epoch 29, Loss: 0.04024186025774392
Epoch 30, Loss: 0.035917685499567094
Epoch 31, Loss: 0.05630761318188935
Epoch 32, Loss: 0.04741305533578903
Epoch 33, Loss: 0.03450998875716388
Epoch 34, Loss: 0.026101152849783513
Epoch 35, Loss: 0.04043268449376184
Epoch 36, Loss: 0.05614453680241508
Epoch 37, Loss: 0.04552093629957587
Epoch 38, Loss: 0.05625817607628995
Epoch 39, Loss: 0.051276947508248696
Epoch 40, Loss: 0.05116312902783552
Epoch 41, Loss: 0.0506434013921423
Epoch 42, Loss: 0.03950540525150773
Epoch 43, Loss: 0.028058243477494856
Epoch 44, Loss: 0.025465805387768315
Epoch 45, Loss: 0.015670493431875945
Epoch 46, Loss: 0.009212030935217013
Epoch 47, Loss: 0.010866385460604944
Epoch 48, Loss: 0.0023385185289496297
Epoch 49, Loss: 0.00031057698314286087
Epoch 50, Loss: 0.00013414931240010594
Finished Training
Files already downloaded and verified
Accuracy on the test set: 70.45%
Precision on the test set: 70.45%
Recall on the test set: 70.45%
F1 score on the test set: 70.43%
Confusion Matrix:
[[760  22  51  22  17   8   7  10  65  38]
 [ 19 811  10  11   9   5  12   4  29  90]
 [ 66  13 597  70  74  60  58  33  11  18]
 [ 29  11  71 531  56 166  59  45  10  22]
 [ 26   1  76  73 643  44  42  75  11   9]
 [ 18   5  58 164  48 610  26  55   8   8]
 [ 13   9  57  59  37  20 777  11  10   7]
 [ 32   4  33  41  53  71  11 736   1  18]
 [ 67  39  11  15   5  10   7   8 800  38]
 [ 35  85  19  18   7  12   5  14  25 780]]

lr=0.1,epoch=100
Files already downloaded and verified
Epoch 1, Loss: 1.6246517764028077
Epoch 2, Loss: 1.1911767638857713
Epoch 3, Loss: 0.9922209399587968
Epoch 4, Loss: 0.8551519284467868
Epoch 5, Loss: 0.7372988517708181
Epoch 6, Loss: 0.6369013413215232
Epoch 7, Loss: 0.5423925443912101
Epoch 8, Loss: 0.454932041656788
Epoch 9, Loss: 0.3732094561200008
Epoch 10, Loss: 0.3031567393151848
Epoch 11, Loss: 0.23625286984378877
Epoch 12, Loss: 0.17990688321268772
Epoch 13, Loss: 0.1606298317213345
Epoch 14, Loss: 0.14619871279191407
Epoch 15, Loss: 0.10748118151912985
Epoch 16, Loss: 0.08878044466085522
Epoch 17, Loss: 0.06982871670576046
Epoch 18, Loss: 0.05447813521509709
Epoch 19, Loss: 0.05985416142273542
Epoch 20, Loss: 0.06234366760399821
Epoch 21, Loss: 0.07981972383154208
Epoch 22, Loss: 0.07982710180262251
Epoch 23, Loss: 0.05857980263430168
Epoch 24, Loss: 0.04598563117370524
Epoch 25, Loss: 0.05627292445238532
Epoch 26, Loss: 0.06127439043305986
Epoch 27, Loss: 0.0631565304739964
Epoch 28, Loss: 0.03792494762083873
Epoch 29, Loss: 0.041078911583876304
Epoch 30, Loss: 0.02024353537554442
Epoch 31, Loss: 0.010599997433000262
Epoch 32, Loss: 0.0023240355423876724
Epoch 33, Loss: 0.00042794782593610735
Epoch 34, Loss: 0.0002574766077879963
Epoch 35, Loss: 0.00021104447739937006
Epoch 36, Loss: 0.00018301328226817017
Epoch 37, Loss: 0.00016330080361536215
Epoch 38, Loss: 0.00014772698824528602
Epoch 39, Loss: 0.00013559397144336983
Epoch 40, Loss: 0.00012551783148966253
Epoch 41, Loss: 0.00011753574610380415
Epoch 42, Loss: 0.00010997948156425554
Epoch 43, Loss: 0.00010368489250633683
Epoch 44, Loss: 9.804815815589296e-05
Epoch 45, Loss: 9.33982220613659e-05
Epoch 46, Loss: 8.87656142995672e-05
Epoch 47, Loss: 8.481784865303463e-05
Epoch 48, Loss: 8.12522233228146e-05
Epoch 49, Loss: 7.803084770359391e-05
Epoch 50, Loss: 7.504286646412894e-05
Epoch 51, Loss: 7.242014336948054e-05
Epoch 52, Loss: 6.967587116412727e-05
Epoch 53, Loss: 6.731986357587567e-05
Epoch 54, Loss: 6.5177248626658e-05
Epoch 55, Loss: 6.31858007406484e-05
Epoch 56, Loss: 6.11830426623085e-05
Epoch 57, Loss: 5.949540813478803e-05
Epoch 58, Loss: 5.775751989795887e-05
Epoch 59, Loss: 5.614006986527149e-05
Epoch 60, Loss: 5.4711792592891416e-05
Epoch 61, Loss: 5.323880061595169e-05
Epoch 62, Loss: 5.190370057042881e-05
Epoch 63, Loss: 5.062823764806277e-05
Epoch 64, Loss: 4.948968603225008e-05
Epoch 65, Loss: 4.829532129835449e-05
Epoch 66, Loss: 4.73042346525092e-05
Epoch 67, Loss: 4.615747065713695e-05
Epoch 68, Loss: 4.5221205338668154e-05
Epoch 69, Loss: 4.422634148151301e-05
Epoch 70, Loss: 4.3316722920983354e-05
Epoch 71, Loss: 4.254193540315272e-05
Epoch 72, Loss: 4.163168450596433e-05
Epoch 73, Loss: 4.097494023278701e-05
Epoch 74, Loss: 4.007783220336686e-05
Epoch 75, Loss: 3.93096724935539e-05
Epoch 76, Loss: 3.859252288821665e-05
Epoch 77, Loss: 3.797436025198455e-05
Epoch 78, Loss: 3.7264992554012465e-05
Epoch 79, Loss: 3.668966533064645e-05
Epoch 80, Loss: 3.6027281977482426e-05
Epoch 81, Loss: 3.5425367203868035e-05
Epoch 82, Loss: 3.48520788902603e-05
Epoch 83, Loss: 3.43071734583934e-05
Epoch 84, Loss: 3.3730128743400784e-05
Epoch 85, Loss: 3.324175522278953e-05
Epoch 86, Loss: 3.272384171042407e-05
Epoch 87, Loss: 3.22249165467389e-05
Epoch 88, Loss: 3.180593453098923e-05
Epoch 89, Loss: 3.130347585477971e-05
Epoch 90, Loss: 3.088459508253808e-05
Epoch 91, Loss: 3.0399443408891098e-05
Epoch 92, Loss: 2.9982182588676497e-05
Epoch 93, Loss: 2.9588234441924497e-05
Epoch 94, Loss: 2.919816084175093e-05
Epoch 95, Loss: 2.8792852474935618e-05
Epoch 96, Loss: 2.8421700431698276e-05
Epoch 97, Loss: 2.805917150399089e-05
Epoch 98, Loss: 2.7712625456107904e-05
Epoch 99, Loss: 2.7351195896403797e-05
Epoch 100, Loss: 2.7011607318128922e-05
Finished Training
Files already downloaded and verified
Accuracy on the test set: 72.09%
Precision on the test set: 72.03%
Recall on the test set: 72.09%
F1 score on the test set: 72.05%
Confusion Matrix:
[[786  17  49  23  17   9  10  10  54  25]
 [ 20 840   6   7   3   3   8   6  29  78]
 [ 56   8 619  66  78  65  52  31  11  14]
 [ 26  11  74 519  68 178  55  34  15  20]
 [ 15   4  75  87 658  45  34  64  15   3]
 [ 20   7  52 178  49 598  20  57   9  10]
 [  4   6  44  55  39  31 797   6   9   9]
 [ 13   1  28  33  57  63   9 770   5  21]
 [ 57  41  16  11   7   7   6   3 832  20]
 [ 32  75  11  17   7   5  10  16  37 790]]

lr=0.1,epoch=20 ,batch_size=8 
Files already downloaded and verified
Epoch 1, Loss: 1.4749188236570359
Epoch 2, Loss: 1.2010872246336937
Epoch 3, Loss: 1.1085276967656612
Epoch 4, Loss: 1.0553572822341324
Epoch 5, Loss: 1.0097942295650393
Epoch 6, Loss: 1.0131440028712153
Epoch 7, Loss: 0.9935795693805813
Epoch 8, Loss: 1.004803787701428
Epoch 9, Loss: 1.014912204418853
Epoch 10, Loss: 1.0460888545903564
Epoch 11, Loss: 1.1022168888149038
Epoch 12, Loss: 1.110869554005228
Epoch 13, Loss: 1.1468604897685721
Epoch 14, Loss: 1.2572872441217304
Epoch 15, Loss: 1.3176158723652363
Epoch 16, Loss: 1.3880911533899232
Epoch 17, Loss: 1.4474589259192348
Epoch 18, Loss: 1.6034626783040167
Epoch 19, Loss: 1.9464148384445905
Epoch 20, Loss: 2.1484280881261824
Finished Training
Files already downloaded and verified
/home/lichenglin/anaconda3/envs/myenv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy on the test set: 10.08%
Precision on the test set: 21.00%
Recall on the test set: 10.08%
F1 score on the test set: 1.98%
Confusion Matrix:
[[   0 1000    0    0    0    0    0    0    0    0]
 [   0 1000    0    0    0    0    0    0    0    0]
 [   0  995    2    1    2    0    0    0    0    0]
 [   0 1000    0    0    0    0    0    0    0    0]
 [   0  995    0    1    2    0    0    0    2    0]
 [   0  999    0    0    0    0    0    0    1    0]
 [   0  999    0    0    0    0    0    0    1    0]
 [   0 1000    0    0    0    0    0    0    0    0]
 [   0  996    0    0    0    0    0    0    4    0]
 [   0 1000    0    0    0    0    0    0    0    0]]
lr=0.1,epoch=20 ,batch_size=16
Files already downloaded and verified
Epoch 1, Loss: 1.4192521164131164
Epoch 2, Loss: 1.0201521772956847
Epoch 3, Loss: 0.8469594735813141
Epoch 4, Loss: 0.723952656648159
Epoch 5, Loss: 0.6288715781378746
Epoch 6, Loss: 0.5541277875959874
Epoch 7, Loss: 0.48704436941862106
Epoch 8, Loss: 0.4537885733240843
Epoch 9, Loss: 0.42323101508401334
Epoch 10, Loss: 0.41645028406880796
Epoch 11, Loss: 0.4039421340768784
Epoch 12, Loss: 0.395284806842953
Epoch 13, Loss: 0.3844089791187644
Epoch 14, Loss: 0.4028886750527471
Epoch 15, Loss: 0.40010055971691383
Epoch 16, Loss: 0.4050274316376727
Epoch 17, Loss: 0.4457801815317804
Epoch 18, Loss: 0.4759032438350236
Epoch 19, Loss: 0.4365119897894934
Epoch 20, Loss: 0.4700709429378435
Finished Training
Files already downloaded and verified
Accuracy on the test set: 63.00%
Precision on the test set: 64.93%
Recall on the test set: 63.00%
F1 score on the test set: 63.23%
Confusion Matrix:
[[626  31  62  60  47  12  19  15  68  60]
 [ 17 763  14  24  11  12  16   7  30 106]
 [ 46   6 454 141 158  53  91  32   6  13]
 [ 18  16  62 567 111  94  72  38  11  11]
 [ 20   4  38 103 687  33  66  40   3   6]
 [  7   8  77 297  97 394  48  46   8  18]
 [  9   6  28 118  59  21 743  10   3   3]
 [ 12   6  31 104 122  31  26 640   2  26]
 [ 74  49  20  46  20   4  21   8 695  63]
 [ 28  96  14  34  20  13  27  14  23 731]]
lr=0.1,epoch=20 ,batch_size=32

Files already downloaded and verified
Epoch 1, Loss: 1.5019734164345975
Epoch 2, Loss: 1.0586511067526507
Epoch 3, Loss: 0.86132586227345
Epoch 4, Loss: 0.7216732748429591
Epoch 5, Loss: 0.5997561924898388
Epoch 6, Loss: 0.49527358443441105
Epoch 7, Loss: 0.40201855766433825
Epoch 8, Loss: 0.3265152188114493
Epoch 9, Loss: 0.26902827159075576
Epoch 10, Loss: 0.24484619035630445
Epoch 11, Loss: 0.1995447008981967
Epoch 12, Loss: 0.19129709752740712
Epoch 13, Loss: 0.18234890902454245
Epoch 14, Loss: 0.15592138964092325
Epoch 15, Loss: 0.1593879846914392
Epoch 16, Loss: 0.1547555544208652
Epoch 17, Loss: 0.13353631196824967
Epoch 18, Loss: 0.13430296794562918
Epoch 19, Loss: 0.12708030166632514
Epoch 20, Loss: 0.12846399713215262
Finished Training
Files already downloaded and verified
Accuracy on the test set: 68.27%
Precision on the test set: 68.28%
Recall on the test set: 68.27%
F1 score on the test set: 68.02%
Confusion Matrix:
[[759  29  32  24  20  18   8  15  44  51]
 [ 25 799   4  13   4   8   7   7  24 109]
 [ 81   7 529  64  78  88  82  35  19  17]
 [ 24  15  56 434  74 227  76  46  12  36]
 [ 37   5  64  52 591  63  75  80  12  21]
 [ 18   3  39 138  39 635  39  60   7  22]
 [  6   6  31  55  38  42 785   7  13  17]
 [ 22   4  20  48  46  69  14 732   8  37]
 [109  34   7  12   7  14   6  11 755  45]
 [ 28  72  11  15   8   7   7  17  27 808]]


lr=0.1,epoch=20 ,batch_size=128
Files already downloaded and verified
Epoch 1, Loss: 1.7785391624626297
Epoch 2, Loss: 1.3644537454675836
Epoch 3, Loss: 1.1711567993968954
Epoch 4, Loss: 1.0259068067116506
Epoch 5, Loss: 0.9221219754280032
Epoch 6, Loss: 0.8273851816611522
Epoch 7, Loss: 0.7489693481903856
Epoch 8, Loss: 0.6724961743787732
Epoch 9, Loss: 0.6013906976908369
Epoch 10, Loss: 0.5382925380983621
Epoch 11, Loss: 0.4714050201503822
Epoch 12, Loss: 0.41124689739073633
Epoch 13, Loss: 0.3566831312597255
Epoch 14, Loss: 0.299079967924701
Epoch 15, Loss: 0.24186394064475203
Epoch 16, Loss: 0.2020467802729753
Epoch 17, Loss: 0.16650550548568407
Epoch 18, Loss: 0.12946652038894652
Epoch 19, Loss: 0.10825263403470406
Epoch 20, Loss: 0.06859141862129464
Finished Training
Files already downloaded and verified
Accuracy on the test set: 69.91%
Precision on the test set: 70.47%
Recall on the test set: 69.91%
F1 score on the test set: 69.94%
Confusion Matrix:
[[758  28  50  28  33   0  13   9  49  32]
 [ 16 829  16  14   8   1  10   3  29  74]
 [ 64   9 599  88  75  46  54  40  14  11]
 [ 14  15  90 575  75  89  59  55  14  14]
 [ 19   4  88  79 652  20  44  83   8   3]
 [ 14   5  66 251  54 486  32  73   8  11]
 [  6   6  49  90  30  19 775   9   5  11]
 [ 10   9  31  49  67  35   5 780   1  13]
 [ 65  56  14  31  11   9   3   5 778  28]
 [ 30  97  10  25   9   9   9  23  29 759]]

lr=0.1,epoch=20 ,batch_size=512
Files already downloaded and verified
Epoch 1, Loss: 2.098127479455909
Epoch 2, Loss: 1.798296188821598
Epoch 3, Loss: 1.6221810457657795
Epoch 4, Loss: 1.4999080409809036
Epoch 5, Loss: 1.4243747725778697
Epoch 6, Loss: 1.3346345704429003
Epoch 7, Loss: 1.270900924595035
Epoch 8, Loss: 1.2119513214850912
Epoch 9, Loss: 1.15703869960746
Epoch 10, Loss: 1.115263893288009
Epoch 11, Loss: 1.071386135354334
Epoch 12, Loss: 1.0254353011141017
Epoch 13, Loss: 0.9822823399183701
Epoch 14, Loss: 0.9525019441332135
Epoch 15, Loss: 0.9185886936528342
Epoch 16, Loss: 0.8818261410508837
Epoch 17, Loss: 0.8592700727131902
Epoch 18, Loss: 0.8260575678883767
Epoch 19, Loss: 0.7953897781518041
Epoch 20, Loss: 0.7686145062349281
Finished Training
Files already downloaded and verified
Accuracy on the test set: 67.63%
Precision on the test set: 69.29%
Recall on the test set: 67.63%
F1 score on the test set: 67.91%
Confusion Matrix:
[[700  28  51  36  27  10   3  20  74  51]
 [ 14 776  12  20   6  13   7   9  24 119]
 [ 45  13 483  95 116 140  28  55   8  17]
 [ 10   7  48 511  53 263  31  46   9  22]
 [ 18   3  51  71 625  96  19  99  17   1]
 [ 14   2  29 147  37 709   4  45   4   9]
 [  8   7  41 109  78  85 649  11   4   8]
 [  9   3  17  43  48 110   2 752   3  13]
 [ 57  48  11  25  12  15   2  13 771  46]
 [ 23  85   4  31   5  23   2  19  21 787]]


