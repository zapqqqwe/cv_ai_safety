# %%
import numpy as np
import random
import torchvision

# 下载CIFAR-10数据集
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 提取特征和标签
X_train = []
y_train = []

X_test = []
y_test = []

for data in trainset:
    image, label = data
    # 将图像数据展平为特征向量
    feature_vector = image.view(-1).numpy()
    X_train.append(feature_vector)
    y_train.append(label)

for data in testset:
    image, label = data
    feature_vector = image.view(-1).numpy()
    X_test.append(feature_vector)
    y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# 定义SVM训练函数
def train_linear_svm(X, y, learning_rate=0.01, num_epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(1, num_epochs + 1):
        for i, x in enumerate(X):
            condition = y[i] * (np.dot(x, weights) - bias) >= 1
            if condition:
                weights -= learning_rate * (2 / epoch * weights)
            else:
                weights -= learning_rate * (2 / epoch * weights - x * y[i])
                bias -= learning_rate * y[i]

    return weights, bias

# 训练SVM
weights, bias = train_linear_svm(X_train, y_train)

# 预测
def predict(X, weights, bias):
    return np.sign(np.dot(X, weights) - bias)

y_pred = predict(X_test, weights, bias)

# 评估性能
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

accuracy_value = accuracy(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy_value * 100:.2f}%')


# %%

class SVM(object):
    def __init__(self):
        # W为加上偏置的权重（D,num_class)
        self.W = None
    def svm_loss_naive(self, x, y, reg):
        """
        功能：非矢量化版本的损失函数
        输入：
        -x：(numpy array)样本数据（N,D)
        -y：(numpy array)标签（N，）
        -reg：(float)正则化强度
        输出：
        (float)损失函数值loss
        (numpy array)权重梯度dW
        """
        num_train = x.shape[0]
        num_class = self.W.shape[1]
        # 初始化
        loss = 0.0
        dW = np.zeros(self.W.shape)
        for i in range(num_train):
            scores = x[i].dot(self.W)
            # 计算边界,delta=1
            margin = scores - scores[y[i]] + 1
            # 把正确类别的归0
            margin[y[i]] = 0
            for j in range(num_class):
                # max操作
                if j == y[i]:
                    continue
                if margin[j] > 0:
                    loss += margin[j]
                    dW[:, y[i]] += -x[i]
                    dW[:, j] += x[i]
        # 要除以N
        loss /= num_train
        dW /= num_train
        # 加上正则项
        loss += 0.5 * reg * np.sum(self.W * self.W)
        dW += reg * self.W
        return loss, dW
    def svm_loss_vectorized(self, x, y, reg):
        """
        功能：矢量化版本的损失函数
        输入：
        -x：(numpy array)样本数据（N,D)
        -y：(numpy array)标签（N，）
        -reg：(float)正则化强度
        输出：
        (float)损失函数值loss
        (numpy array)权重梯度dW
        """
        loss = 0.0
        dW = np.zeros(self.W.shape)
        num_train = x.shape[0]
        scores = x.dot(self.W)
        margin = scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1
        margin[np.arange(num_train), y] = 0.0
        # max操作
        margin = (margin > 0) * margin
        loss += margin.sum() / num_train
        # 加上正则化项
        loss += 0.5 * reg * np.sum(self.W * self.W)
        # 计算梯度
        margin = (margin > 0) * 1
        row_sum = np.sum(margin, axis=1)
        margin[np.arange(num_train), y] = -row_sum
        dW = x.T.dot(margin) / num_train + reg * self.W
        return loss, dW
    def train(self, x, y, reg=1e-5, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False):
        """
        功能：使用随机梯度下降法训练SVM
        输入：
        -x:(numpy array)训练样本（N,D）
        -y:(numpy array)训练样本标签(N,)
        -reg:(float)正则化强度
        -learning_rate:(float)进行权重更新的学习率
        -num_iters:(int)优化的迭代次数
        -batch_size:(int)随机梯度下降法每次使用的梯度大小
        -verbose:(bool)取True时，打印输出loss的变化过程
        输出：-history_loss:(list)存储每次迭代后的loss值
        """
        num_train, dim = x.shape
        num_class = np.max(y) + 1
        # 初始化权重
        if self.W is None:
            self.W = 0.005 * np.random.randn(dim, num_class)
        batch_x = None
        batch_y = None
        history_loss = []
        # 随机梯度下降法优化权重
        for i in range(num_iters):
            # 从训练样本中随机取样作为更新权重的小批量样本
            mask = np.random.choice(num_train, batch_size, replace=False)
            batch_x = x[mask]
            batch_y = y[mask]
            # 计算loss和权重的梯度
            loss, grad = self.svm_loss_vectorized(batch_x, batch_y, reg)
            # 更新权重
            self.W += -learning_rate * grad
            history_loss.append(loss)
            # 打印loss的变化过程
            if verbose == True and i % 100 == 0:
                print("iteratons:%d/%d,loss:%f" % (i, num_iters, loss))
        return history_loss
    def predict(self, x):
        """
        功能：利用训练得到的最优权值预测分类结果
        输入：
        -x:(numpy array)待分类的样本(N,D)
        输出：y_pre(numpy array)预测的便签(N,)
        """
        y_pre = np.zeros(x.shape[0])
        scores = x.dot(self.W)
        y_pre = np.argmax(scores, axis=1)
        return y_pre

# %%
import numpy as np
import random
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

# 下载CIFAR-10数据集
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 提取特征和标签
X = []
y = []

for data in trainset:
    image, label = data
    feature_vector = image.view(-1).numpy()
    X.append(feature_vector)
    y.append(label)

X = np.array(X)
y = np.array(y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化SVM模型
svm = SVM()

# 训练SVM模型
history_loss = svm.train(X_train, y_train, reg=1e-5, learning_rate=1e-3, num_iters=1000, batch_size=200, verbose=True)

# 预测
y_pred = svm.predict(X_test)

# 评估性能
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')


# %%
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')

# 计算精确度
precision = precision_score(y_test, y_pred, average='macro')
print(f'Precision on the test set: {precision * 100:.2f}%')

# 计算召回率
recall = recall_score(y_test, y_pred, average='macro')
print(f'Recall on the test set: {recall * 100:.2f}%')

# 计算F1分数
f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score on the test set: {f1 * 100:.2f}%')

# 输出详细的分类报告
report = classification_report(y_test, y_pred)
print(report)



