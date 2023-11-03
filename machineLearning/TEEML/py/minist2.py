import torch
import torchvision
from torch import nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib import font_manager
# https://blog.csdn.net/weixin_44327634/article/details/130454083

class Model(nn.Module):  # 创建模型，继承自nn.Module
    def __init__(self):
        super().__init__()
        #  第一层输入展平后的特征长度 28×28，创建120个神经元
        self.liner_1 = nn.Linear(28 * 28, 120)
        #  第二层输入的是前一层的输出,创建84个神经元
        self.liner_2 = nn.Linear(120, 84)
        #  输出层接收第二层的输入84，输出分类个数10
        self.liner_3 = nn.Linear(84, 10)

    def forward(self, input):
        x = input.view(-1, 28 * 28)      # 将输入展平为二维,（1, 28,28)→(28*28)
        x = torch.relu(self.liner_1(x))  # 连接第一层liner_1并使用ReLU函数激活
        x = torch.relu(self.liner_2(x))  # 连接第二层liner_2并使用ReLU函数激活
        #  输出层，输出张量的长度，与类别数量一致
        x = self.liner_3(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))  # 如果安装了GPu版本,显示Using cuda device
model = Model().to(device)  # 初始化模型，并设置模型使用device

loss_fn = nn.CrossEntropyLoss()  # 初始化交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 初始化优化器


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 获取当前数据集样本总数量
    num_batches = len(dataloader)  # 获取当前dataloader总批次数

    # train_loss用于累计所有批次的损失之和，correct用于累计预测正确的样本总数
    train_loss, correct = 0, 0
    for x, y in dataloader:
        # 对dataloader进行迭代
        x, y = x.to(device), y.to(device)  # 每一批次的数据设置为使用当前device
        # 进行预测，并计算一个批次的损失
        pred = model(x)
        loss = loss_fn(pred, y)  # 返回的是平均损失

        # 使用反向传播算法，根据损失优化模型参数
        optimizer.zero_grad()  # 将模型参数的梯度先全部归零
        loss.backward()  # 损失反向传播，计算模型参数梯度
        optimizer.step()  # 根据梯度优化参数
        with torch.no_grad():
            # correct用于累计预测正确的样本总数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # train_loss用于累计所有批次的损失之和
            train_loss += loss.item()
    # train_loss是所有批次的损失之和，所以计算全部样本的平均损失时需要除以总批次数
    train_loss /= num_batches
    # correct是预测正确的样本总数,若计算整个epoch总体正确率,需除以样本总数量
    correct /= size
    return train_loss, correct


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss, correct


train_ds = torchvision.datasets.MNIST('data/', train=True, transform=ToTensor(), download=True)
test_ds = torchvision.datasets.MNIST('data/', train=False, transform=ToTensor(), download=True)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=46)

epochs = 50  # 一个epoch代表对全部数据训练一遍
train_loss = []  # 每个epoch训练中训练数据集的平均损失被添加到此列表
train_acc = []  # 每个epoch训练中训练数据集的平均正确率被添加到此列表
test_loss = []  # 每个epoch训练中测试数据集的平均损失被添加到此列表
test_acc = []  # 每个epoch 训练中测试数据集的平均正确率被添加到此列表
for epoch in range(epochs):
    # 调用train()函数训练
    epoch_loss, epoch_acc = train(train_dl, model, loss_fn, optimizer)
    # 调用test()函数测试
    epoch_test_loss, epoch_test_acc = test(test_dl, model)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    # 定义一个打印模板
    template = "epoch: {:2d}, train_loss: {:.5f}, train_acc: {:.1f}% ,test_loss: {:.5f}, test_acc: {:.1f}%"
    # 输出当前epoch 的训练集损失、训练集正确率、测试集损失、测试集正确率
    print(template.format(epoch, epoch_loss, epoch_acc * 100, epoch_test_loss, epoch_test_acc * 100))

print("Done!")

# 调用windows中字体文件，使label标签中的中文正常显示，不然会乱码
font = font_manager.FontProperties(fname=r"C:\\Windows\\Fonts\\msyh.ttc", size=20)

# 绘制训练与测试损失比较图像
plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
plt.plot(range(1, epochs + 1), test_loss, label='test_loss', ls="--")
plt.xlabel('训练与测试损失比较', fontproperties=font)
plt.legend()
plt.show()

# 绘制训练与测试成功率比较图像
plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
plt.plot(range(1, epochs + 1), test_acc, label='test_acc', ls="--")
plt.xlabel('训练与测试成功率比较', fontproperties=font)
plt.legend()
plt.show()
