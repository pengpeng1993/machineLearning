import tensorflow as tf
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

data, target = make_blobs(centers=2)
plt.scatter(data[:, 0], data[:, 1], c=target)
x = data.copy()
y = target.copy()
print(x.shape, y.shape)  # (100, 2) (100,)

plt.show()

x = tf.constant(data, dtype=tf.float32)
y = tf.constant(target, dtype=tf.float32)

# 定义预测变量
W = tf.Variable(np.random.randn(2, 1) * 0.2, dtype=tf.float32)
B = tf.Variable(0., dtype=tf.float32)


def sigmoid(x):
    linear = tf.matmul(x, W) + B
    return tf.nn.sigmoid(linear)


# 定义损失
def cross_entropy_loss(y_true, y_pred):
    # y_pred 是概率,存在可能性是0, 需要进行截断
    y_pred = tf.reshape(y_pred, shape=[100])
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1)
    return tf.reduce_mean(-(tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply((1 - y_pred),
                                                                                   tf.math.log(1 - y_pred))))


# 定义优化器
optimizer = tf.optimizers.SGD()


def run_optimization():
    with tf.GradientTape() as g:
        # 计算预测值
        pred = sigmoid(x)  # 结果为概率
        loss = cross_entropy_loss(y, pred)

    # 计算梯度
    gradients = g.gradient(loss, [W, B])
    # 更新W, B
    optimizer.apply_gradients(zip(gradients, [W, B]))


# 计算准确率
def accuracy(y_true, y_pred):
    # 需要把概率转换为类别
    # 概率大于0.5 可以判断为正例
    y_pred = tf.reshape(y_pred, shape=[100])
    y_ = y_pred.numpy() > 0.5
    y_true = y_true.numpy()
    return (y_ == y_true).mean()


# 定义训练过程
for i in range(5000):
    run_optimization()
    if i % 100 == 0:
        pred = sigmoid(x)
        acc = accuracy(y, pred)
        loss = cross_entropy_loss(y, pred)
        print(f'训练次数:{i}, 准确率: {acc}, 损失: {loss}')
