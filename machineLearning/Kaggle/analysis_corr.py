import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
train_df = pd.read_csv("data/cs-training-p.csv")
test_df = pd.read_csv("data/cs-test-p.csv")

print(train_df.shape)  # (146392, 11)
print(test_df.shape)  # (99096, 11)

# 一般相关系数大于0.6可以进行变量剔除
# 相关系数的取值范围为[-1,1]
# 属于0.8-1：极强相关；
# 属于0.6-0.8：强相关；
# 属于0.4-0.6：中等程度相关；
# 属于0.2-0.4：
# 弱相关；0-0.2：极弱相关或无相关
# pearson https://zh.wikipedia.org/wiki/%E7%9A%AE%E5%B0%94%E9%80%8A%E7%A7%AF%E7%9F%A9%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0

# print(train_df.corr())

sns.set()
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
f, ax = plt.subplots()
sns.heatmap(train_df.corr(), annot=True, ax=ax, cmap='RdBu')  # 画热力图
ax.set_title('相关性分析')  # 标题
ax.set_xlabel('x轴')  # x轴
ax.set_ylabel('y轴')  # y轴
plt.show()

