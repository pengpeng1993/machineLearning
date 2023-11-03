import pandas as pd
import numpy as np
import scipy.stats as stats

train_df = pd.read_csv("./data/cs-training-p.csv")
test_df = pd.read_csv("./data/cs-test-p.csv")


# print(train_df.shape)  # (146392, 11)
# print(test_df.shape)  # (99096, 11)

# # print(train_df.head())
# print(train_df["负债率"])
# print(train_df["可用额度比例"])
# print(train_df["年龄"])
# print(train_df["⽉收⼊"])
# print(train_df["逾期30-59天笔数"])


# 特征选择,利用IV对特征进行选择,并使用WOE对数据进行分箱
# IV WOE 介绍https://zhuanlan.zhihu.com/p/80134853

# 定义自动分箱函数
def mono_bin(Y, X, n=10):
    r = 0
    badNum = Y.sum()  # 坏人数
    goodNum = Y.count() - Y.sum()  # 好人数
    while abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})  # x切成n等分
        d2 = d1.groupby('Bucket', as_index=True, observed=False)  # 按照分箱结果进行分组聚合
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame()
    d3['min'] = d2.min().X  # 箱体的左边界
    d3['max'] = d2.max().X  # 箱体的右边界
    d3['badcostum'] = d2.sum().Y  # 每个箱体中坏样本的数量
    d3['goodcostum'] = d2.count().Y - d2.sum().Y  # 每个箱体好样本数
    d3['total'] = d2.count().Y  # 每个箱体的总样本数
    d3['bad_rate'] = d2.sum().Y / d2.count().Y  # 每个箱体中坏样本所占总样本数的比例
    d3['woe'] = np.log((d3['badcostum'] / d3['goodcostum']) * (goodNum / badNum))
    iv = ((d3['badcostum'] / badNum - d3['goodcostum'] / goodNum) * d3['woe']).sum()
    d3['iv'] = iv
    woe = list(d3['woe'].round(6))
    cut = list(d3['max'].round(6))
    cut.insert(0, float('-inf'))
    cut[-1] = float('inf')
    return d3, cut, woe, iv


dfx1, cut1, x1_woe, iv1 = mono_bin(train_df["好坏客户"], train_df["可用额度比例"], 5)
dfx2, cut2, x2_woe, iv2 = mono_bin(train_df["好坏客户"], train_df["年龄"], 5)
dfx4, cut4, x4_woe, iv4 = mono_bin(train_df["好坏客户"], train_df["负债率"], 5)
dfx5, cut5, x5_woe, iv5 = mono_bin(train_df["好坏客户"], train_df["⽉收⼊"], 5)


# 手动分箱
def hand_bin(Y, X, cut):
    badNum = Y.sum()  # 坏人数
    goodNum = Y.count() - Y.sum()  # 好人数
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, cut, duplicates="drop")})
    d2 = d1.groupby('Bucket', as_index=True, observed=False)
    d3 = pd.DataFrame()
    d3['min'] = d2.min().X  # 箱体的左边界
    d3['max'] = d2.max().X  # 箱体的右边界
    d3['badcostum'] = d2.sum().Y  # 每个箱体中坏样本的数量
    d3['goodcostum'] = d2.count().Y - d2.sum().Y  # 每个箱体好样本数
    d3['total'] = d2.count().Y  # 每个箱体的总样本数
    d3['bad_rate'] = d2.sum().Y / d2.count().Y  # 每个箱体中坏样本所占总样本数的比例
    d3['woe'] = np.log((d3['badcostum'] / d3['goodcostum']) * (goodNum / badNum))
    iv = ((d3['badcostum'] / badNum - d3['goodcostum'] / goodNum) * d3['woe']).sum()
    d3['iv'] = iv
    woe = list(d3['woe'].round(6))
    return d3, cut, woe, iv


ninf = float('-inf')
pinf = float('inf')
cut3 = [ninf, 0, 1, 3, 5, pinf]
cut6 = [ninf, 1, 2, 3, 5, pinf]
cut7 = [ninf, 0, 1, 3, 5, pinf]
cut8 = [ninf, 0, 1, 2, 3, pinf]
cut9 = [ninf, 0, 1, 3, 5, pinf]
cut10 = [ninf, 0, 1, 2, 3, 5, pinf]

dfx3, cut3, x3_woe, iv3 = hand_bin(train_df["好坏客户"], train_df["逾期30-59天笔数"], 5)
dfx6, cut6, x6_woe, iv6 = hand_bin(train_df["好坏客户"], train_df["信贷数量"], 5)
dfx7, cut7, x7_woe, iv7 = hand_bin(train_df["好坏客户"], train_df["逾期90天笔数"], 5)
dfx8, cut8, x8_woe, iv8 = hand_bin(train_df["好坏客户"], train_df["固定资产贷款数量"], 5)
dfx9, cut9, x9_woe, iv9 = hand_bin(train_df["好坏客户"], train_df["逾期60-89天笔数"], 5)
dfx10, cut10, x10_woe, iv10 = hand_bin(train_df["好坏客户"], train_df["家属数量"], 5)

print(pd.DataFrame([iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10]))


def replace_woe(X, cut, woe):
    x_woe = pd.cut(X, cut, labels=woe)
    return x_woe

train_x1_woe = replace_woe(train_df)