# 数据集来源 kaggle:Give me some credit
# https://www.kaggle.com/datasets?search=Give+me+some+credit
# A B 两个银行拼接数据,横向 or 纵向,包括下面11个表属性;
# columns = {"SeriousDlqin2yrs": "好坏客户",
#            "RevolvingUtilizationOfUnsecuredLines": "可用额度比例",
#            "age": "年龄",
#            "NumberOfTime30-59DaysPastDueNotWorse": "逾期30-59天笔数",
#            "DebtRatio": "负债率",
#            "MonthlyIncome": "⽉收⼊",
#            "NumberOfOpenCreditLinesAndLoans": "信贷数量",
#            "NumberOfTimes90DaysLate": "逾期90天笔数",
#            "NumberRealEstateLoansOrLines": "固定资产贷款数量",
#            "NumberOfTime60-89DaysPastDueNotWorse": "逾期60-89天笔数",
#            "NumberOfDependents": "家属数量"}

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据拼接已经完成
train_df = pd.read_csv("data/cs-training.csv")
test_df = pd.read_csv("data/cs-test.csv")

pd.set_option('display.max_columns', None)

columns = {"Unnamed: 0": "id",
           "SeriousDlqin2yrs": "好坏客户",
           "RevolvingUtilizationOfUnsecuredLines": "可用额度比例",
           "age": "年龄",
           "NumberOfTime30-59DaysPastDueNotWorse": "逾期30-59天笔数",
           "DebtRatio": "负债率",
           "MonthlyIncome": "⽉收⼊",
           "NumberOfOpenCreditLinesAndLoans": "信贷数量",
           "NumberOfTimes90DaysLate": "逾期90天笔数",
           "NumberRealEstateLoansOrLines": "固定资产贷款数量",
           "NumberOfTime60-89DaysPastDueNotWorse": "逾期60-89天笔数",
           "NumberOfDependents": "家属数量"}

train_df.rename(columns=columns, inplace=True)
test_df.rename(columns=columns, inplace=True)

# 特征工程

# 第一处理空值
# print(train_df.shape)  # (150000, 12) 看有多少行数据
# print(test_df.shape)  # (101503, 12)
# print(train_df.info())  # 能看非空值有哪些
# print(test_df.info())  # 能看非空值有哪些
#
# print("月收入缺失比：{:.2%}".format(train_df["⽉收⼊"].isnull().sum() / train_df.shape[0]))
# print("家属数量缺失比：{:.2%}".format(train_df['家属数量'].isnull().sum() / train_df.shape[0]))

# 缺失值处理
# （1） 直接删除含有缺失值的样本。
# （2） 不予理睬
# （3） 填补缺失值。
# 针对1 缺失情况较少,并且无业务要求----删除行组件;
# 针对2 缺失情况较少,业务有要求不能删除该行,该行与预测值关联不大,后期特征工程不用或者少用
# 针对3 缺失情况较少,业务有要求不能删除该行, 并且与预测值关联关系很密切的情况下,填补缺失值
# 填补方法 1.常数填补,包括均值,同类均值,众数,随机数填补,  2.魔法填补:极大似然估计,随机森林,多重插补等

train_df['⽉收⼊'] = train_df['⽉收⼊'].fillna(train_df['⽉收⼊'].mean())  # 均值替换
train_df = train_df.dropna()  # 删除空行

print("月收入缺失比：{:.2%}".format(train_df["⽉收⼊"].isnull().sum() / train_df.shape[0]))
print("家属数量缺失比：{:.2%}".format(train_df['家属数量'].isnull().sum() / train_df.shape[0]))

# 异常值处理
# 异常值分为业务异常值与统计学异常值, 规则配置+箱线图
# for column in train_df.columns:
#     print(f"Field: {column}")
#     print(train_df[column].describe())
# fig = plt.figure(figsize=(15, 10))
# a = fig.add_subplot(3, 2, 1)
# b = fig.add_subplot(3, 2, 2)
# c = fig.add_subplot(3, 2, 3)
# d = fig.add_subplot(3, 2, 4)
# e = fig.add_subplot(3, 2, 5)
# f = fig.add_subplot(3, 2, 6)
#
# a.boxplot(train_df["可用额度比例"])
# b.boxplot([train_df["年龄"], train_df["好坏客户"]])
# c.boxplot([train_df["逾期30-59天笔数"], train_df["逾期60-89天笔数"], train_df["逾期90天笔数"]])
# d.boxplot([train_df["信贷数量"], train_df["固定资产贷款数量"], train_df["家属数量"]])
# e.boxplot(train_df["⽉收⼊"])
# f.boxplot(train_df["负债率"])
# fig.show()


# 处理思路,连续变量服从正态分布,均值方差等,数据量大的情况可以删除,不影响整体数据的特征;
# 2:可用额度比例 3:年龄 5:负债率 6:月收入 直接删除太偏的行
# 逾期数呈现明显分化,固定资产贷款量 家属数量 排除异常值
for k in [2, 3, 5, 6]:
    q1 = train_df.iloc[:, k].quantile(0.25)
    q3 = train_df.iloc[:, k].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr
    if k == 2:
        train1 = train_df
    train1 = train1[(train1.iloc[:, k] > low) & (train1.iloc[:, k] < up)]
train_df = train1
train_df.info()
# fig.add_subplot(train_df.iloc[:, [2, 3, 5, 6]].boxplot(figsize=(15, 10)))

train_df = train_df[train_df["逾期30-59天笔数"] < 80]
train_df = train_df[train_df["逾期60-89天笔数"] < 80]
train_df = train_df[train_df["逾期90天笔数"] < 80]
train_df = train_df[train_df["固定资产贷款数量"] < 50]
train_df = train_df[train_df["家属数量"] < 15]


# sns.set()
# f, ax = plt.subplots()
# sns.heatmap(train_df.corr(), annot=True, ax=ax, cmap='RdBu')  # 画热力图
# ax.set_title('相关性分析')  # 标题
# ax.set_xlabel('x轴')  # x轴
# ax.set_ylabel('y轴')  # y轴
# plt.show()


# why 分箱 https://zhuanlan.zhihu.com/p/503235392
# 变量分箱的目的是增加变量的预测能力或减少变量的自身冗余。
# 当预测能力不再提升或冗余性不再降低时，则分箱完毕。
# 因此分箱过程也是一个优化过程，所有满足上述要求的指标都可以用于变量分箱;
# 这个指标也可以叫做目标函数，可以终止或改变分箱的限制就是优化过程的约束条件。
# 优化的目标函数可以是卡方值、KS值、IV值、WOE值、信息熵和Gini值等，只要是可以提高变量的预测能力或减少变量自身冗余的指标，都可以作为目标函数使用

# 监督学习 二分类问题, logistic回归(分类) 特征选择 IV(Information Value)
# IV(Information Value) ------------- WOE(证据权重)  变量进行分组实际上也理解为离散化,也叫分箱
# 等距分段 gap 简单
# 等深分段 先确定分段数量，然后令每个分段中数据数量大致相等
# 最优分段

# 1. WOE describes the relationship between a predictive variable and a binary target variable.
# 2. IV measures the strength of that relationship.

# woe分箱种类 IV WOE 介绍 https://zhuanlan.zhihu.com/p/80134853
# 分箱单调性是指分箱后的WOE值随着分箱索引的增加而呈现增加或减少的趋势。
# 分箱单调是为了让Logistic回归模型可以得到更好的预测结果（线性特征更容易学习），但是往往有的变量自身就是U形结构，不太好通过分箱的方式让数据达到单调的效果。
# 这时候只是Logistic回归模型可能效果不佳
# 皮尔逊相关系数  Spearman p值(1-p = %把握)

def op(y, x, n=20):
    r = 0
    bad = y.sum()  # 变量为1,代表坏
    good = y.count() - bad
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"x": x, "y": y, "bucket": pd.qcut(x, n)})
        d2 = d1.groupby('bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().x,
                               d2.mean().y)  # Spearman相关系数的取值范围是-1到1，其中-1表示完全负相关，1表示完全正相关，0表示无相关性。p值用于检验Spearman相关系数是否显著。
        n = n - 1
    d3 = pd.DataFrame(d2.x.min(), columns=['min'])
    d3['min'] = d2.min().x  # 箱体的左边界
    d3['max'] = d2.max().x  # 箱体的右边界
    d3['sum'] = d2.sum().y  # 每个箱体中坏样本的数量
    d3['total'] = d2.count().y  # 每个箱体的总数
    d3['rate'] = d2.mean().y  # 每个箱体中坏样本所占总样本数的比例
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d4 = (d3.sort_values(by='min')).reset_index(drop=True)
    # print("=" * 60)
    # print(d4)
    return d4


x2 = op(train_df['好坏客户'], train_df['年龄'])
x4 = op(train_df['好坏客户'], train_df['负债率'])
x5 = op(train_df['好坏客户'], train_df['⽉收⼊'])


def funqcut(y, x, n):
    cut1 = pd.qcut(x.rank(method='first'), n)
    data = pd.DataFrame({"x": x, "y": y, "cut1": cut1})
    cutbad = data.groupby(cut1).y.sum()
    cutgood = data.groupby(cut1).y.count() - cutbad
    bad = data.y.sum()
    good = data.y.count() - bad
    woe = np.log((cutbad / bad) / (cutgood / good))
    iv = (cutbad / bad - cutgood / good) * woe

    cut = pd.DataFrame({"坏客户数": cutbad, "好客户数": cutgood, "woe": woe, "iv": iv})
    print(cut)

    return cut


x1 = funqcut(train_df["好坏客户"], train_df["可用额度比例"], 5).reset_index()
x2 = funqcut(train_df["好坏客户"], train_df["年龄"], 12).reset_index()
x4 = funqcut(train_df["好坏客户"], train_df["负债率"], 4).reset_index()
x5 = funqcut(train_df["好坏客户"], train_df["⽉收⼊"], 5).reset_index()
x6 = funqcut(train_df["好坏客户"], train_df["信贷数量"], 6).reset_index()

x3 = funqcut(train_df["好坏客户"], train_df["逾期30-59天笔数"], 5).reset_index()
x7 = funqcut(train_df["好坏客户"], train_df["逾期90天笔数"], 5).reset_index()
x8 = funqcut(train_df["好坏客户"], train_df["固定资产贷款数量"], 5).reset_index()
x9 = funqcut(train_df["好坏客户"], train_df["逾期60-89天笔数"], 5).reset_index()
x10 = funqcut(train_df["好坏客户"], train_df["家属数量"], 5).reset_index()

fig, axes = plt.subplots(4, 3, figsize=(20, 15))
x1.woe.plot(ax=axes[0, 0], title="可用额度比值")
x2.woe.plot(ax=axes[0, 1], title="年龄")
x3.woe.plot(ax=axes[0, 2], title="逾期30-59天笔数")
x4.woe.plot(ax=axes[1, 0], title="负债率")
x5.woe.plot(ax=axes[1, 1], title="月收入")
x6.woe.plot(ax=axes[1, 2], title="信贷数量")
x7.woe.plot(ax=axes[2, 0], title="逾期90天笔数")
x8.woe.plot(ax=axes[2, 1], title="固定资产贷款量")
x9.woe.plot(ax=axes[2, 2], title="逾期60-89天笔数")
x10.woe.plot(ax=axes[3, 0], title="家属数量")


# 数据异常值处理
def exceptionProcess(data_df):
    data_df = data_df[data_df['可用额度比例'] < 1]
    data_df = data_df[data_df['年龄'] > 18]
    data_df = data_df[data_df['逾期30-59天笔数'] < 80]
    data_df = data_df[data_df['逾期60-89天笔数'] < 80]
    data_df = data_df[data_df['逾期90天笔数'] < 80]
    return data_df


ivx1 = x1.iv.sum()
ivx2 = x2.iv.sum()
ivx3 = x3.iv.sum()
ivx4 = x4.iv.sum()
ivx5 = x5.iv.sum()
ivx6 = x6.iv.sum()
ivx7 = x7.iv.sum()
ivx8 = x8.iv.sum()
ivx9 = x9.iv.sum()
ivx10 = x10.iv.sum()
IV = pd.DataFrame({"可用额度比值": ivx1,
                   "年龄": ivx2,
                   "逾期30-59天笔数": ivx3,
                   "负债率": ivx4,
                   "月收入": ivx5,
                   "信贷数量": ivx6,
                   "逾期90天笔数": ivx7,
                   "固定资产贷款量": ivx8,
                   "逾期60-89天笔数": ivx9,
                   "家属数量": ivx10}, index=[0])

ivplot = IV.plot.bar(figsize=(15, 10))
ivplot.set_title('特征变量的IV值分布')
plt.show()


# 通过IV值判断变量预测能力的标准是：
# < 0.02: unpredictive
# 0.02 to 0.1: weak
# 0.1 to 0.3: medium
# 0.3 to 0.5: strong

def cutdata(x, n):
    a = pd.qcut(x.rank(method='first'), n, labels=False)
    return a


# x1 = funqcut(train_df["好坏客户"], train_df["可用额度比例"], 5).reset_index()
# x2 = funqcut(train_df["好坏客户"], train_df["年龄"], 12).reset_index()
# x4 = funqcut(train_df["好坏客户"], train_df["负债率"], 4).reset_index()
# x5 = funqcut(train_df["好坏客户"], train_df["⽉收⼊"], 5).reset_index()
# x6 = funqcut(train_df["好坏客户"], train_df["信贷数量"], 6).reset_index()
#
# x3 = funqcut(train_df["好坏客户"], train_df["逾期30-59天笔数"], 5).reset_index()
# x7 = funqcut(train_df["好坏客户"], train_df["逾期90天笔数"], 5).reset_index()
# x8 = funqcut(train_df["好坏客户"], train_df["固定资产贷款数量"], 5).reset_index()
# x9 = funqcut(train_df["好坏客户"], train_df["逾期60-89天笔数"], 5).reset_index()
# x10 = funqcut(train_df["好坏客户"], train_df["家属数量"], 5).reset_index()

cut1 = cutdata(train_df['可用额度比例'], 5)
cut2 = cutdata(train_df['年龄'], 12)
cut3 = cutdata(train_df['逾期30-59天笔数'], 5)
cut4 = cutdata(train_df['负债率'], 4)
cut5 = cutdata(train_df['⽉收⼊'], 5)
cut6 = cutdata(train_df['信贷数量'], 6)
cut7 = cutdata(train_df['逾期90天笔数'], 5)
cut8 = cutdata(train_df['固定资产贷款数量'], 5)
cut9 = cutdata(train_df['逾期60-89天笔数'], 5)
cut10 = cutdata(train_df['家属数量'], 5)


# print(cut1.head())

def replace_train(cut, cut_woe):
    a = []
    for i in cut.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        cut.replace(a[m], cut_woe.values[m], inplace=True)
    return cut


train_new = pd.DataFrame()
train_new['好坏客户'] = train_df['好坏客户']
train_new['可用额度比值'] = replace_train(cut1, x1.woe)
train_new['年龄'] = replace_train(cut2, x2.woe)
train_new['逾期30-59天笔数'] = replace_train(cut3, x3.woe)
train_new['负债率'] = replace_train(cut4, x4.woe)
train_new['月收入'] = replace_train(cut5, x5.woe)
train_new['信贷数量'] = replace_train(cut6, x6.woe)
train_new['逾期90天笔数'] = replace_train(cut7, x7.woe)
train_new['固定资产贷款量'] = replace_train(cut8, x8.woe)
train_new['逾期60-89天笔数'] = replace_train(cut9, x9.woe)
train_new['家属数量'] = replace_train(cut10, x10.woe)
print(train_new.head())
train_new1 = train_new.drop(["负债率", "月收入", "信贷数量", "固定资产贷款量", "家属数量"], axis=1)
print(train_new1.head())

x = train_new1.iloc[:, 1:]
y = train_new.iloc[:, 0]

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=4)

model = LogisticRegression()
result = model.fit(train_x, train_y)
pred_y = model.predict(test_x)
result.score(test_x, test_y)

# 指标 https://img-blog.csdnimg.cn/2020052210345735.png https://img-blog.csdnimg.cn/2020052210372254.png https://img-blog.csdnimg.cn/20200522103935274.png
proba_y = model.predict_proba(test_x)

fpr, tpr, threshold = roc_curve(test_y, proba_y[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, "b", label="AUC= %0.2f" % roc_auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("真正率")
plt.xlabel("假正率")
plt.show()
print(roc_auc)

dataks = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold})
print(dataks.head())
dataks = dataks.sort_values(["threshold"])
plt.plot(dataks.iloc[:, 2], dataks['fpr'], label='fpr')
plt.plot(dataks.iloc[:, 2], dataks['tpr'], label='tpr')
plt.xlim([0, 1])
plt.legend(loc='upper left')
plt.show()
ks = max(tpr - fpr)
print("ks值为：", ks)  # ks值为0.47大于0.4，表明分类器具有区分能力。

# 信用评分 https://img-blog.csdnimg.cn/2020052210444290.png
B = PDD / log(2)
A = Z + B * log(v)

B = 30 / np.log(2)
A = 700 + B * np.log(1 / 70)

c = result.intercept_
coef = result.coef_
BaseScore = A - B * c
print(BaseScore)


def get_score(x, coef, B):
    score = []
    for w in x.woe:
        a = round(B * coef * w, 0)
        score.append(a)
    datascore = pd.DataFrame({"分组": x.iloc[:, 0], "得分": score})
    return datascore


scorex1 = get_score(x1, coef[0][0], B)
scorex2 = get_score(x2, coef[0][1], B)
scorex3 = get_score(x3, coef[0][2], B)
scorex7 = get_score(x7, coef[0][3], B)
scorex9 = get_score(x9, coef[0][4], B)

# display("可用额度比值",scorex1)
# display("年龄",scorex2)
# display("逾期30-59天笔数",scorex3)
# display("逾期90天笔数",scorex7)
# display("逾期60-89天笔数",scorex9)
print("基础分值为：", BaseScore)

train_df = exceptionProcess(train_df)
test_df = exceptionProcess(test_df)


# print(train_df.loc[(train_df['年龄'] < 18) | (train_df['可用额度比例'] > 1) | (train_df['逾期35-59天笔数'] > 80) | (
#         train_df['逾期60-89天笔数'] > 80) | (train_df['逾期90天笔数'] > 80)])
# print(test_df.loc[(test_df['年龄'] < 18) | (test_df['可用额度比例'] > 1) | (test_df['逾期35-59天笔数'] > 80) | (
#         test_df['逾期60-89天笔数'] > 80) | (test_df['逾期90天笔数'] > 80)])

# print(train_df.shape)
# print(test_df.shape)


# 数据缺失值填充,月收入缺失较多,使用随机森林模型进行填充

def fillMonthIncome(data_df):
    know = data_df[data_df['⽉收⼊'].notnull()]
    unknown = data_df[data_df['⽉收⼊'].isnull()]

    x_train = know.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9]]
    y_train = know.iloc[:, 5]
    x_test = unknown.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9]]

    clf = RandomForestRegressor(n_estimators=200, random_state=0, max_depth=3, n_jobs=-1)
    pred = clf.fit(x_train, y_train).predict(x_test)
    return pred


pred_train = fillMonthIncome(train_df)
pred_test = fillMonthIncome(test_df)

train_df.loc[train_df["⽉收⼊"].isnull(), "⽉收⼊"] = pred_train
test_df.loc[test_df["⽉收⼊"].isnull(), "⽉收⼊"] = pred_test

print("---------------------------------------------")
print(train_df.isnull().sum())
print("---------------------------------------------")
print(test_df.isnull().sum())


def fillNumDependent(data_df):
    data_df["家属数量"].fillna(0, inplace=True)
    return data_df


train_df = fillNumDependent(train_df)
print(train_df.isnull().sum())

test_df = fillNumDependent(test_df)
print(test_df.isnull().sum())

train_df.to_csv("./data/cs-training-p.csv", index=False, header=True)
test_df.to_csv("./data/cs-test-p.csv", index=False, header=True)
