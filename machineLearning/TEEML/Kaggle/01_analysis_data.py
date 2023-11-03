
# 数据集来源 kaggle:Give me some credit
# https://www.kaggle.com/datasets?search=Give+me+some+credit

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("./data/cs-training.csv")
test_df = pd.read_csv("./data/cs-test.csv")

train_df.set_index('Unnamed: 0', inplace=True)
test_df.set_index('Unnamed: 0', inplace=True)

# print(train_df.shape)  # (150000, 12)
# print(test_df.shape)  # (101503, 12)
# print(train_df.head())
# print(test_df.head())
# print(train_df.info())
# print(test_df.info())

pd.set_option('display.max_columns', None)
# print(train_df.head())

columns = {"SeriousDlqin2yrs": "好坏客户",
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


print(train_df.head())

print(train_df.isnull().sum())
print(test_df.isnull().sum())


for column in train_df.columns:
    print(f"Field: {column}")
    print(train_df[column].value_counts())
    print(train_df[column].describe())
    print("===================================")

# print(train_df[(train_df['年龄'] < 18)])


# 好坏客户分布极其不均衡
# 可用额度比值应该小于1，所以后面将大于1的值当做异常值剔除。
# 年龄小于18岁剔除
# 逾期比数大于80作为异常值
#

# 数据异常值处理
def exceptionProcess(data_df):
    data_df = data_df[data_df['可用额度比例'] < 1]
    data_df = data_df[data_df['年龄'] > 18]
    data_df = data_df[data_df['逾期30-59天笔数'] < 80]
    data_df = data_df[data_df['逾期60-89天笔数'] < 80]
    data_df = data_df[data_df['逾期90天笔数'] < 80]
    return data_df


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
