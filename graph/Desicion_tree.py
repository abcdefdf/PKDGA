from sklearn import tree  # 导包
from sklearn.datasets import load_wine  # 红酒数据集
from sklearn.model_selection import train_test_split  # 数据划分训练集与测试集
import numpy as np


# 获取数据
def get_data(name):
    f = open(name, 'r')
    data = f.read().splitlines()
    f.close()
    train_x = []
    train_y = []
    for i in data:
        i = i.replace(' ', '')
        c = i.split(',')
        x = np.array(c[:4])
        x = x.astype(np.float)
        y = np.array(c[5])
        train_x.append(x)
        train_y.append(y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

    
train_x, train_y = get_data('./graph.txt')
test_x, test_y = get_data('./test.txt')

# 实例化模型，划分规则用的是熵
clf = tree.DecisionTreeClassifier(criterion="entropy")

# 拟合数据
clf = clf.fit(train_x, train_y)

prob = clf.predict_proba(test_x)
print(prob, test_y)
# 测试评分
score = clf.score(test_x, test_y)
print(score)
# # 查看特征重要程度
# print(clf.feature_importances_)
