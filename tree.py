import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import datetime


def createDataSet():
    # dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    # labels = ['no surfacing', 'flippers']
    a = numpy.loadtxt('data.txt')
    train = a[:, 2:]
    dataSet = numpy.array(train)
    labels = []  # 0表示IEPSD，1表示log-NDSF
    # group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    for i in range(60):
        labels.append(1)  # douyu
    for i in range(60):
        labels.append(2)  # xunlei
    for i in range(60):
        labels.append(3)  # qq
    for i in range(60):
        labels.append(4)  # AHD
    return dataSet, labels


def classify(dataSet):
    dataBase = dataSet
    for i in range(len(dataSet)):
        if dataBase[i][0] < 2.5:
            dataBase[i][0] = 1
        elif dataBase[i][0] < 6.5:
            dataBase[i][0] = 2
        else:
            dataBase[i][0] = 3
        if dataBase[i][1] < 8:
            dataBase[i][1] = 3
        elif dataBase[i][1] < 10:
            dataBase[i][1] = 1
        else:
            dataBase[i][1] = 2
    return dataBase


if __name__ == '__main__':
    # digits = datasets.load_digits()
    # X = digits.data  # 特征矩阵
    # y = digits.target  # 标签矩阵
    x, y = createDataSet()
    X = classify(x)
    print(len(X))
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2., random_state=8)  # 分割训练集和测试集
    estimators = {}

# criterion: 分支的标准(gini/entropy)
    estimators['tree'] = tree.DecisionTreeClassifier(criterion='gini', random_state=8)  # 决策树

# n_estimators: 树的数量
# bootstrap: 是否随机有放回
# n_jobs: 可并行运行的数量
    estimators['forest'] = RandomForestClassifier(n_estimators=20, criterion='gini', bootstrap=True, n_jobs=2, random_state=8)  # 随机森林
    estimators['SVM'] = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')  # 支持向量机
    estimators['KNN'] = KNeighborsClassifier(3)  # K近邻
    estimators['Bayes'] = MultinomialNB(alpha=0.0001)   # 朴素贝叶斯
    estimators['MLP'] = MLPClassifier(solver='lbfgs', alpha=1e-4, random_state=1)   # 神经网络，hidden_layer_sizes=(5, 2),
    for k in estimators.keys():
        start_time = datetime.datetime.now()
        print('----%s----' % k)
        estimators[k] = estimators[k].fit(X_train, y_train)
        pred = estimators[k].predict(X_test)
        print(pred[:10])
        print("%s Score: %0.4f" % (k, estimators[k].score(X_test, y_test)))
        scores = cross_val_score(estimators[k], X_train, y_train, scoring='accuracy', cv=10)
        print("%s Cross Avg. Score: %0.4f (+/- %0.4f)" % (k, scores.mean(), scores.std() * 2))
        end_time = datetime.datetime.now()
        time_spend = end_time - start_time
        print("%s Time: %0.2f" % (k, time_spend.total_seconds()))

