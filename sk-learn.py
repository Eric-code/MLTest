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
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open('flow_30s_all_19_train.csv', 'r') as f1:
        for line in f1.readlines():
            # line = f1.readline()
            row = line.split(',')
            label = list(row.pop(-1))[0]
            y_train.append(label)
            x_train.append(row)
    with open('flow_30s_all_19_test.csv', 'r') as f2:
        for line in f2.readlines():
            row = line.split(',')
            label = list(row.pop(-1))[0]
            y_test.append(label)
            x_test.append(row)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = createDataSet()
    estimators = {}
    # criterion: 分支的标准(gini/entropy)
    estimators['tree'] = tree.DecisionTreeClassifier(criterion='gini', random_state=8)  # 决策树

    # n_estimators: 树的数量
    # bootstrap: 是否随机有放回
    # n_jobs: 可并行运行的数量
    estimators['forest'] = RandomForestClassifier(n_estimators=20, criterion='gini', bootstrap=True, n_jobs=2,
                                                  random_state=8)  # 随机森林
    # estimators['SVM'] = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')  # 支持向量机
    estimators['KNN'] = KNeighborsClassifier(3)  # K近邻
    # estimators['Bayes'] = MultinomialNB(alpha=0.0001)  # 朴素贝叶斯
    # estimators['MLP'] = MLPClassifier(solver='lbfgs', alpha=1e-4, random_state=1)  # 神经网络，hidden_layer_sizes=(5, 2),
    for k in estimators.keys():
        start_time = datetime.datetime.now()
        print('----%s----' % k)
        estimators[k] = estimators[k].fit(X_train, Y_train)
        pred = estimators[k].predict(X_test)
        # print(pred[:10])
        print("%s Score: %0.4f" % (k, estimators[k].score(X_test, Y_test)))
        scores = cross_val_score(estimators[k], X_train, Y_train, scoring='accuracy', cv=10)
        print("%s Cross Avg. Score: %0.4f (+/- %0.4f)" % (k, scores.mean(), scores.std() * 2))
        end_time = datetime.datetime.now()
        time_spend = end_time - start_time
        print("%s Time: %0.2f" % (k, time_spend.total_seconds()))
