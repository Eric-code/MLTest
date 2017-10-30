from sklearn import svm
import numpy
from sklearn.model_selection import train_test_split
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
    return dataSet, labels

if __name__ == '__main__':
    x, y = createDataSet()
    print(len(x))
    print(len(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3., random_state=8)  # 分割训练集和测试集
    start_time = datetime.datetime.now()
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))  # 精度
    y_hat = clf.predict(x_train)
    print(y_hat[:10])
    print("Score: %0.2f" % (clf.score(x_test, y_test)))
    scores = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=10)
    print("Cross Avg. Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print("Time: %0.2f" % (time_spend.total_seconds()))

