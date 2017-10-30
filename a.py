from numpy import *
import numpy
import operator
# 朴素贝叶斯算法

# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    # create a matrix: each row as a sample
    a = numpy.loadtxt('data3.txt')
    train = a[:, 2:]
    dataSet = array(train)
    # group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = []
    for i in range(30):
        labels.append(0)
    for i in range(30):
        labels.append(1)
    for i in range(30):
        labels.append(2)
    return dataSet, labels

def train(dataSet, labels):
    dataNum = len(dataSet)
    featureNum = len(dataSet[0])
    p0Num = ones(featureNum)
    p1Num = ones(featureNum)
    p2Num = ones(featureNum)
    p0Denom = 2.0
    p1Denom = 2.0
    p2Denom = 2.0
    p0 = 0
    p1 = 0
    for i in range(dataNum):
        if labels[i] == 1:
            p1 += 1
            p1Num += dataSet[i]
            p1Denom += sum(dataSet[i])
        elif labels[i] == 2:
            p2Num += dataSet[i]
            p2Denom += sum(dataSet[i])
        else:
            p0 += 1
            p0Num += dataSet[i]
            p0Denom += sum(dataSet[i])
    p0Rate = p0 / dataNum
    p1Rate = p1 / dataNum
    p0Vec = log(p0Num / p0Denom)
    p1Vec = log(p1Num / p1Denom)
    p2Vec = log(p2Num / p2Denom)
    return p0Rate, p1Rate, p0Vec, p1Vec, p2Vec


def maxP(p0, p1, p2):
    p = p0
    if p1 > p:
        p = p1
    if p2 > p:
        p = p2
        print('qq')
    elif p == p0:
        print('douyu')
    else:
        print('xunlei')


if __name__ == '__main__':
    b = numpy.loadtxt('test3.txt')
    test = b[::, 2:]
    dataSet, labels = createDataSet()
    p0Rate, p1Rate, p0Vec, p1Vec, p2Vec = train(dataSet, labels)
    for i in range(90):
        testX = array(test[i])
        p1 = sum(testX * p1Vec) + log(p1Rate)
        p0 = sum(testX * p0Vec) + log(p0Rate)
        p2 = sum(testX * p2Vec) + log(1 - p0Rate - p1Rate)
        print(i + 1)
        maxP(p0, p1, p2)
