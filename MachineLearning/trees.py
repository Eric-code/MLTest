from math import log
import operator
from MachineLearning import treePlotter
from numpy import *
import numpy


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * numpy.math.log(prob, 2)
    return shannonEnt


def createDataSet():
    # dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    # labels = ['no surfacing', 'flippers']
    a = numpy.loadtxt('data3.txt')
    train = a[:, 2:]
    dataSet = array(train)
    # group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['IEPSD', 'log-NDSF']
    return dataSet, labels


def changeDat(dataSet):
    dataBase = dataSet
    for i in range(len(dataSet)):
        if dataBase[i][0] < 2.5:
            dataBase[i][0] = 1.0
        elif dataBase[i][0] > 6.5:
            dataBase[i][0] = 3.0
        else:
            dataBase[i][0] = 2.0
        if dataBase[i][1] < 8:
            dataBase[i][1] = 3.0
        elif dataBase[i][1] > 10:
            dataBase[i][1] = 2.0
        else:
            dataBase[i][1] = 1.0
    return dataBase


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for fearVec in dataSet:
        if fearVec[axis] == value:
            # 在数据集筛选出的元素中剔除了目标特征
            reducedFeatVec = list(fearVec[:axis])
            reducedFeatVec.extend(fearVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算条件熵
def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet)  # ∑pH(Y|X=xi) 条件熵的计算
    return ce


# 计算信息增益
def calcInformationGain(dataSet, baseEntropy, i):
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就是熵的减少，也就是不确定性的减少
    return infoGain


# 计算信息增益比
def calcInformationGainRate(dataSet, baseEntropy, i):
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy


def chooseBestFeatureToSplitByC45(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGainRate = calcInformationGainRate(dataSet, baseEntropy, i)
        if (infoGainRate > bestInfoGainRate):  # 选择最大的信息增益比
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitByC45(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 决策树的分类函数，返回当前节点的分类标签
def classify(inputTree, featLabels, testVec):  # 传入的数据为dict类型
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    # 这里表明了python3和python2版本的差别，上述两行代码在2.7中为：firstStr = inputTree.key()[0]
    secondDict = inputTree[firstStr]  # 建一个dict
    # print(secondDict)
    featIndex = featLabels.index(firstStr)  # 找到在label中firstStr的下标
    for i in secondDict.keys():
        print(i)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:  # 判断一个变量是否为dict，直接type就好
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel  # 比较测试数据中的值和树上的值，最后得到节点


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    Dat, labels = createDataSet()
    myDat = changeDat(Dat)
    # myDat = [[1.0, 1.0]]
    # for i in range(9):
    #     myDat.append([1.0, 2.0])
    # for i in range(10):
    #     myDat.append([2.0, 3.0])
    # for i in range(10):
    #     myDat.append([3.0, 1.0])
    print(myDat)
    myTree = createTree(myDat, labels)
    # print(splitDataSet(myDat, 0, 0))
    print(myTree)
    # print(classify(myTree, labels, [1, 0]))
    # print(classify(myTree, labels, [1, 1]))
