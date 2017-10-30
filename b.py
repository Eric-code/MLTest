from numpy import *
import numpy
import operator
# KNN算法

# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    # create a matrix: each row as a sample
    a = numpy.loadtxt('data3.txt')
    train = a[:, 2:]
    group = array(train)
    # group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = []
    for i in range(30):
        labels.append('douyu')
    for i in range(30):
        labels.append('xunlei')
    for i in range(30):
        labels.append('qq')
    return group, labels


# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


if __name__ == '__main__':
    b = numpy.loadtxt('test3.txt')
    test = b[::, 2:]
    dataSet, labels = createDataSet()
    for i in range(90):
        testX = array(test[i])
        outputLabel = kNNClassify(testX, dataSet, labels, 3)
        print("Your input is:", testX, "and classified to class:", outputLabel, i)

