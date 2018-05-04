import numpy as np
# f2 = open('flow_15s_all_19_test.csv', 'r')

x_train = []
y_train = []
x_test = []
y_test = []
with open('flow_15s_all_19_train.csv', 'r') as f1:
    for line in f1.readlines():
    # line = f1.readline()
        row = line.split(',')
        label = list(row.pop(-1))[0]
        y_train.append(label)
        x_train.append(row)

