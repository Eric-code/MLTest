import numpy

train = []
# with open('xunlei_output.txt', 'r') as f:
#     for line in f.readlines():
#         linestr = line.strip()
#         print(linestr)
#         linestrlist = linestr.split("\t")
#         print(linestrlist)
#         linelist = map(int, linestrlist)
#         print(linelist)
a = numpy.loadtxt('douyu_output.txt')
print(a)
b = numpy.loadtxt('xunlei_output.txt')
print(b)
# test = a[3::4, 2:]

# for i in range(10):
#     train = train + a[4 * i:4 * i + 3, 2:]

# print(test)
# print(train)
