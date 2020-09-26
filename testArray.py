
from mpl_toolkits import mplot3d
from kNN import kNN
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import math
import copy

ten = range(10)
test_array = []
for a in ten:
    for b in ten:
        for c in ten:
            test_array.append([a,b,c,int(a/2)])
training_data = np.array(test_array)
print(training_data, training_data.shape)
test_data = np.array([
    [5,5,5, int(5/2)],
    [0,0,0,0],
    [9,9,9, int(9/2)],
    [4.5,4.5,4.5, int(4.5/2)]
])

# savetxt('training_data.csv', training_data, delimiter=',')

# x, y, z = training_data[:,0], training_data[:,1], training_data[:,2]

# fig  = plt.figure(figsize=(10,7))
# ax = plt.axes(projection ="3d")

# ax.scatter3D(x,y,z, color="green")
# plt.title("print 3d")

# plt.show()

knn = kNN(
    # int(math.sqrt(len(training_data))), 
    5,
    training_data,
    [],
    False
)
classifications = knn.classify(training_data, test_data)
