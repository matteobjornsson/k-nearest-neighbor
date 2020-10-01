
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
training = np.array(test_array)
test = np.array([
    [5,5,5, int(5/2)],
    [0,0,0,0],
    [9,9,9, int(9/2)],
    [4.9,4.9,4.9, int(4.5/2)]
])

knn = kNN(
    1,
    "real",
    [],
    True,
    alpha=1, beta=2, h=.5, d=3
)
classifications = knn.classify(training, test)
for c in classifications:
    print(c)
