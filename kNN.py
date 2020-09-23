#Written by 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import copy
import RealDistance
import heapq
import pandas as pd
import numpy as np
import DataUtility

class kNN:

    def __init__(self, k, data_set, categorical_features, regression_data_set):
        self.k = k
        self.data_set = data_set
        self.categorical_features = categorical_features
        self.regression_data_set = regression_data_set
        self.rd = RealDistance.RealDistance()

    def get_neighbors(self, exampleData, new_sample):
        neighbors = [(float('inf'), -1) for i in range(self.k)]
        for i in range(len(exampleData)):
            x = exampleData[i].tolist()[:-1]
            distance = self.rd.Distance(x, new_sample.tolist())
            heapq.heappush(neighbors, (distance, i))
        kNeighbors = heapq.nsmallest(self.k, neighbors)
        return kNeighbors

    def classify(self, exampleData, testData):
        
        for new_sample in testData:
            neighbors = self.get_neighbors(exampleData, new_sample)
            print(neighbors)
            votes = [exampleData[n[1]].tolist()[-1] for n in neighbors]
            print([exampleData[n[1]].tolist() for n in neighbors])
            print(votes)

if __name__ == '__main__':
    categorical_attribute_indices = {
        "segmentation": [],
        "vote": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "glass": [],
        "fire": [0,1,2,3],
        "machine": [0,1],
        "abalone": [0]
    }

    regression_data_set = {
        "segmentation": False,
        "vote": False,
        "glass": False,
        "fire": True,
        "machine": True,
        "abalone": True
    }
    data_set = "glass"
    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
    test = copy.deepcopy(tenFolds[0][0:5])
    test = test[:,:-1]
    training = np.concatenate(tenFolds[1:])
    print(len(test[0]))
    print(len(training))

    kNN = kNN(3, data_set, categorical_attribute_indices, regression_data_set)
    kNN.classify(training, test)

