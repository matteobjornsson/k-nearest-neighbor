import numpy as np
import pandas as pd
import copy
import pprint

class ValueDifferenceMetric:

    def __init__(self, data_set: np.ndarray, categorical_features: list):
        self.categorical_features = categorical_features
        self.data_set = copy.deepcopy(data_set)
        self.featureDifferenceMatrix = self.calculateFDM(self.data_set)

    def calculateFDM(self, data_set: np.ndarray) -> dict:
        featureDifferenceMatrix = {}
        counts_per_feature = {}
        for index in self.categorical_features:
            counts_per_feature[index]["unique_values"] = np.unique(data_set[:, index])
        classes = np.unique(data_set[:,-1])
        

        print("calculating feature difference matrix")
        return data_set

    def distance(self, x1, x2):
        print("returning distance based on FDM")
        return 8.5


####################################### UNIT TESTING #################################################

if __name__ == '__main__':
    df = pd.read_csv("./ProcessedData/vote.csv")
    data = df.to_numpy()
    categorical_features = [1,2,4,5]
    unique_values = {}
    for index in categorical_features:
        unique_values[index] = np.unique(data[:,index]).tolist()
    print(unique_values)
    classes = np.unique(data[:,-1])
    count = dict.fromkeys(categorical_features)
    class_count = dict.fromkeys(categorical_features)
    print(count, class_count)
    for key, values in unique_values.items():
        count[key] = dict.fromkeys(values)
        class_count[key] = dict.fromkeys(values)
        for v in values:
            count[key][v] = 0
            class_count[key][v] = dict.fromkeys(classes)
            for c in classes:
                class_count[key][v][c] = 0
    


    for j in range(len(data)):
        x = data[j].tolist()
        for i in categorical_features:
            count[i][x[i]] += 1
            class_count[i][x[i]][x[-1]] += 1    

    pprint.pprint(count)
    print()
    pprint.pprint(class_count)
####################################### UNIT TESTING #################################################

