#Written By 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import numpy as np
import DataUtility, kNN, Results
import copy, random


class kMeansClustering:
    
    def __init__(self, kValue: int, dataSet: np.ndarray, categorical_features: list, d: int):
        self.categorical_features = categorical_features
        real_features = list(range(d))
        for i in categorical_features:
            real_features.remove(i)
        self.real_features = real_features
        self.kValue = kValue
        self.dataSet = dataSet
        # dimensionality of data set
        self.d = d

    def generateClusterPoints(self):
        feature_values = [None] * self.d
        for i in self.categorical_features:
            print(self.dataSet[:,i])
            feature_values[i] = np.unique(self.dataSet[:,i]).tolist()
            print("distinct features:", feature_values[i])
        for j in self.real_features:
            print(self.dataSet[:,j])
            min = np.min(self.dataSet[:,j])
            max = np.max(self.dataSet[:,1])
            feature_values[j] = [min, max]
            print("min:", min, "max:", max)
        points = []

        for k in range(self.kValue):
            new_point = [None] * self.d
            for l in self.categorical_features:
                new_point[l] = random.choice(feature_values[l])
            for m in self.real_features:
                new_point[m] = random.uniform(feature_values[m][0],feature_values[m][1])
            points.append(new_point)
        return points

    def closest_centroid_to_point(self, point: list, centroids: np.ndarray):


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

    feature_data_types = {
        "segmentation": 'real',
        "vote": 'categorical',
        "glass": 'real',
        "fire": 'mixed',
        "machine": 'mixed',
        "abalone": 'mixed'
    }

    data_sets = ["segmentation", "vote", "glass", "fire", "machine", "abalone"]

    regression = [x for x in data_sets if regression_data_set[x]]

    for i in range(1):
        data_set = "vote"

        print("Data set: ", data_set)
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        # print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        test = copy.deepcopy(tenFolds[0])
        training = np.concatenate(tenFolds[1:])

        d = len(headers)-1
        kMC = kMeansClustering(kValue=d, dataSet=training, categorical_features=categorical_attribute_indices[data_set], d=d)
        print(kMC.generateClusterPoints())