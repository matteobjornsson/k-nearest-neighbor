#Written by 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import copy, math
from collections import Counter
import RealDistance, HammingDistance
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
        self.data_type = self.feature_data_types(data_set, categorical_features)
        self.rd = RealDistance.RealDistance()
        # tunable parameter, weight of real distance value
        self.alpha = 1
        self.hd = HammingDistance.HammingDistance()
        # tunable parameter, weight of categorical distance value
        self.beta = 1

    # function determines the nature of the data set features: real, categorical, or mixed
    def feature_data_types(self, data_set: np.ndarray, categorical_features: list) -> str:
        feature_count = data_set.shape[1]-1
        # if the number of features is the same as the number of columns that
        # are categorical, the entire feature set is categorical
        if len(categorical_features) == feature_count:
            return "categorical"
        # else if the list of categorical features is non-zero, feature set is mixed
        elif len(categorical_features) > 0:
            return "mixed"
        # last remaining option, all features are real
        else:
            return "real"

    def get_k_neighbors(self, exampleData: np.ndarray, new_sample: list, k) -> list:
        if self.data_type == "mixed":
            sample_cat = [new_sample[i] for i in range(len(new_sample)) if i in self.categorical_features]
            sample_real = [new_sample[j] for j in range(len(new_sample)) if j not in self.categorical_features]

        neighbors = []
        for n in range(len(exampleData)):
            x = exampleData[n].tolist()[:-1]

            if self.data_type == "real":
                distance = self.rd.Distance(x, new_sample)
            elif self.data_type == "categorical":
                distance = self.hd.Distance(x, new_sample)
            else:
                x_cat = [x[k] for k in range(len(x)) if k in self.categorical_features]
                x_real = [x[l] for l in range(len(x)) if l not in self.categorical_features]
                distance = self.alpha * self.rd.Distance(x_real, sample_real) + self.beta * self.hd.Distance(x_cat, sample_cat)
                if n < 5:
                    print("sample real: ", sample_real, "x real: ", x_real, "real distance: ", self.rd.Distance(x_real, sample_real))
                    print("sample cat: ", sample_cat, "x cat: ", x_cat, "categorical distance: ", self.hd.Distance(x_cat,sample_cat))
            heapq.heappush(neighbors, (distance, n))
        kNeighbors = heapq.nsmallest(k, neighbors)
        return kNeighbors

    def classify(self, exampleData: np.ndarray, testData: np.ndarray) -> list:
        classifications = []
        for new_sample in testData:
            new_vector = new_sample.tolist()[:-1]
            neighbors = self.get_k_neighbors(exampleData, new_vector, self.k)
            print("neighbors: ", neighbors)
            votes = [exampleData[n[1]].tolist()[-1] for n in neighbors]
            print("votes: ", votes)
            # print([exampleData[n[1]].tolist() for n in neighbors])
            # print(votes)
            if self.regression_data_set:
                # SIMPLE AVERAGE. TODO: IMPLEMENT GAUSSIAN KERNEL HERE
                estimate = sum(votes) / len(votes)
            else:
                most_common_class = self.most_common_class(votes)
                print("most common classes: ", most_common_class)
                if len(most_common_class) == 1:
                    estimate = most_common_class[0]
                else:
                    for i in range(len(neighbors)):
                        n_index = neighbors[i][1]
                        neighbor_class = exampleData[n_index][-1]
                        if  neighbor_class in most_common_class:
                            print("defining neighbor, index # ", n_index)
                            break
                    estimate = neighbor_class
            ground_truth =  new_sample.tolist()[-1]
            classifications.append([ground_truth, estimate])
        print(classifications)
        return classifications


    def most_common_class(self, votes: list) -> list:
        freqDict = Counter(votes)
        ordered_frequency = freqDict.most_common()
        most_common = [ordered_frequency[0]]
        for i in range(1, len(ordered_frequency)):
            next_most_common = ordered_frequency[i]
            if next_most_common[1] < most_common[-1][1]:
                break
            most_common.append(next_most_common)
        return [x[0] for x in most_common]

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
    data_set = "segmentation"
    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
    print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
    test = copy.deepcopy(tenFolds[0][0:5])
    training = np.concatenate(tenFolds[1:])
    print(len(test[0]))
    print(len(training))

    kNN = kNN(
        int(math.sqrt(len(full_set))), 
        full_set,
        categorical_attribute_indices[data_set],
        regression_data_set[data_set]
    )
    classifications = kNN.classify(training, test)

