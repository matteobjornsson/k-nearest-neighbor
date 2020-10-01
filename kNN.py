#Written by Matteo Bjornsson edited by Nick Stone 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import copy, math
from collections import Counter
import RealDistance, HammingDistance
import heapq
import pandas as pd
import numpy as np
import DataUtility
import KernelSmoother

class kNN:

    def __init__(
            self,
            k: int,
            data_type: str,
            categorical_features: list,
            regression_data_set: bool,
            alpha: float,
            beta: float,
            h: float,
            d: int):
        self.k = k
        self.categorical_features = categorical_features
        self.regression_data_set = regression_data_set
        self.data_type = data_type

        # tools for calculating distance and regression estimation:
        # Real Distance
        self.rd = RealDistance.RealDistance()
        # tunable parameter, weight of real distance value
        self.alpha = alpha
        self.hd = HammingDistance.HammingDistance()
        # tunable parameter, weight of categorical distance value
        self.beta = beta
        # bandwidth parameter for the gaussian kernel
        self.h = h
        # kernel smoother, window size and data set dimensionality as inputs
        self.kernel = KernelSmoother.KernelSmoother(self.h, d)

    # this code block was replaced with a hardcoded dictionary indicating type of each set

    # # function determines the nature of the data set features: real, categorical, or mixed
    # def feature_data_types(self, data_set: np.ndarray, categorical_features: list) -> str:
    #     feature_count = data_set.shape[1]-1
    #     # if the number of features is the same as the number of columns that
    #     # are categorical, the entire feature set is categorical
    #     if len(categorical_features) == feature_count:
    #         return "categorical"
    #     # else if the list of categorical features is non-zero, feature set is mixed
    #     elif len(categorical_features) > 0:
    #         return "mixed"
    #     # last remaining option, all features are real
    #     else:
    #         return "real"

    def get_k_neighbors(self, exampleData: np.ndarray, new_sample: list, k) -> list:
        #If the dataset has both categorical and real values 
        if self.data_type == "mixed":
            #Store off those values which are cetegorical and real into 2 sperate lists 
            sample_cat = [new_sample[i] for i in range(len(new_sample)) if i in self.categorical_features]
            sample_real = [new_sample[j] for j in range(len(new_sample)) if j not in self.categorical_features]
        #Create an empty new list 
        neighbors = []
        #For each of those values in the exaple data set that we take in 
        for n in range(len(exampleData)):
            #Set the value to be x 
            x = exampleData[n].tolist()[:-1]
            #Store off the response variable 
            responseVariable = exampleData[n].tolist()[-1]
            #If the data is real 
            if self.data_type == "real":
                #Calculate Real distance distance 
                distance = self.rd.Distance(x, new_sample)
            #If the data is categorical 
            elif self.data_type == "categorical":
                #Use hamming distance to calculate distance 
                distance = self.hd.Distance(x, new_sample)
            #Otherwise 
            else:
                #Set a list of categorical values 
                x_cat = [x[k] for k in range(len(x)) if k in self.categorical_features]
                #Set the list of real values
                x_real = [x[l] for l in range(len(x)) if l not in self.categorical_features]
                #Set the distance and store the value 
                distance = self.alpha * self.rd.Distance(x_real, sample_real) + self.beta * self.hd.Distance(x_cat, sample_cat)
                # if n < 5:
                #     print("sample real: ", sample_real, "x real: ", x_real, "real distance: ", self.rd.Distance(x_real, sample_real))
                #     print("sample cat: ", sample_cat, "x cat: ", x_cat, "categorical distance: ", self.hd.Distance(x_cat,sample_cat))
            #Push the distance and neighbors onto the heap 
            heapq.heappush(neighbors, (distance, n, responseVariable))
        #Store off the neighbors with the smallest distance 
        kNeighbors = heapq.nsmallest(k, neighbors)
        #Return the neighbors with the smallest distance 
        return kNeighbors

    def classify(self, exampleData: np.ndarray, testData: np.ndarray) -> list:
        #Create a new list 
        classifications = []
        #For each of the datapoints in the test data set we take in 
        for new_sample in testData:
            #Create a list from the data source that we take in 
            new_vector = new_sample.tolist()[:-1]
            #Set the number of neighbors to the K neighbors function 
            neighbors = self.get_k_neighbors(exampleData, new_vector, self.k)
            # printNeighbors = [(f"Distance: {n[0]}", f"example: {n[1]}", exampleData[n[1]].tolist()) for n in neighbors]
            
            # # code just for printout:
            # print(f"Neighbors of {new_vector}:")
            # count = 0
            # for index in range(len(neighbors)): 
            #     if index <= 3 or index >= len(neighbors)-3:
            #         x = printNeighbors[index]
            #         print(x[2], x[0], x[1])
            #     if 3 < index < len(neighbors)-3 and len(neighbors) > 6 and count < 1:
            #         count += 1
            #         print("...")
            #Create a list of votes for each of the neighbors above 
            votes = [exampleData[n[1]].tolist()[-1] for n in neighbors]
            # print("votes: ", votes)
            # print([exampleData[n[1]].tolist() for n in neighbors])
            # print(votes)
            #If the data set is a regression data set 
            if self.regression_data_set:
                #Use a guassian kernel to generate a kernel estimate 
                estimate = self.kernel.estimate(neighbors)
                # print("kernel estimate: ", estimate)
                # print()\
            #If it is a categorical dataset 
            else:
                #Get the most common class depending on the vote data above 
                most_common_class = self.most_common_class(votes)
                # print("most common classes: ", most_common_class)
                #If their is only one common class 
                if len(most_common_class) == 1:
                    #Set the estimate to be the most common class occurence in votes 
                    estimate = most_common_class[0]
                #More than one most common class 
                else:
                    #Loop through all of the neighbors 
                    for i in range(len(neighbors)):
                        #Set the index to grab information about the neighbor 
                        n_index = neighbors[i][1]
                        #Get the neighbors class 
                        neighbor_class = exampleData[n_index][-1]
                        #If the class is in that of the most common class 
                        if  neighbor_class in most_common_class:
                            # print("defining neighbor, index # ", n_index)
                            #Break out of th eloop 
                            break
                    #Set the estimate to the neighbor class 
                    estimate = neighbor_class
                    # print("Estimate: ", estimate, '\n')
            #Store off the ground truth value 
            ground_truth =  new_sample.tolist()[-1]
            #Add the ground tryth and estimate to list to be returned 
            classifications.append([ground_truth, estimate])
        # for clss in classifications: print(f"Ground truth: {clss[0]}, Estimate: {clss[1]}")
        #Return the classification list of ground truth and the estimate 
        return classifications


    def most_common_class(self, votes: list) -> list:
        #Get a dictionary of most common classes based on the list that we are passed in 
        freqDict = Counter(votes)
        #Order the list to be by the most common classes 
        ordered_frequency = freqDict.most_common()
        #Set the most common list to hold the neigbor that appeared the most frequently
        most_common = [ordered_frequency[0]]
        #Loop through the number of individual neighbor classes there are 
        for i in range(1, len(ordered_frequency)):
            #Find the next most common neighbor depending on the position in the loop   
            next_most_common = ordered_frequency[i]
            #If there are less neighbors with the class we are looking at then previously stored 
            if next_most_common[1] < most_common[-1][1]:
                #Break out of the loop 
                break
            #Append the next most common neighbor to the list of neighbors
            most_common.append(next_most_common)
        #Return the most common Neighbor
        return [x[0] for x in most_common]





#TESTING THE KNN DATA OBJECT 
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
    data_sets = ["segmentation", "vote", "glass", "fire", "machine", "abalone"]

    regression = [x for x in data_sets if regression_data_set[x]]

    for data_set in regression:
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        test = copy.deepcopy(tenFolds[0][0:5])
        training = np.concatenate(tenFolds[1:])
        print(len(test[0]))
        print(len(training))

        knn = kNN(
            int(math.sqrt(len(training))),
            training,
            categorical_attribute_indices[data_set],
            regression_data_set[data_set],
            1,2,.5
        )
        classifications = knn.classify(training, test)
