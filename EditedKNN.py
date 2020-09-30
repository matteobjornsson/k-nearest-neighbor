#Written by 
#################################################################### MODULE COMMENTS ############################################################################4
##
#################################################################### MODULE COMMENTS ############################################################################

import kNN
import numpy as np
import copy


class EditedKNN:

    def __init__(self, error: float, k: int, data_set: np.ndarray, categorical_features: list, regression_data_set: bool):
        # initialize a knn object
        self.knn = kNN.kNN(self, k, data_set, categorical_features, regression_data_set)
        # error threshhold for regression classification
        self.error = error
        # store if this data set is a regression data set (True) or not (False)
        self.regression_data_set = regression_data_set

    # top level function called to create an edited knn dataset
    def reduce_data_set(self, data_set:np.ndarray) -> np.ndarray:
        # true while performance improves or does not degrde
        reduce = True
        # new set of data from which we will remove examples
        reduced_set = copy.deepcopy(data_set)
        
        # generate a first pass of classification
        results = self.knn.classify(data_set)
        # evaluate first pass perfromance, F1 for classification, MAE for regression
        performance = self.evaluate_performance(results, self.knn.regression_data_set)
        # remove all missclassified examples
        for i in range(len(results)):
            outcome = results[i]
            # if regression, remove examples outside of error threshold
            if self.regression_data_set:
                if (outcome[0] - self.error) <= outcome[1] <= (outcome[0] + self.error):
                    continue
                else:
                    reduced_set = np.delete(reduced_set, i, 0)
            # otherwise just remove any misclassified
            else:
                if outcome[0] != outcome[1]:
                    reduced_set = np.delete(reduced_set, i, 0)
        
        while reduce:
            print("whittle down data set here")
            break
        return reduced_set


    def evaluate_performance(self, classified_examples, data_set_type)-> float:
        # calculate loss function here
        return .95