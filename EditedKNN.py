#Written by 
#################################################################### MODULE COMMENTS ############################################################################4
##
#################################################################### MODULE COMMENTS ############################################################################

import kNN
from Results import Results
import numpy as np
import copy
from typing import Tuple


class EditedKNN:

    def __init__(self, error: float, k: int, data_set: np.ndarray, categorical_features: list, regression_data_set: bool):
        # initialize a knn object
        self.knn = kNN.kNN(self, k, data_set, categorical_features, regression_data_set)
        # error threshhold for regression classification
        self.error = error
        # store if this data set is a regression data set (True) or not (False)
        self.regression_data_set = regression_data_set
        self.results = Results()

    # top level function called to create an edited knn dataset
    def reduce_data_set(self, data_set:np.ndarray) -> Tuple[np.ndarray, list, float]:
        # array to keep track of edits
        reduction_record = []
        
        # run a first pass knn to classifiy examples in data set
        reduced_set = copy.deepcopy(data_set)
        results = self.knn.classify(reduced_set)
        performance = self.evaluate_performance(results, self.regression_data_set)
        reduction_record.append([copy.deepcopy(reduced_set), copy.deepcopy(results), copy(performance)])

        # while performance continues to improve or maintain, keep editing out
        # incorrect classifications from data set
        while True:
            reduced_set = self.remove_incorrect_estimates(reduced_set, results)

            results = self.knn.classify(reduced_set)
            performance = self.evaluate_performance(results, self.regression_data_set)
            reduction_record.append([copy.deepcopy(reduced_set), copy.deepcopy(results), copy(performance)])

            if reduction_record[-1][2] < reduction_record[-2][2]:
                break

        return reduction_record[-2]

    def remove_incorrect_estimates(self, data_set, results):
        # remove all missclassified examples
        for i in range(len(results)):
            ground_truth, estimate = results[i]
            # if regression, remove examples outside of error threshold
            if self.regression_data_set:
                lower_bound = ground_truth - self.error
                upper_bound = ground_truth + self.error
                if not (lower_bound <= estimate <= upper_bound):
                    data_set = np.delete(data_set, i, 0)
            # otherwise just remove any misclassified
            else:
                if ground_truth != estimate:
                    data_set = np.delete(data_set, i, 0)
        return data_set

    def evaluate_performance(self, classified_examples: list, regression: bool)-> float:
        loss= self.results.LossFunctionPerformance(regression, classified_examples)
        return loss[0]