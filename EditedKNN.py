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
        iter_count = 0
        reduction_record = []

        reduced_set = copy.deepcopy(data_set)


        results = self.knn.classify(reduced_set)
        performance = self.evaluate_performance(results, self.regression_data_set)
        reduction_record.append([reduced_set, results, performance])
        iter_count +=1

        while True:
            reduced_set = self.remove_incorrect_estimates(reduced_set, results)
            results = self.knn.classify(reduced_set)
            performance = self.evaluate_performance(results, self.regression_data_set)
            reduction_record.append([reduced_set, results, performance])
            iter_count +=1
            if reduction_record[-1] < reduction_record[-2]:
                break
        return reduced_set

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

    def evaluate_performance(self, classified_examples, data_set_type)-> float:
        # calculate loss function here
        return .95