#Written by Matteo Bjornsson Edited by Nick Stone 
#################################################################### MODULE COMMENTS ############################################################################4
##
#################################################################### MODULE COMMENTS ############################################################################

import kNN, Results, DataUtility
import numpy as np
import copy, math


class EditedKNN:

    def __init__(self, error: float, k: int, data_type: str, categorical_features: list, regression_data_set: bool, alpha:float, beta:float, h:float, d:int):
        # initialize a knn object
        self.knn = kNN.kNN(k, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        # error threshhold for regression classification
        self.error = error
        # store if this data set is a regression data set (True) or not (False)
        self.regression_data_set = regression_data_set
        self.results = Results.Results()

    def classify_in_place(self, data: np.ndarray):
        results = []
        for i in range(len(data)):
            sample = data[i,:].reshape(1,data.shape[1])
            set_minus_sample = np.delete(data,i,0)
            # print("sample: ", sample, type(sample), '\n', "Remainder: ", set_minus_sample, type(set_minus_sample))
            results.append(self.knn.classify(set_minus_sample, sample )[0])
        return results

    # top level function called to create an edited knn dataset
    def reduce_data_set(self, data_set:np.ndarray) -> np.ndarray:
        # array to keep track of edits
        reduction_record = []
        
        # run a first pass knn to classifiy examples in data set
        # initialize set as the input data set
        reduced_set = copy.deepcopy(data_set)
        # classify all examples in the data set
        results = self.classify_in_place(data_set)
        # evaluate the estimation performance
        performance = self.evaluate_performance(results, self.regression_data_set)
        # save data set, results, and performance to array
        reduction_record.append([copy.deepcopy(reduced_set), copy.deepcopy(results), copy.copy(performance)])

        # while performance continues to improve or maintain, keep editing out
        # incorrect classifications from data set
        while True:
            # remove all examples that were incorrectly estimated
            reduced_set = self.remove_incorrect_estimates(reduced_set, results)

            # re-estimate data set and save results
            results = self.classify_in_place(reduced_set)
            performance = self.evaluate_performance(results, self.regression_data_set)
            reduction_record.append([copy.deepcopy(reduced_set), copy.deepcopy(results), copy.copy(performance)])

            # if the most recent knn performance is worse than the last one,
            # stop editing data set
            if reduction_record[-1][2] < reduction_record[-2][2] or len(reduction_record[-1][0]) == len(reduction_record[-2][0]):
                break
        # return the next-to-last edited data set
        return copy.deepcopy(reduction_record[-2][0])

    # remove all missclassified examples until performance degrades or no more examples are removed
    def remove_incorrect_estimates(self, data_set: np.ndarray, results: list) -> np.ndarray:
        indices_to_remove = []
        # inspect every example. If it is incorrectly classified (or not estimated
        # correctly within a margin of error) record its index for deletion.
        for i in range(len(results)):
            ground_truth, estimate = results[i]
            # if regression, remove examples outside of error threshold
            if self.regression_data_set:
                lower_bound = ground_truth - self.error
                upper_bound = ground_truth + self.error
                if not (lower_bound <= estimate <= upper_bound):
                    indices_to_remove.append(i)
            # otherwise just remove any misclassified
            else:
                if ground_truth != estimate:
                    indices_to_remove.append(i)
        # remove all the examples marked for deletion
        data_set = np.delete(data_set, indices_to_remove, 0)
        return data_set

    # given a set of estimates, evaluate the performance with F1 for classification
    # or MAE for regression. 
    def evaluate_performance(self, classified_examples: list, regression: bool)-> float:
        loss= self.results.LossFunctionPerformance(regression, classified_examples)
        return loss[0]
    
    # simple classify method that mirrors KNN, exept with an edited training set
    def classify(self, training, test):
        edited_training = self.reduce_data_set(training)
        return self.knn.classify(edited_training, test)



####################################### UNIT TESTING #################################################
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

    data_sets = ["segmentation", "vote", "glass", "fire", "machine"]

    total_stats = []
    for data_set in data_sets:
        print(data_set)
        #print(regression_data_set.get(key))
        #Create a data utility to track some metadata about the class being Examined
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        #Store off the following values in a particular order for tuning, and 10 fold cross validation 
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        #Print the data to the screen for the user to see 
        # print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        #Create and store a copy of the first dataframe of data 
        test = copy.deepcopy(tenFolds[0])
        #Append all data folds to the training data set
        training = np.concatenate(tenFolds[1:])

        k = int(math.sqrt(len(training)))
        # dimensionality of data set
        d = len(headers) - 1
        # is this data set a regression data set? 
        regression = regression_data_set[data_set]

        if regression:
            regression_error = np.mean(training[:,-1], dtype=np.float64)
        else:
            regression_error = 0

        eknn = EditedKNN(
            error=regression_error,
            k=k,
            data_type=feature_data_types[data_set],
            categorical_features=categorical_attribute_indices[data_set],
            regression_data_set=regression,
            alpha=1,
            beta=1,
            h=.5, 
            d=d
        )

        classifications = eknn.classify(training, test)
        for c in classifications:
            print(c)
        print("Regression:", regression)
        print("final stats:" )
        stats = eknn.results.LossFunctionPerformance(regression, classifications)
        total_stats.append([data_set, stats])
    print(total_stats)
####################################### UNIT TESTING #################################################