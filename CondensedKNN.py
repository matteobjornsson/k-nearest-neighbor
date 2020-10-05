#Written By Nick Stone and Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import random, copy, math
import numpy as np
import kNN, Results, DataUtility
class CondensedKNN:

    def __init__(self, 
        error: float,
        k: int,
        data_type: str,
        categorical_features: list,
        regression_data_set: bool,
        alpha:float,
        beta:float,
        h:float,
        d:int):

        # initialize a knn object
        self.knn = kNN.kNN(k, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        self.nn = kNN.kNN(1, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        # error threshhold for regression classification
        self.error = error
        # store if this data set is a regression data set (True) or not (False)
        self.regression_data_set = regression_data_set
        self.results = Results.Results()

    def condense_data_set(self, data: np.ndarray) -> np.ndarray:
        # pick a random sample and initialize Z and X \ Z
        i = random.randint(0, len(data)-1)
        Z = data[i,:].reshape(1,data.shape[1])
        X = np.delete(data,i,0)

        indices = list(range(len(X)))
        random.shuffle(indices)
        # for every example in X, randomly selected, find its nearest neighbor in Z. If their 
        # classes do not match (or response variable not the same within some error),
        # move the example from X to Z. 
        for x in X:
            # reshape example x array to match expected argument structure for classify
            x = x.reshape(1, X.shape[1])
            # classify x by using single nearest neighbor in set Z. (in other
            # words, "look up" the class/estimate value of the nearest neighbor in Z)
            compare = self.nn.classify(Z, x)
            x_value = compare[0][0]
            nearest_neighbor_in_Z = compare[0][1]

            # if new sample x is not valued the same as the nearest neighbor in Z
            # within some error tolerance, add it set Z
            if self.regression_data_set:
                lower_bound = x_value - self.error
                upper_bound = x_value + self.error
                if not (lower_bound <= nearest_neighbor_in_Z <= upper_bound):
                    Z = np.concatenate((Z, x), axis=0)
            # Same for classification, add to Z if not the same class
            else:
                if x_value != nearest_neighbor_in_Z:
                    Z = np.concatenate((Z, x), axis=0)
        return Z

    # classify the test data against a condensed training set using KNN
    def classify(self, training, test):
        condensed_training = self.condense_data_set(training)
        return self.knn.classify(condensed_training, test)

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

    data_sets = ["segmentation", "vote", "glass", "fire", "machine", "abalone"]

    total_stats = []
    for i in range(5):
        data_set = data_sets[i]
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
            regression_error = np.mean(training[:,-1], dtype=np.float64) * .1
        else:
            regression_error = 0

        cknn = CondensedKNN(
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

        classifications = cknn.classify(training, test)
        for c in classifications:
            print(c)
        print("Regression:", regression)
        print("final stats:" )
        stats = cknn.results.LossFunctionPerformance(regression, classifications)
        total_stats.append([data_set, stats])
    print(total_stats)