import copy
from numpy.lib.type_check import real
import multiprocessing
from pandas.core import internals
import DataUtility
import kNN
import Results
import math
import numpy as np
import EditedKNN
import CondensedKNN
import kMeansClustering
import kMedoidsClustering

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

data_sets = ["segmentation", "vote"]
#  "glass", "fire", "machine", "abalone"
data_utility = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
results = Results.Results()

tuning_data = {}
for ds in data_sets:
    tuning_data[ds] = data_utility.get_tuning_data(ds)

bin_values = [.1, .25, .5, 1, 2, 4, 8]
delta_values = [.1, .25, .5, 1, 2, 4, 10]

def tune_knn(how_many_values_of_k: int, delta_values: list, bin_values: list):
    for ds in data_sets:
        data_dimension = tuning_data[ds].shape[1]-1
        last_k = len(tuning_data[ds])
        # step = int(last_k/how_many_values_of_k)
        # base_k_values = [1, 2, 3, 4, 5, 7, 8, 9, 10]
        # generated_k_values = [i*step for i in range(1, how_many_values_of_k + 1)]
        # k_values = base_k_values + generated_k_values
        k_values = [i for i in range(1, last_k)]
        for k in k_values:
            for delta in delta_values:
                if feature_data_types[ds] != "mixed":
                    alpha = 1
                    beta = 1
                else: 
                    alpha = 1
                    beta = alpha * delta
                for bin_width in bin_values:
                    knn = kNN.kNN(
                    #k value
                    k, 
                    # supply mixed, real, categorical nature of features
                    feature_data_types[ds],
                    #Feed in the categorical attribute indicies stored in a global array 
                    categorical_attribute_indices[ds],
                    #Store the data set key for the dataset name 
                    regression_data_set[ds],
                    # weight for real distance
                    alpha,
                    # weight for categorical distance
                    beta,
                    # kernel window size
                    bin_width,
                    #Set the dimensionality of the data set in KNN
                    data_dimension
                    )
                    classifications = knn.classify(tuning_data[ds], tuning_data[ds])
                    metadata = [ds, k, beta/alpha, bin_width]
                    results_set = results.StartLossFunction(regression_data_set[ds], classifications, metadata, "knn_tuning.csv")
                    if not regression_data_set[ds]:
                        break
                if feature_data_types[ds] != "mixed":
                    break


def tune_knn_parallel(data_set: str, k_value: int,  delta_value: int, bin_value: int):
    data_dimension = tuning_data[data_set].shape[1]-1
    if feature_data_types[data_set] != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * delta_value
    knn = kNN.kNN(
    #k value
    k_value, 
    # supply mixed, real, categorical nature of features
    feature_data_types[data_set],
    #Feed in the categorical attribute indicies stored in a global array 
    categorical_attribute_indices[data_set],
    #Store the data set key for the dataset name 
    regression_data_set[data_set],
    # weight for real distance
    alpha,
    # weight for categorical distance
    beta,
    # kernel window size
    bin_value,
    #Set the dimensionality of the data set in KNN
    data_dimension
    )
    classifications = knn.classify(tuning_data[data_set], tuning_data[data_set])
    metadata = [data_set, k_value, beta/alpha, bin_value]
    results_set = results.StartLossFunction(regression_data_set[ds], classifications, metadata, "knn_tuning.csv")


# tune_knn(values_of_k, delta_values, bin_values)

# manager = multiprocessing.Manager()
# data = manager.list()
# start = time.time()
pool = multiprocessing.Pool()
for ds in data_sets:
    tuning_length = len(tuning_data[ds])
    if 0 < tuning_length < 50:
        k_values = [i for i in range(1,30)]
    else: 
        remainder = tuning_length - 30
        init_k = [i for i in range(1,30)]
        step = int(remainder/10)
        extra_k = [30 + i*step for i in range(1, 10)]
        k_values = init_k + extra_k
    for k in k_values:
        for delta in delta_values:
            for b in bin_values:
                pool.apply_async(tune_knn_parallel, args=(ds, k, delta, b))
                if not regression_data_set[ds]:
                    break
            if feature_data_types[ds] != "mixed":
                break
pool.close()
pool.join()