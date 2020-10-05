import time
from numpy.lib.type_check import real
import multiprocessing

from pandas.core.arrays import categorical
import DataUtility
import kNN, EditedKNN, CondensedKNN, kMeansClustering, kMedoidsClustering
import Results
import numpy as np

##################### DATA SET METADATA ###########################################
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

tuned_k = {
    "segmentation": 2,
    "vote": 5,
    "glass": 2,
    "fire": 2,
    "machine": 5,
    "abalone": 12
}
tuned_bin_value = {
    "segmentation": .25,
    "vote": .25,
    "glass": .25,
    "fire": .1,
    "machine": .25,
    "abalone": .1
}

tuned_delta_value = {
    "segmentation": .25,
    "vote": .25,
    "glass": .25,
    "fire": .5,
    "machine": .1,
    "abalone": .5
}

tuned_error_value = {
    "fire": 1,
    "abalone": 1,
    "machine":2
}

regression_variable_mean = {
    "abalone": 9.9,
    "machine": 105,
    "fire": 12.8
}
    

data_sets = ["segmentation", "vote", "glass", "fire", "machine", "abalone"]

#################################################################################

##################### GET DATA FOR TUNING #######################################


data_utility = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
results = Results.Results()

tuning_data_dict = {}
full_data_dict = {}
for ds in data_sets:
    headers, full_set, tuning_data, tenFolds  = data_utility.generate_experiment_data(ds)
    tuning_data_dict[ds] = tuning_data
    full_data_dict[ds] = full_set
    print("tuning set length: ", ds, len(tuning_data_dict[ds]))

#################################################################################

##################### WORKER METHODS FOR MULTIPROCESSING ########################


def tune_knn_parallel_worker(q, data_set: str, k_value: int,  delta_value: int, bin_value: int):
    # print('inside function', data_set, k_value)
    data_dimension = tuning_data_dict[data_set].shape[1]-1
    if feature_data_types[data_set] != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * delta_value
    # print("this far", data_set)
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
    classifications = knn.classify(tuning_data_dict[data_set], tuning_data_dict[data_set])
    metadata = [data_set, k_value, beta/alpha, bin_value]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    # print(data_point_string)
    q.put(data_point_string)
    # print("q.get(): ", q.get())



def tune_eknn_parallel_worker(q, data_set: str, error_value: float):
    data_dimension = tuning_data_dict[data_set].shape[1]-1
    data_type = feature_data_types[data_set]

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    eknn = EditedKNN.EditedKNN(
        error=error_value,
        k=tuned_k[data_set],
        data_type=feature_data_types[data_set],
        categorical_features=categorical_attribute_indices[data_set],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension
        )

    classifications = eknn.classify(full_data_dict[data_set], tuning_data_dict[data_set])
    metadata = [data_set, error_value]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

def tune_cknn_parallel_worker(q, data_set:str, error_value: float):
    data_dimension = tuning_data_dict[data_set].shape[1]-1
    data_type = feature_data_types[data_set]

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    cknn = CondensedKNN.CondensedKNN(
        error=error_value,
        k=tuned_k[data_set],
        data_type=feature_data_types[data_set],
        categorical_features=categorical_attribute_indices[data_set],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension
        )

    classifications = cknn.classify(full_data_dict[data_set], tuning_data_dict[data_set])
    metadata = [data_set, error_value]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)
    
def tune_kmeans_parallel_worker(q, data_set:str, kNeighbors: int, kClusters: int):
    data_dimension = tuning_data_dict[data_set].shape[1]-1
    data_type = feature_data_types[data_set]

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    kmeans = kMeansClustering.kMeansClustering(
        kNeighbors=kNeighbors,
        kValue=kClusters,
        dataSet=full_data_dict[data_set],
        data_type="real",
        categorical_features=[],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension,
        name=data_set,
        Testdata=tuning_data_dict[data_set]
        )

    classifications = kmeans.classify()
    metadata = [data_set, kNeighbors, kClusters]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

# source of data writer asynch code 'data_writer'
# https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
def data_writer(q, filename):
    while True:
        with open(filename, 'a') as f:
            data_string = q.get()
            if data_string == 'kill':
                f.write('\n')
                break
            f.write(data_string + '\n')

#################################################################################

#####################  TUNING METHODS  ##########################################

bin_values = [.1, .25, .5, 1, 2, 4, 8]
delta_values = [.1, .25, .5, 1, 2, 4, 10]

def knn_asynch_tuner(filename):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    start = time.time()
    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()

    pool = multiprocessing.Pool()

    for ds in data_sets:
        tuning_length = len(tuning_data_dict[ds])
        if tuning_length < 30:
            k_values = [i for i in range(2, tuning_length)]
        elif 0 < tuning_length < 50:
            k_values = [i for i in range(2,30)]
        else: 
            remainder = tuning_length - 30
            init_k = [i for i in range(2,30)]
            step = int(remainder/10)
            extra_k = [30 + i*step for i in range(1, 10)]
            k_values = init_k + extra_k
        for k in k_values:
            for delta in delta_values:
                for b in bin_values:
                    pool.apply_async(tune_knn_parallel_worker, args=(q, ds, k, delta, b))
                    if not regression_data_set[ds]:
                        break
                if feature_data_types[ds] != "mixed":
                    break
    pool.close()
    pool.join()
    q.put('kill')
    writer.join()
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')


def eknn_asynch_error_tuner(filename):

    manager = multiprocessing.Manager()
    q = manager.Queue()
    start = time.time()
    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()

    pool = multiprocessing.Pool()
    
    for ds in data_sets:
        if regression_data_set[ds]:

            regression_mean = regression_variable_mean[ds]
            small_step = .001*regression_mean
            small_error_values = [small_step + small_step*i for i in range(100)]
            big_step = 100*small_step
            large_error_values = [big_step + big_step*j for j in range(30)]
            error_values = small_error_values + large_error_values

            for e in error_values:
                pool.apply_async(tune_eknn_parallel_worker, args=(q, ds, e))

    pool.close()
    pool.join()
    q.put('kill')
    writer.join()
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')

def cknn_asynch_error_tuner(filename):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    start = time.time()
    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()

    pool = multiprocessing.Pool()
    
    for ds in data_sets:
        if regression_data_set[ds]:

            regression_mean = regression_variable_mean[ds]
            small_step = .001*regression_mean
            small_error_values = [small_step + small_step*i for i in range(100)]
            big_step = 100*small_step
            large_error_values = [big_step + big_step*j for j in range(30)]
            error_values = small_error_values + large_error_values

            for e in error_values:
                pool.apply_async(tune_cknn_parallel_worker, args=(q, ds, e))

    pool.close()
    pool.join()
    q.put('kill')
    writer.join()
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')

def kmeans_asynch_error_tuner(filename):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    start = time.time()
    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()

    pool = multiprocessing.Pool()
    
    for ds in data_sets:
        for i in range(1,30):
            for j in range(1,30):
                pool.apply_async(tune_kmeans_parallel_worker, args=(q, ds, i, j))

    pool.close()
    pool.join()
    q.put('kill')
    writer.join()
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')



#knn_asynch_tuner('knn_tuning2.csv')

# eknn_asynch_error_tuner('edited_error_full.csv')

# cknn_asynch_error_tuner('condensed_error_full.csv')

kmeans_asynch_error_tuner('kmeans_clustering_full.csv')
