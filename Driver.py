#Written by Nick Stone edited by Matteo Bjornsson 
##################################################################### MODULE COMMENTS #####################################################################
# This is the main function for the Naive Bayes project that was created by Nick Stone and Matteo Bjornsson. The purpose of this class is to import all of #
# The other classes and objects that were created and tie them together to run a series of experiments about the outcome stats on the data sets in question#
# The following program is just intended to run as an experiment and hyper parameter tuning will need to be done in each of the respective classes.        #
# It is important to note that the main datastructure that is used by these classes and objects is the pandas dataframe and numpy arrays, and lists, and   #
#is used to pass the datasets                                                                                                                              #
# Between all of the objects and functions that have been created. The classes are set up for easy modification for hyper parameter tuning.                #
##################################################################### MODULE COMMENTS #####################################################################

import copy
from numpy.lib.type_check import real
import DataUtility
import kNN
import Results
import math
import numpy as np
import EditedKNN
import CondensedKNN
import kMeansClustering
import kMedoidsClustering
import multiprocessing
#TESTING LIBRARY 
import time 

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

data_sets = [ "segmentation", "vote", "glass", "fire", "machine", "abalone"]


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

tuned_cluster_number = {
    "segmentation": 80,
    "vote": 15,
    "glass": 60,
    # not sure about fire, weird behavior
    "fire": 60,
    "machine": 50,
    "abalone": 50

}

experimental_data_sets = {}
#For ecah of the data set names that we have stored in a global variable 
for data_set in data_sets:
    #Create a data utility to track some metadata about the class being Examined
    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    #Store off the following values in a particular order for tuning, and 10 fold cross validation 
    # return from generate experiment data: [headers, full_set, tuning_data, tenFolds]
    if regression_data_set.get(data_set) == False: 
        experimental_data_sets[data_set]= du.generate_experiment_data_Categorical(data_set)
    else:
        experimental_data_sets[data_set] = du.generate_experiment_data(data_set)

results = Results.Results()

def knn_worker(q, fold, data_set):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])
    #Append all data folds to the training data set
    remaining_folds = [x for i, x in enumerate(tenFolds) if i!=fold]
    training = np.concatenate(remaining_folds)    

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    if feature_data_types[data_set] != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    knn = kNN.kNN(
        #Feed in the square root of the length 
        tuned_k[data_set], 
        # supply mixed, real, categorical nature of features
        feature_data_types[data_set],
        #Feed in the categorical attribute indicies stored in a global array 
        categorical_attribute_indices[data_set],
        #Store the data set key for the dataset name 
        regression_data_set[data_set],
        # weight for real distance
        alpha=alpha,
        # weight for categorical distance
        beta=beta,
        # kernel window size
        h=tuned_bin_value[data_set],
        #Set the dimensionality of the data set in KNN
        d=data_dimension
    )
    classifications = knn.classify(training, test)
    metadata = ["KNN", data_set]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

def eknn_worker(q, fold, data_set: str):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])
    #Append all data folds to the training data set
    remaining_folds = [x for i, x in enumerate(tenFolds) if i!=fold]
    training = np.concatenate(remaining_folds)    

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    data_type = feature_data_types[data_set]

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    eknn = EditedKNN.EditedKNN(
        error=tuned_error_value[data_set],
        k=tuned_k[data_set],
        data_type=feature_data_types[data_set],
        categorical_features=categorical_attribute_indices[data_set],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension
        )

    classifications = eknn.classify(training, test)
    metadata = ["Edited", data_set]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

def cknn_worker(q, fold, data_set: str):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])
    #Append all data folds to the training data set
    remaining_folds = [x for i, x in enumerate(tenFolds) if i!=fold]
    training = np.concatenate(remaining_folds)    

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    data_type = feature_data_types[data_set]
    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    cknn = CondensedKNN.CondensedKNN(
        error=tuned_error_value[data_set],
        k=tuned_k[data_set],
        data_type=feature_data_types[data_set],
        categorical_features=categorical_attribute_indices[data_set],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension
        )

    classifications = cknn.classify(training, test)
    metadata = ["Condensed", data_set]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

def kmeans_worker(q, fold, data_set:str):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    data_type = feature_data_types[data_set]

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    kmeans = kMeansClustering.kMeansClustering(
        kNeighbors=tuned_k[data_set],
        kValue=tuned_cluster_number[data_set],
        dataSet=experimental_data_sets[data_set][1],
        data_type="real",
        categorical_features=[],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension,
        name=data_set,
        Testdata=test
        )

    classifications = kmeans.classify()
    metadata = ["K-Means", data_set]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

def kmedoids_worker(q, fold, data_set:str):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    data_type = feature_data_types[data_set]

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    kmedoids = kMedoidsClustering.kMedoidsClustering(
        kNeighbors=tuned_k[data_set],
        kValue=tuned_cluster_number[data_set],
        dataSet=experimental_data_sets[data_set][1],
        data_type="real",
        categorical_features=[],
        regression_data_set=regression_data_set[data_set],
        alpha=alpha,
        beta=beta,
        h=tuned_bin_value[data_set], 
        d=data_dimension,
        Testdata=test
        )

    classifications = kmedoids.classify()
    metadata = ["K-Medoids", data_set]
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    q.put(data_point_string)

def data_writer(q, filename):
    while True:
        with open(filename, 'a') as f:
            data_string = q.get()
            if data_string == 'kill':
                f.write('\n')
                break
            f.write(data_string + '\n')

def main(): 
    print("Program Start")
    filename = "experimental_data.csv"
    manager = multiprocessing.Manager()
    q = manager.Queue()
    start = time.time()

    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()
    pool = multiprocessing.Pool()
    
    for ds in data_sets:
        for i in range(10):
            pool.apply_async(knn_worker, args=(q, i, ds))
            pool.apply_async(eknn_worker, args=(q, i, ds))
            pool.apply_async(cknn_worker, args=(q, i, ds))
            pool.apply_async(kmeans_worker, args=(q, i, ds))
            pool.apply_async(kmedoids_worker, args=(q, i, ds))
            
    pool.close()
    pool.join()
    q.put('kill')
    writer.join()
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')

    #Print some meta data to the screen letting the user know the program is ending 
    print("Program End")
#On invocation run the main method

main()

# print("Program Start")
# filename = "experimental_data.csv"
# manager = multiprocessing.Manager()
# q = manager.Queue()
# start = time.time()

# writer = multiprocessing.Process(target=data_writer, args=(q,filename))
# writer.start()

# kmedoids_worker(q, 1, "fire")

# q.put('kill')
# writer.join()