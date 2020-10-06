#Written by Nick Stone Matteo Bjornsson 
##################################################################### MODULE COMMENTS #####################################################################
# This is the main function for the Naive Bayes project that was created by Nick Stone and Matteo Bjornsson. The purpose of this class is to import all of #
# The other classes and objects that were created and tie them together to run a series of experiments about the outcome stats on the data sets in question#
# The following program is just intended to run as an experiment and hyper parameter tuning will need to be done in each of the respective classes.        #
# It is important to note that the main datastructure that is used by these classes and objects is the pandas dataframe and numpy arrays, and lists, and   #
#is used to pass the datasets                                                                                                                              #
# Between all of the objects and functions that have been created. The classes are set up for easy modification for hyper parameter tuning.       
# 
# 
# This module was heavily edited for parallelism  after it was built                                                                                      #
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
import kMedoids_parallel
import multiprocessing
#TESTING LIBRARY 
import time 

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

data_sets = ["abalone", "segmentation", "vote", "glass", "fire", "machine"]

############### tuned hyperparameters: #####################################

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
    "abalone": 20
}

#################################################################################

##################### GET DATA FOR EXPERIMENT ###################################

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
# create a results object used to process all classification outcomes
results = Results.Results()


#################################################################################

##################### WORKER METHODS FOR MULTIPROCESSING ########################

# target function to run one 10fold instance of KNN, given the fold and data set
def knn_worker(q, fold, data_set):
    # get the ten folds
    tenFolds = experimental_data_sets[data_set][3]
    # pick the fold according to this run
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
    # Classify the fold against the remaining training data
    classifications = knn.classify(training, test)
    metadata = ["KNN", data_set]
    # calculate the performance
    results_set = results.LossFunctionPerformance(regression_data_set[data_set], classifications)
    data_point = metadata + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    # queue the results and return them 
    q.put(data_point_string)
    return(data_point_string)

# same notes as above but for EKNN
def eknn_worker(q, fold, data_set: str):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])
    #Append all data folds to the training data set
    remaining_folds = [x for i, x in enumerate(tenFolds) if i!=fold]
    training = np.concatenate(remaining_folds)    

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    data_type = feature_data_types[data_set]

    if regression_data_set[data_set]:
        error = tuned_error_value[data_set]
    else:
        error = 1

    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    eknn = EditedKNN.EditedKNN(
        error=error,
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
    return(data_point_string)

# same notes as KNN but for cKNN
def cknn_worker(q, fold, data_set: str):
    tenFolds = experimental_data_sets[data_set][3]
    test = copy.deepcopy(tenFolds[fold])
    #Append all data folds to the training data set
    remaining_folds = [x for i, x in enumerate(tenFolds) if i!=fold]
    training = np.concatenate(remaining_folds)    

    data_dimension = len(experimental_data_sets[data_set][0]) - 1
    data_type = feature_data_types[data_set]
    
    if regression_data_set[data_set]:
        error = tuned_error_value[data_set]
    else:
        error = 1
    if data_type != "mixed":
        alpha = 1
        beta = 1
    else: 
        alpha = 1
        beta = alpha * tuned_delta_value[data_set]

    cknn = CondensedKNN.CondensedKNN(
        error=error,
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
    return(data_point_string)

# Same notes as KNN but for kmeans
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
        # note all instances assume real values
        data_type="real",
        # and no categorical features
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
    return(data_point_string)

# same as above but for kmedoids
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

    kmedoids = kMedoids_parallel.kMedoids_parallel(
        kNeighbors=tuned_k[data_set],
        kValue=15,
        dataSet=experimental_data_sets[data_set][1],
        data_type=feature_data_types[data_set],
        categorical_features=categorical_attribute_indices[data_set],
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
    return(data_point_string)

#################################################################################

#####################    EXPERIMENT DRIVER    ########################

# target function to start a process that writes all results to file
def data_writer(q, filename):
    while True:
        with open(filename, 'a') as f:
            data_string = q.get()
            if data_string == 'kill':
                f.write('\n')
                break
            f.write(data_string + '\n')

# main function, drives the entire 10 fold cross validation across all 6 data sets
# for each of the five algorithms
def main(): 
    print("Program Start")
    # specify the filename
    filename = "experimental_data-medoid_fix.csv"
    # start the multiprocessing helpers 
    manager = multiprocessing.Manager()
    q = manager.Queue()
    start = time.time()

    # start a file writer to save results
    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()
    pool = multiprocessing.Pool()
    
    # fire off an instance of each algorithm (minus medoids) for each data set and each fold
    results = []
    for ds in data_sets:
        # one for each fold in 10-fold cross validation
        for i in range(10):
            results.append(pool.apply_async(knn_worker, args=(q, i, ds)))
            results.append(pool.apply_async(eknn_worker, args=(q, i, ds)))
            results.append(pool.apply_async(cknn_worker, args=(q, i, ds)))
            results.append(pool.apply_async(kmeans_worker, args=(q, i, ds)))
   
    pool.close()
    pool.join()
    q.put('kill')
    writer.join()
    # these two lines are for some notion of what's happening and also r.get() 
    # forces an exception to be raised if one of the processes failed, otherwise the exception is suprressed
    for r in results:
        print(r.get())
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')

    #########################   SEPARATE OUT MEDOIDS THEY TAKE FOREVER  ########
    ######## does the same thing as above but just for medoids
    
    q2 = manager.Queue()
    start = time.time()

    writer2 = multiprocessing.Process(target=data_writer, args=(q2,filename))
    writer2.start()
    results2 = []

    for ds in data_sets:
        if ds == "fire" or ds == "machine":
            for j in range(10):
                results2.append(kmedoids_worker(q2,j, ds))

    q2.put('kill')
    writer2.join()
    for r in results2:
        print(r)
    elapsed_time = time.time() - start
    print("Elapsed time: ", elapsed_time, 's')

    #Print some meta data to the screen letting the user know the program is ending 
    print("Program End")
#On invocation run the main method

main()


