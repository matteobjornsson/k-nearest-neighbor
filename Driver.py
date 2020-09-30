#Written by Nick Stone edited by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################

import copy
import DataUtility
import kNN
import Results
import math
import numpy as np
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

def PlotCSV():
    pass


def main(): 
    #Print some data to the screen to let the user know we are starting the program 
    print("Program Start")
    #For ecah of the data set names that we have stored in a global variable 
    for key in regression_data_set.keys():
        #store which dataset we are wroking on 
        data_set = key
        #print(regression_data_set.get(key))
        #Create a data utility to track some metadata about the class being Examined
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        #Store off the following values in a particular order for tuning, and 10 fold cross validation 
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        #Print the data to the screen for the user to see 
        print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        #Create and store a copy of the first dataframe of data 
        test = copy.deepcopy(tenFolds[0])
        #Append all data folds to the training data set
        training = np.concatenate(tenFolds[1:])
        #Print the length of the first array for debugging
        #print(len(test[0]))
        #Print the length of the training data set for testing 
        #print(len(training))
        #Create a KNN data object and insert the following data 
        knn = kNN.kNN(
            #Feed in the square root of the length 
            int(math.sqrt(len(full_set))), 
            #Feed in the full set 
            full_set,
            #Feed in the categorical attribute indicies stored in a global array 
            categorical_attribute_indices[data_set],
            #Store the data set key for the dataset name 
            regression_data_set[data_set]
        )
        #Store and run the classification associated with the KNN algorithm 
        classifications = knn.classify(training, test)
        #Create a Results function to feed in the KNN Classification data and produce Loss Function Values 
        ResultObject = Results.Results() 
        #Create a list and gather some meta data for a given experiment, so that we can pipe all of the data to a file for evaluation
        MetaData = list() 
        MetaData.append(key)
        MetaData.append("TRIAL: ")
        print(classifications)
        print(regression_data_set.get(key))
        #Create a list to store the Results that are generated above FOR TESTING 
        ResultSet = ResultObject.StartLossFunction(regression_data_set.get(key),classifications, MetaData)
        print(ResultSet)
        

    #Print some meta data to the screen letting the user know the program is ending 
    print("Program End")
#On invocation run the main method
main()