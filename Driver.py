#Written by Nick Stone edited by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################

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
#TESTING LIBRARY 
import time 

categorical_attribute_indices = {
    
    "vote": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
   
    "fire": [0,1,2,3],
    "machine": [0,1],
    "abalone": [0]
}

regression_data_set = {
    
    "vote": False,

    "fire": True,
    "machine": True,
    "abalone": True
}

feature_data_types = {
    
    "vote": 'categorical',
    
    "fire": 'mixed',
    "machine": 'mixed',
    "abalone": 'mixed'
}

data_sets = [ "vote",  "fire", "machine", "abalone"]

def PlotCSV():
    pass


def main(): 
    #Print some data to the screen to let the user know we are starting the program 
    print("Program Start")
    k = 0 
    for zz in range(30):
         
        #For ecah of the data set names that we have stored in a global variable 
        for data_set in data_sets:
            
            #print(regression_data_set.get(key))
            #Create a data utility to track some metadata about the class being Examined
            du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
            #Store off the following values in a particular order for tuning, and 10 fold cross validation 
            if regression_data_set.get(data_set) == False: 
                headers, full_set, tuning_data, tenFolds = du.generate_experiment_data_Categorical(data_set)
            else:
                headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)

            # dimensionality of data set
            ds = len(headers) - 1
            #Print the data to the screen for the user to see 
            print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
            #Create and store a copy of the first dataframe of data 
            test = copy.deepcopy(tenFolds[0])
            #Append all data folds to the training data set
            training = np.concatenate(tenFolds[1:])
            if k >= len(training)/2:
                k = 0 
            else: 
                k +=1
            #Print the length of the first array for debugging
            #print(len(test[0]))
            #Print the length of the training data set for testing 
            #print(len(training))
            #Create a KNN data object and insert the following data 
            knn = kNN.kNN(
                #Feed in the square root of the length 
                int(math.sqrt(len(full_set))), 
                # supply mixed, real, categorical nature of features
                feature_data_types[data_set],
                #Feed in the categorical attribute indicies stored in a global array 
                categorical_attribute_indices[data_set],
                #Store the data set key for the dataset name 
                regression_data_set[data_set],
                # weight for real distance
                alpha=1,
                # weight for categorical distance
                beta=1,
                # kernel window size
                h=.5,
                #Set the dimensionality of the data set in KNN
                d=ds
            )
            #Store and run the classification associated with the KNN algorithm 
            classifications = knn.classify(training, test)
            #Create a Results function to feed in the KNN Classification data and produce Loss Function Values 
            ResultObject = Results.Results() 
            #Create a list and gather some meta data for a given experiment, so that we can pipe all of the data to a file for evaluation
            MetaData = list() 
            MetaData.append(data_set)
            MetaData.append("TRIAL: ")
            MetaData.append("KNN")
            MetaData.append("K Value: ")
            MetaData.append(k)
            #Create a list to store the Results that are generated above FOR TESTING 
            ResultSet = ResultObject.StartLossFunction(regression_data_set.get(data_set),classifications, MetaData)
            #Now test the dataset on Edited KNN 
            #Print the Results to a file 
            Eknn = EditedKNN.EditedKNN( 
                #Error
                ResultSet[1], 
                #Feed in the square root of the length 
                int(math.sqrt(len(full_set))), 
                # supply mixed, real, categorical nature of features
                feature_data_types[data_set],
                #Feed in the categorical attribute indicies stored in a global array 
                categorical_attribute_indices[data_set],
                #Store the data set key for the dataset name 
                regression_data_set[data_set],
                # weight for real distance
                alpha=1,
                # weight for categorical distance
                beta=1,
                # kernel window size
                h=.5,
                #Set the dimensionality of the data set in KNN
                d=ds)
            classifications = Eknn.classify(training, test)
            MetaData = list() 
            MetaData.append(data_set)
            MetaData.append("TRIAL: ")
            MetaData.append("EDITED KNN")
            MetaData.append("K Value: ")
            MetaData.append(k)
            ResultSet = list() 
            ResultSet = ResultObject.StartLossFunction(regression_data_set.get(data_set),classifications, MetaData)
            print("EDITED FINISHED") 
            #Now test the dataset on Condensed KNN 
            #Print the Results to a file 
            MetaData = list() 
            MetaData.append(data_set)
            MetaData.append("TRIAL: ")
            MetaData.append("CONDENSED KNN")
            MetaData.append("K Value: ")
            MetaData.append(k)
            Cknn = CondensedKNN.CondensedKNN( 
                ResultSet[1],
                #Feed in the square root of the length 
                int(math.sqrt(len(full_set))), 
                # supply mixed, real, categorical nature of features
                feature_data_types[data_set],
                #Feed in the categorical attribute indicies stored in a global array 
                categorical_attribute_indices[data_set],
                #Store the data set key for the dataset name 
                regression_data_set[data_set],
                # weight for real distance
                alpha=1,
                # weight for categorical distance
                beta=1,
                # kernel window size
                h=.5,
                #Set the dimensionality of the data set in KNN
                d=ds
            )
            classifications = Cknn.classify(training, test)
            ResultSet = list() 
            ResultSet = ResultObject.StartLossFunction(regression_data_set.get(data_set),classifications, MetaData)
            MetaData = list() 
            MetaData.append(data_set)
            MetaData.append("TRIAL:")
            MetaData.append("KMEANS CLUSTERING")
            MetaData.append("K Value: ")
            MetaData.append(k)
            kmean = kMeansClustering.kMeansClustering(k,kValue=k, dataSet=training, data_type=feature_data_types[data_set], categorical_features=categorical_attribute_indices[data_set], regression_data_set=regression_data_set[data_set], alpha=1, beta=1, h=.5, d=ds,name=data_set,Testdata= test)

            classifications = kmean.classify()
            ResultSet = list() 
            ResultSet = ResultObject.StartLossFunction(regression_data_set.get(data_set),classifications, MetaData)
            MetaData = list() 
            MetaData.append(data_set)
            MetaData.append("TRIAL:")
            MetaData.append("KMEDOIDS CLUSTERING")
            MetaData.append("K Value: ")
            MetaData.append(k)
            medoids = kMedoidsClustering.kMedoidsClustering(kNeighbors=k,kValue=k, dataSet=training, data_type=feature_data_types[data_set], categorical_features=categorical_attribute_indices[data_set], regression_data_set=regression_data_set[data_set], alpha=1, beta=1, h=.5, d=ds,Testdata= test)
            classifications = medoids.classify()
            ResultSet = list() 
            ResultSet = ResultObject.StartLossFunction(regression_data_set.get(data_set),classifications, MetaData)


        

    #Print some meta data to the screen letting the user know the program is ending 
    print("Program End")
#On invocation run the main method
main()