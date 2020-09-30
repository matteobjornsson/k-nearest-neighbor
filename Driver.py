#Written by Nick Stone edited by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################

import copy
import DataUtility
import kNN
import math
import numpy as np

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
    print("Program Start")
   # Doj = DataUtility.DataUtility()
    #Read in all of the data sets
    """
    vote = 
    abalone = 
    glass = 
    machine = 
    fire = 
    image = 
    """



    for key in regression_data_set.keys():
        data_set = key
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        test = copy.deepcopy(tenFolds[0])
        training = np.concatenate(tenFolds[1:])
        print(len(test[0]))
        print(len(training))

        knn = kNN.kNN(
            int(math.sqrt(len(full_set))), 
            full_set,
            categorical_attribute_indices[data_set],
            regression_data_set[data_set]
        )
        classifications = knn.classify(training, test)





    print("Program End")

main()