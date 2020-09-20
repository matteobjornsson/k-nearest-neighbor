#Written by Nick Stone edited by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
# This program will run the data processor which will create the data processor to change the missing value of the data sets and return a pandas data frame     #
# This program is meant to run every KNN algorithm on a single dataset and NOT every dataset that is given in the project.                                      #
##
##
#################################################################### MODULE COMMENTS ############################################################################

import pandas as pd
import numpy as np
import sys
import random 
import copy 
import math 
import TrainingAlgorithm 
import Dataprocessor as Dataprocessor 



class DataUtiliti: 
    def __init__(self):
        print("initializing the Data") 


    #Remove 10 % of the data to be used as tuning data 
    def TuningData(self,):
        pass 

    #Break down the reminaing 90% of the data to be returned into 10 unique Numpy arrays for cross validation
    def CrossValiedation(self): 
        pass

    #Run the dataprocessor on a given data set that will return a pandas dataframe, convert the dataframe to a numpy array and send this numpy array to be converted into a list of data 
    #For the methods above to be called 
    def Dataprocessing(self,): 
        pass





if __name__ == '__main__':
    print("Testing the interface between pandas and numpy arrays")


    print("End of the tesitng interface")

