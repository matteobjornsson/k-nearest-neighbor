#Written by Nick Stone edited by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
# This program will run the data processor which will create the data processor to change the missing value of the data sets and return a pandas data frame     #
# This program is meant to run every KNN algorithm on a single dataset and NOT every dataset that is given in the project.                                      #
# The data structure that will be passforward is a numpy array                                                                                                  #
##
#################################################################### MODULE COMMENTS ############################################################################

import pandas as pd
import numpy as np
import sys
import random 
import copy 
import math 
import DataProcessor 



class DataUtility: 
    def __init__(self):
        print("initializing the Data")     

    def ConvertDatastructure(self,df: pd.DataFrame): 
        #length = 3
        #Create a dataprocessor object and convert the data in the csv and change all missing attribtues 
        Dp = DataProcessor.DataProcessor()
        #Start the process to change the integrity of the dataframe from within the data processor
        data = Dp.ReplaceMissingValue(df) 
        #Convert the given Dataframe to a numpy array 
        Numpy = data.to_numpy() 
        #print the type of the class above, shoudl be a numpy array for testing 
        print(type(Numpy))
        #Return the numpy array 
        return Numpy

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
    Vote_Data = "C:/Users/nston/Desktop/MachineLearning/Project 2/Vote/Votes.data"
    df = pd.read_csv(Vote_Data,index_col = False)
    Df1 = DataUtility()
    Numpys = Df1.ConvertDatastructure(df)
    for i in Numpys: 
        print(i)
    print(Df1)



    print("End of the tesitng interface")

