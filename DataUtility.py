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
        #print(df)
        #Start the process to change the integrity of the dataframe from within the data processor
        data = Dp.ReplaceMissingValue(df) 

       # print(data)
        #Convert the given Dataframe to a numpy array 
        Numpy = data.to_numpy() 
        #print(Numpy)
              # print(Numpy)
        #Return the numpy array 
        return Numpy

    #Remove 10 % of the data to be used as tuning data 
    def TuningData(self,NParray: np.array):
        TuningArray = np.array([])
        NParrays = np.array([]) 
        Data = list() 
        #Grab 10% of the data to be removed for tuning 
        TuningData = len(NParray) * .1
        print(TuningData)
        print(int(TuningData))

        

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
    df = pd.read_csv(Vote_Data)
    Df1 = DataUtility()
    Numpys = Df1.ConvertDatastructure(df)
    print(len(df))
    print(len(Numpys))
    
    #for i in Numpys: 
    #    print(i )
    test = Df1.TuningData(Numpys)
 


    print("End of the tesitng interface")
