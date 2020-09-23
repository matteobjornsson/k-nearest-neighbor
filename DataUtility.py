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

    def ReplaceMissing(self,df: pd.DataFrame):
        #length = 3
        #Create a dataprocessor object and convert the data in the csv and change all missing attribtues 
        Dp = DataProcessor.DataProcessor()
        #Start the process to change the integrity of the dataframe from within the data processor
        data = Dp.ReplaceMissingValue(df) 
        return data 

    def ConvertDatastructure(self,df: pd.DataFrame): 
        #Convert the given Dataframe to a numpy array 
        Numpy = df.to_numpy() 
        #Return the numpy array 
        return Numpy

    #Remove 10 % of the data to be used as tuning data and seperate them into a unique dataframe 
    def TuningData(self,df: pd.DataFrame):
        remaining_data = copy.deepcopy(df)
        Records = int(len(df) * .1)
        tuning_data = copy.deepcopy(df)
        tuning_data = tuning_data[0:0]
        for i in range(Records):
            Random =  random.randint(0,len(remaining_data)-1)
            rec = remaining_data.iloc[Random]
           
            tuning_data = tuning_data.append(remaining_data.iloc[Random],ignore_index = True)
            
            remaining_data = remaining_data.drop(remaining_data.index[Random])
            remaining_data.reset_index()
        return tuning_data, remaining_data
        

    #Break down the reminaing 90% of the data to be returned into 10 unique Numpy arrays for cross validation
    def CrossValiedation(self): 
        pass

    #Run the dataprocessor on a given data set that will return a pandas dataframe, convert the dataframe to a numpy array and send this numpy array to be converted into a list of data 
    #For the methods above to be called 
    def Dataprocessing(self,): 
        pass


    #Parameters: DataFrame
    #Returns: List of dataframes 
    #Function: Take in a dataframe and break dataframe into 10 similar sized sets and append each of these to a list to be returned 
    def BinTestData(self, df: pd.DataFrame) -> list(): 
        #Set the bin size to 10 
        Binsize = 10
        #Create a List of column names that are in the dataframe 
        columnHeaders = list(df.columns.values)
        #Create an empty list 
        bins = []
        #Loop through the size of the bins 
        for i in range(Binsize):
            #Append the dataframe columns to the list created above 
            bins.append(pd.DataFrame(columns=columnHeaders))
        #Set a list of all rows in the in the dataframe 
        dataIndices = list(range(len(df)))
        #Shuffle the data 
        random.shuffle(dataIndices)
        #Shuffle the count to 0 
        count = 0
        #For each of the indexs in the dataIndices 
        for index in dataIndices:
            #Set the bin number to count mod the bin size 
            binNumber = count % Binsize
            bins[binNumber] = bins[binNumber].append(df.iloc[index], ignore_index=True)
            #Increment count 
            count += 1
            #Go to the next 
            continue
        #Return the list of Bins 
        for i in range(Binsize):
            bins[i] = bins[i].to_numpy()
        return bins

    def generate_experiment_data(self, filename):
        df = pd.read_csv(filename)
        headers = df.columns.values
        print("headers: ", headers)
        tuning_data, remainder = self.TuningData(df)
        tuning_data = tuning_data.to_numpy()
        print("tuning data type: ", type(tuning_data))
        print(tuning_data)
        tenFolds = self.BinTestData(remainder)
        print("tenfold type: ", type(tenFolds[1]))
        for i in range(len(tenFolds)):
            print (f"fold {i}: ")
            print(tenFolds[i])
        full_set = remainder.to_numpy()
        print(full_set)

        return headers, full_set, tuning_data, tenFolds 





if __name__ == '__main__':
    print("Testing the interface between pandas and numpy arrays")
    # Vote_Data = "C:/Users/nston/Desktop/MachineLearning/Project 2/Vote/Votes.data"
    # df = pd.read_csv(Vote_Data)
    # Df1 = DataUtility()
    # dfs = Df1.ReplaceMissing(df)
    # test = list() 
    # Tuning, df = Df1.TuningData(dfs)
    # bins = [] 
    # bins = Df1.BinTestData(df)
    # Tuning = Df1.ConvertDatastructure(Tuning)
    # print(type(Tuning))
    # for i in range(len(bins)):
    #     bins[i] = Df1.ConvertDatastructure(bins[i])
    # for i in bins: 
    #     print(type(i))
    
    du = DataUtility()
    headers, full_set, tuning_data, tenFolds = du.generate_experiment_data("./Data/vote.csv")
    assert len(headers) == len(tuning_data[0])
    count = 0
    for fold in tenFolds:
        count+= len(fold)
    assert count == len(full_set)
    print("End of the testing interface")

