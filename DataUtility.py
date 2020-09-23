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
    def __init__(self, categorical_attribute_indices, regression_data_set):
        self.categorical_attribute_indices = categorical_attribute_indices
        self.regression_data_set = regression_data_set
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

    # this function takes in the name of a preprocessed data set and normalizes
    # all continuous attributes within that dataset to the range 0-1.
    def min_max_normalize_real_features(self, data_set: str) -> None:
        # read in processed dataset
        df = pd.read_csv(f"./ProcessedData/{data_set}.csv")
        # create new data frame to store normalized data
        normalized_df = pd.DataFrame()
        # keep track of which column index we are looking at
        index = -1
        headers = df.columns.values
        # iterate over all columns
        for col in headers:
            index += 1
            # check if the index is categorical or ground truth. in this case do not normalize
            if index in self.categorical_attribute_indices[data_set] or col == headers[-1]:
                normalized_df[col] = df[col]
                continue
            # generate a normalized column and add it to the normalized dataframe
            min = df[col].min()
            max = df[col].max()
            normalized_df[col] = (df[col] - min)/(max - min)
        # save the new normalized dataset to file
        normalized_df.to_csv(f"./Data/{data_set}.csv", index=False)

    # this function takes in experiment ready data and returns all forms of data required for the experiment 
    def generate_experiment_data(self, data_set)-> list, np.ndarray, np.ndarray, list:
        # read in data set
        df = pd.read_csv(f"./NormalizedData/{data_set}.csv")
        # save the column labels
        headers = df.columns.values
        # extract data from dataset to tune parameters
        tuning_data, remainder = self.TuningData(df)
        # convert the tuning data set to numpy array
        tuning_data = tuning_data.to_numpy()
        # split the remaining data into 10 chunks for 10fold cros validation
        tenFolds = self.BinTestData(remainder)
        # save the full set as numpy array
        full_set = remainder.to_numpy()
        # return the headers, full set, tuning, and 10fold data
        return headers, full_set, tuning_data, tenFolds 





if __name__ == '__main__':

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
    
    du = DataUtility(categorical_attribute_indices, regression_data_set)
    for key in categorical_attribute_indices.keys():
        du.min_max_normalize_real_features(key)
    # headers, full_set, tuning_data, tenFolds = du.generate_experiment_data("vote")
    # assert len(headers) == len(tuning_data[0])
    # count = 0
    # for fold in tenFolds:
    #     count+= len(fold)
    # assert count == len(full_set)
    # print("End of the testing interface")

