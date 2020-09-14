import pandas as pd
import numpy as np
import sys
import random 
import copy 
import math 
import TrainingAlgorithm 


class DataProcessor:
    

    def __init__(self):
        pass

    def RandomVote(): 
    




    #Takes in a dataframe and populates attributes based on the existing distribution of attribute values 
    #Parameters: Pandas DataFrame 
    #Returns: a Data frame with no missing attributes 
    #Function: Take in a given dataframe and replace all missing attributes with a randomly assigned value 
    def fix_missing_attrs(self, df: pd.DataFrame) -> pd.DataFrame:
        #Get the total percentage of rows missing values in the dataframe
        PercentRowsMissing = self.PercentRowsMissingValue(df)
        #Get the total number of columns missing values in the dataframe 
        PercentColumnsMissingData = self.PercentColumnsMissingData(df)
        #If the total number of rows missing data is less than the value specified in the init 
        if(PercentRowsMissing < self.PercentBeforeDrop): 
            #Return the dataframe that removes all rows with missing values 
            return self.KillRows(df)
        #If the percentage of columns missing values is less than the value specified in the init 
        elif(PercentColumnsMissingData < self.PercentBeforeDrop):
            #Return the dataframe with all columns including missing values dropped 
            return self.KillColumns(df)  
        #otherwise 
        else: 
            #If the Data frame has no missing attributes than the Data frame is ready to be processed 
            if self.has_missing_attrs(df) == False:
                #Return the dataframe 
                return df  
            #Find the Type of the first entry of data
            types = type(df.iloc[1][1])
            #If it is a string then we know it is a yes or no value 
            if types == str: 
                #Set the dataframe equal to the dataframe with all missing values randmoly generated
                df = self.RandomRollVotes(df) 
            #Else this is an integer value 
            else:
                #Set the dataframe equal to the dataframe with all missing values randmoly generated
                df =self.RandomRollInts(df) 
        #Return the dataframe 
        return df

    #Parameters: Pandas DataFrame 
    #Returns: Integer; Total number of rows in a dataframe
    #Function: Take in a dataframe and return the number of rows in the dataframe 
    def CountTotalRows(self,df: pd.DataFrame) -> int: 
        #Return the total number of rows in the data frame 
        return len(df)


    #Parameters: Pandas DataFrame 
    #Returns: Integer; Number of rows missing values 
    #Function: Take in a dataframe and return the number of rows in the dataframe with missing attribute values 
    def CountRowsMissingValues(self,df: pd.DataFrame ) -> int:
        #Set a Counter Variable for the number of columns in the data frame 
        Count = 0 
        #Set a counter to track the number of rows that have a missing value 
        MissingValues = 0 
        #Get the total number of rows in the data set 
        TotalNumRows = self.CountTotalRows(df)
        #For each of the columns in the data frame 
        for i in df: 
            #increment by 1 
            Count+=1 
        #For each of the records in the data frame 
        for i in range(TotalNumRows): 
            #For each column in each record 
            for j in range(Count): 
                #If the specific value in the record is a ? or a missing value 
                if self.IsMissingAttribute(df.iloc[i][j]):
                    #Increment Missing Values 
                    MissingValues+=1
                    self.MissingRowIndexList.add(i)
                    #Go to the next one 
                    continue 
                #Go to the next ones
                continue  
        #Return the number of rows missing values in the data set 
        return MissingValues 


    #Parameters: Pandas DataFrame 
    #Returns: float; Percent rows missing data
    #Function: Take in a dataframe and count the number of rows with missing attributes, return the percentage value 
    def PercentRowsMissingValue(self,df: pd.DataFrame) -> float: 
        #Get the total number of rows in the dataset
        TotalNumRows = self.CountTotalRows(df)
        #Get the total number of rows with missing values 
        TotalMissingRows = self.CountRowsMissingValues(df)
        #Return the % of rows missing values  
        return (TotalMissingRows/TotalNumRows) * 100 
    #Parameters: Pandas DataFrame 
    #Returns: Integer; Number of columns with missing attributes
    #Function: Return a count of the number of columns with atleast one missing attribute value in the data frame 
    def ColumnMissingData(self,df: pd.DataFrame) -> int: 
        #Create a counter variable to track the total number of columns missing data 
        Count = 0 
        #Store the total number of columns in the data set 
        TotalNumberColumns = self.NumberOfColumns(df)
        #Store the total number of rows in the data set 
        TotalNumberRows = self.CountTotalRows(df) 
        #For each of the columns in the dataset 
        for j in range(TotalNumberColumns): 
            #For each of the records in the data set 
            for i in range(TotalNumberRows): 
                #If the value at the specific location is ? or a missing value 
                if self.IsMissingAttribute(df.iloc[i][j]): 
                    #Increment the counter
                    Count+=1 
                    Names = df.columns
                    self.MissingColumnNameList.add(Names[j])
                    #Break out of the loop 
                    break 
                #Go to the next record 
                continue 
        #Return the count variable 
        return Count


    #Parameters: Pandas DataFrame 
    #Returns: Integer; Number of columns
    #Function: Take in a given dataframe and count the number of columns in the dataframe 
    def NumberOfColumns(self,df: pd.DataFrame) -> int: 
        #Create a counter variable 
        Count = 0 
        #For each of the columns in the dataframe 
        for i in df: 
            #Increment Count 
            Count+=1 
        #Return the total number of Columns 
        return Count 

    #Parameters: Pandas DataFrame 
    #Returns: Float; The percentage of columns with missing data 
    #Function: Take in a given dataframe and find the total number of columns divided by the number of columns with missing attribute values 
    def PercentColumnsMissingData(self,df: pd.DataFrame) -> float: 
        #Total Number of Columns in the dataset 
        TotalNumberColumns = self.NumberOfColumns(df)
        #Total number of columns missing values in the dataset
        TotalMissingColumns = self.ColumnMissingData(df)
        #Return the percent number of columns missing data
        return (TotalMissingColumns/TotalNumberColumns) * 100 

    
    #Parameters: Pandas DataFrame
    #Returns: None
    #Function: This is a test function that will print every cell to the screen that is in the dataframe
    def PrintAllData(self,df:pd.DataFrame) -> None: 
        #For each of the rows in the dataframe 
        for i in range(len(df)):
            #For each of the columns in the dataframe 
            for j in range(len(df.columns)): 
                #Print the value in that position of the dataframe 
                print(df.iloc[i][j])












if __name__ == '__main__':
    print("Data Processor Testing")


