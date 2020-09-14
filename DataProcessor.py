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

    
    
    
    
    #Parameters: Pandas DataFrame 
    #Returns: Clean ready to process Dataframe 
    #Function: This is the main function that should be called for each object that takes in the dataframe, processes it and returns the clean dataframe 
    def StartProcess(self, df:pd.DataFrame) -> pd.DataFrame:
        #Get a deep copy of the dataframe 
        df1 = copy.deepcopy(df)
        #SEt the count to 0 
        count = 0 
        #For each of the columns in the dataframe 
        for i in range(len(df.columns)): 
            #If the count is at the last column in the dataframe end because this is the classifier 
            if count == len(df.columns)-1: 
                #Break 
                break
            #bin Integers
           
            #If the type of the dataframe is a float then we need to discretize 
            if type(df1.iloc[0][i]) == np.float64: 
                #Find which column needs to be discretized
                df1 = self.discretize(df1,i)
                #Increment the count 
                count+=1
                #Go to the next one
                continue 
            #If the data frame has missing attributes 
            if self.has_missing_attrs(df1): 
                #Remove the missing attributes 
                df1 = self.fix_missing_attrs(df1)
            #Increment the count 
            count+=1
        #Return the cleaned dataframe 
        return df1
    
    #Parameters: Pandas DataFrame 
    #Returns: A dataframe with all missing values filled in with a Y or N 
    #Function: Take in a dataframe and randomly assigned a Y or a N to a missing value 
    def RandomRollVotes(self, df: pd.DataFrame) -> pd.DataFrame: 
        #Loop through each of the rows in the dataframe 
         for i in range(len(df)):
            #loop through all of the columns except the classification column
            for j in range(len(df.columns)-1): 
                #If the given value in the dataframe is missing a value 
                if self.IsMissingAttribute(df.iloc[i][j]): 
                    #Randomly assign a value from 1 - 100 
                    roll = random.randint(0,99) + 1
                    #If the roll is greater than 50 
                    if roll >50: 
                        #Assign the value to a Y 
                        roll = 'y'
                    #Otherwise 
                    else: 
                        #Assign the value to a N 
                        roll = 'n'
                    #Set the position in the dataframe equal to the value in the roll  
                    df.iloc[i][j] = roll
                #Go to the next  
                continue  
         #Return the dataframe 
         return df 

    #Parameters: Pandas DataFrame 
    #Returns: Bool if the dataframe has a missing attribute in it 
    #Function: Takes in a data frame and returns true if the data frame has  a ? value somewhere in the frame
    def has_missing_attrs(self, df: pd.DataFrame) -> bool:
        #For each row in the dataframe 
        for row in range(self.CountTotalRows(df)): 
            #For each column in the dataframe 
            for col in range(self.NumberOfColumns(df)): 
                #If the dataframe has a missing value in any of the cells
                if self.IsMissingAttribute(df.iloc[row][col]): 
                    #Return true 
                    return True
                #Go to the next value 
                continue  
        #We searched the entire list and never returned true so return false 
        return False
    
    #Parameters: Pandas DataFrame 
    #Returns: Cleaned Dataframe
    #Function: Take in a dataframe and an index and return a new dataframe with the row corresponding to the index removed 
    def KillRow(self, df: pd.DataFrame,index) -> pd.DataFrame: 
        return df.drop(df.Index[index])
          
    #Parameters: Attribute Value 
    #Returns: Bool -> True if the value is a missing value 
    #Function: Take in a given value from a data frame and return true if the value is a missing value false otherwise 
    def IsMissingAttribute(self, attribute) -> bool: 
        #Return true if the value is ? or NaN else return false 
        return attribute == "?" or attribute == np.nan

    #Parameters: Pandas DataFrame 
    #Returns: Clean Dataframe with not missing values 
    #Function: This function takes in a dataframe and returns a dataframe with all rows contianing missing values removed 
    def KillRows(self,df: pd.DataFrame) -> pd.DataFrame:
        # For each of the rows missing a value in the dataframe 
        for i in self.MissingRowIndexList: 
            #Set the dataframe equal to the dataframe with the row missing a value removed 
            df = df.drop(df.index[i])
        #Clear out all of the data in the set as to not try and drop these values again 
        self.MissingRowIndexList = set() 
        #Return the dataframe 
        return df

    #Parameters: Pandas DataFrame 
    #Returns: Dataframe with all columns with missing values dropped 
    #Function: This function takes in a dataframe and drops all columns with missing attributes 
    def KillColumns(self,df: pd.DataFrame) -> pd.DataFrame: 
        #For each of the columns with missing attributes which is appending into a object list 
        for i in self.MissingColumnNameList: 
            #Set the dataframe equal to the dataframe with these values dropped 
            df = df.drop(i,axis=1)
        #Set the object list back to an empty set as to not try and drop these columns again 
        self.MissingColumnNameList = set() 
        #Returnn the dataframe 
        return df
 


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


