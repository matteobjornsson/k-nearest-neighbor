#################################################################### MODULE COMMENTS ####################################################################
# The following Python object is responsible for calculating two loss functions to identify a series of statistical data points for a programmer to view #
# In order to see how 'Well' the Naive bayes program is functioning. The two loss functions that Nick Stone and Matteo Bjornsson implemented for this pr-#
# -oject were the 0/1 loss function which we will use to calculate the algorithms precision and the F1 score for a multidimensional data set.            #
# All of the functions have been documented such that a programmer can understand the mathematics and statistics involved for undersanding each of the l-#
# -oss Functions. The main datastructures used were a dataframe and a dictionary to keep track of a given confusion matrix                               # 
#################################################################### MODULE COMMENTS ####################################################################
import pandas as pd
import numpy as np




class Results: 
    
    def __init__(self):
        print("Starting the Results File")


    """
    loss functions 

    multiclass confusion matrix 
    https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier

    multiclass precision and recall
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

    multiclass f1 score
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

    """

    def MSE(self,data_set: list()) -> float: 
        SquaredError = list()  
        for i in data_set: 
            #First Value is the Ground truth 
            True_Value = data_set[0] 
            #Grab the last value since it is the predicted value 
            Pred_Value = data_set[1]
            #Calculate the error by the difference of the two values above 
            Error = True_Value - Pred_Value
            #Square the error 
            Error = Error * Error
            #Store into the Squared Error list created above 
            SquaredError.append(Error)
        #Set a counter variable 
        Mean = 0 
        #For each of the squared error vales we entered in the list above 
        for i in SquaredError: 
            #Add the value to the overall mean 
            Mean +=i 
        #Divide out by the total number of entries 
        Mean = Mean / len(SquaredError)
        #Return the mean 
        return Mean 
         

    #Parameters: Dataframe 
    #Returns: DataFrame, Dictionary 
    #Function: Take in a given dataframe and get a series of statistics about the given dataframe 
    def statsSummary(self, df: list()) -> (pd.DataFrame, dict, dict):
        #Create a dataframe of the confusion matrix 
        cMatrix = self.ConfusionMatrix(df)
        #Create a Dataframe to get stats about the classes 
        classStats = self.perClassStats(cMatrix)
        # tpList = list(classStats["TP"])
        # fpList = list(classStats["FP"])
        # fnList = list(classStats["FN"])
        # microStats = self.microAverageStats(tpList, fpList, fnList)
        macroF1Average = self.weightedMacroAverageStats(classStats, cMatrix)
        return macroF1Average
    #Parameters: DataFrame 
    #Returns: DataFrame
    #Function: Take in a dataframe and generate all of the class stats from the given dataframe and return it 
    def perClassStats(self, cMatrix): 
        #Set the dataframe of the class stats
        classStats = self.classStats(cMatrix)
        #return the dataframe of class stats
        return classStats
    
    
    #Parameters: Float, Float
    #Returns: Float 
    #Function: Take in the float of class stats and the cmatrix and generate the weighted macro averages 
    def weightedMacroAverageStats(self, perClassStats, cMatrix) -> dict: 
        #Create a list of the names of classes 
        classValues = list(perClassStats.index.values)
        #Create an empty dictionary To store macro stats
        macroStatsDict = {}
        #Get the total count of occurence of each class 
        classCounts = self.countClassOccurence(cMatrix)
        #Set a total class count variable to 0 
        totalClassCount = 0
        #For each of the class keys 
        for key in classCounts.keys():
            #Add the total class occurence to the variable initialized above 
            totalClassCount += classCounts[key]
        #Set a series of statistical variables to 0 
        # macroRecallAverage = 0
        # macroPrecisionAverage = 0
        macroF1Average = 0
        #Loop through all of the classes 
        for i in range(len(classValues)):
            # #Add the total number of occurence of the class multiplied by the stat percentage 
            # macroRecallAverage += (perClassStats["Recall"].iloc[i] * classCounts[classValues[i]])
            # #Add the total number of occurence of the class multiplied by the stat percentage 
            # macroPrecisionAverage += (perClassStats["Precision"].iloc[i] * classCounts[classValues[i]])
            #Add the total number of occurence of the class multiplied by the stat percentage 
            macroF1Average += (perClassStats["F1"].iloc[i] * classCounts[classValues[i]])
        #THen divide by the total number of examples 
        # macroStatsDict["macroRecall"] = macroRecallAverage / totalClassCount
        # macroStatsDict["macroPrecision"] = macroPrecisionAverage / totalClassCount
        macroF1Average = macroF1Average / totalClassCount
        #Return the dictionary 
        return macroF1Average
        
    #Parameters: Float, Float
    #Returns: Float 
    #Function: Count the number of times a class occurs in a given dataframe and return this in a dictionary 
    def countClassOccurence(self, cMatrix: pd.DataFrame) -> dict:
        #Get a list of all the class names 
        classValues = list(cMatrix.index.values)
        #Create an empty dictionary with class names as key values 
        classCounts = dict.fromkeys(classValues)
        # print("perclass stats from countclass occurence method: \n", cMatrix)
        #Loop through all class names stored in the list above 
        for i in range(len(classValues)):
            #Store the given class stats from row i 
            x = cMatrix.iloc[i]
            #Set the count to 0 
            count = 0
            #For each stat and value in the row 
            for stat, statValue in x.items():
                #Adding stat value to the variable we instantiated above 
                count += statValue
            #Set the count to the class value 
            classCounts[classValues[i]] = count
        #Return the dictionary 
        return classCounts
        
    #Parameters: Float, Float
    #Returns: Float 
    #Function: This set takes in the number of true positives followed by a series of lists of several other confusion matrix values to generate micro average stats 
    def microAverageStats(self, truePositives: list, falsePositives: list, falseNegatives: list) -> dict:
        #Create an empty dictionary 
        microStatsDict = {}
        #Store off the sum of the given values into the respective variable name 
        tpSum = sum(truePositives)
        fpSum = sum(falsePositives)
        fnSum = sum(falseNegatives)
        #Get the micro recall from the values above 
        microRecall = self.recall(tpSum, fnSum)
        #Get the micro precision from the values above 
        microPrecision = self.precision(tpSum, fpSum)
        #Save the micro recall value in the dictionary with the following key 
        microStatsDict["microRecall"] = microRecall
        #Save the micro precision value with the following key in the dictionary 
        microStatsDict["microPrecision"] = microPrecision
        #Calculate and save the following F1 score with the following key in the dictionary 
        microStatsDict["microF1Score"] = self.f1Score(precision=microPrecision, recall=microRecall)
        #Return the dicationary that is full of data 
        return microStatsDict

    #Parameters: Float, Float
    #Returns: Float 
    #Function: Take in the precision and recall that is calculated below and generate the F1 score 
    def f1Score(self, precision, recall) -> float:
            #If the precision + recall is 0 
            if (precision + recall) == 0:
                #Return 0 
                return 0
            #Otherwise 
            else:
                #Return the value asscoiated with the following computation 
                return 2 * precision * recall / (precision + recall)

    #Parameters: Inter, Integer
    #Returns: Float 
    #Function: Take in the true positive count and the false negative count to calculate and return the recall 
    def recall(self, truePositiveCount, falseNegativeCount) -> float:
        #If the TP + FN is 0 
        if truePositiveCount + falseNegativeCount == 0:
            #Return 0 
            return 0
        #Otherwise 
        else:
            #Return the following computation( TP/ *TPC + FN )
            return truePositiveCount/(truePositiveCount + falseNegativeCount)
    #Parameters: Integer, Integer
    #Returns: Float
    #Function:Function takes in the true positives and the false positive from the confusion matrix and calulates the precision  
    def precision(self, truePositiveCount, falsePositiveCount) -> float:
        #If the TP + FP is 0 
        if truePositiveCount + falsePositiveCount == 0:
            #Return 0 
            return 0
        #Otherwise 
        else:
            #Return the formula for precision 
            return truePositiveCount/(truePositiveCount + falsePositiveCount)
    
    #Parameters: Integer, DataFame
    #Returns: List 
    #Function: return a list of true positive counts for each class
    def truePositive(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        tp = []
        # print("cmatrix true positive method: \n", cMatrix)
        for i in classCount:
            # true positive for each class is where truth == guess
            tp.append(cMatrix.iloc[i][i])
        return tp

    #Parameters: Integer, DataFrame 
    #Returns: List 
    #Function: return a list of false positive counts for each class
    def falsePositive(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        # print("cmatrix in true positive method: \n")
        # print(cMatrix)
        fp = []
        for i in classCount:
            count = 0
            for j in classCount:
                if i == j:
                    continue
                else:
                    # false positive is the sum of every count in the class
                    # column except the true positive count
                    count += cMatrix.iloc[j][i]
            fp.append(count)
        return fp

    #Parameters: Integer, DataFrame 
    #Returns: List 
    #Function: return a list of false negative counts for each class
    def falseNegative(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        #Create an empty array 
        fn = []
        #For each of the values in the classcount 
        for i in classCount:
            #Set count to 0 
            count = 0
            for j in classCount:
                #If the class count is the same 
                if i == j:
                    #Do nothing 
                    continue
                #Otherwise 
                else:
                    # false negative is the sum of every count in the class
                    # row except the true positive count
                    count += cMatrix.iloc[i][j]
            fn.append(count)
        #Return the list 
        return fn

    #Parameters: Int, DataFrame, List, List,List 
    #Returns: List 
    #Function: return a list of true negative counts for each class
    def trueNegative(self, classCount: range, cMatrix: pd.DataFrame, tp: list, fp: list, fn: list) -> list:
        #Create an empty array 
        tn = []
        #For each of the values in classcount 
        for i in classCount:
            #Set count to 0 d
            count = 0
            # sum the value of every cell
            for j in classCount:
                for k in classCount:
                    #Add the value at the given dataframe position to count 
                    count += cMatrix.iloc[j][k]
            # true negative counts are the sum of every cell minus true 
            # positive, false positive and false negative.
            count = count - tp[i] - fp[i] - fn[i]
            tn.append(count)
        #Return the list 
        return tn

    #Parameters: DataFrame
    #Returns: DataFrame
    #Function: create a stats summary matrix for all classes
    def classStats(self, cMatrix: pd.DataFrame) -> pd.DataFrame:
        # grab the class names
        #Print some data to the screen 
        # print("cmatrix stats method: \n")
        # print(cMatrix)
        #Create a list to the column names from the dataframe 
        ClassList = list(cMatrix.columns.values)
        #Print some data to the screen 
        # print("classList class stats method: \n")
        # print(ClassList)
        #Sent a class count to the len of the class list 
        classCount = range(len(ClassList))
        # init an empty matrix with class indexes labeled
        statsMatrix = pd.DataFrame(index=ClassList)

        # calculate stats
        tp = self.truePositive(classCount, cMatrix)
        fp = self.falsePositive(classCount, cMatrix)
        fn = self.falseNegative(classCount, cMatrix)
        tn = self.trueNegative(classCount, cMatrix, tp, fp, fn)

        # insert stats into matrix
        statsMatrix["TP"] = tp
        statsMatrix["FP"] = fp
        statsMatrix["FN"] = fn
        statsMatrix["TN"] = tn
        #Create some empty lists 
        precisionList = []
        recallList = []
        fScoreList = []
        #For each of the values in the class count 
        for i in classCount:
            #Get the value of the given value 
            singleClassStats = statsMatrix.iloc[i]

            tp = singleClassStats["TP"]
            fp = singleClassStats["FP"]
            fn = singleClassStats["FN"]
            #Set the following values 
            prec = self.precision(tp, fp)
            rec = self.recall(tp, fn)
            #Get the value of the f1 score 
            f1 = self.f1Score(prec, rec)
            #Print some data to the screen 
            # print("i: ", i, " prec: ", prec, " recall: ", rec, " f1: ", f1)
            #Append data to the following lists 
            precisionList.append(prec)
            recallList.append(rec)
            fScoreList.append(f1)

        statsMatrix["Precision"] = precisionList
        statsMatrix["Recall"] = recallList
        statsMatrix["F1"] = fScoreList
        #Return the Dataframe 
        return statsMatrix

    #Parameters: DataFrame 
    #Returns: DataFrame
    #Function: generate a matrix that checks classified test data against ground truth
    def ConfusionMatrix(self, df: list()) -> pd.DataFrame:
        # identify column index of ground truth and classification
        GroundTruthIndex = 0
        ClassifierGuessIndex = 1 

        # generate a list of all unique classes
        UniqueClasses = list() 
        #Loop through all of the rows in the dataframe 
        for i in df: 
            #If the Ground truth classification not in the unique classes list 
            if str(i[GroundTruthIndex]) not in UniqueClasses:
                #Append the value to the list 
                UniqueClasses.append(str(i[GroundTruthIndex]))
            #If the classifcation is not in the unique classes list 
            if str(i[ClassifierGuessIndex]) not in UniqueClasses:
                #Add the value to the list 
                UniqueClasses.append(str(i[ClassifierGuessIndex]))
            #Go to the next one 
            continue 
        #Set the class count to the length of unique classes 
        ClassCount = len(UniqueClasses)

        # initialize empty confusion matrix
        zeroArray = np.zeros(shape=(ClassCount, ClassCount))
        #Set a variable to a dataframe that has the columns from the unique classes 
        matrix = pd.DataFrame(zeroArray, columns=UniqueClasses, index=UniqueClasses)
        #Print some data to the screen 
        # print("empty Cmatrix, cmatrix method: \n", matrix)
        #For each of the rows in the dataframe 
        for i in df:
            # for each example, increment a counter where row = truth, col = guess
            truth = str(i[GroundTruthIndex])
            guess = str(i[ClassifierGuessIndex])
            #Increment the count 
            matrix.at[truth, guess] += 1
            #Go to the next one 
            continue
        #Return the dataframe 
        return matrix

#[{},{},{}]



#Unit Testing the object created above 
#Code not run on creation of object just testing function calls and logic above 
if __name__ == '__main__':
    print("Program Start")
    ClassifiedDataFrame = list() 
    temp = list()
    temp.append(0.0)
    temp.append(2.162857142857143)
    ClassifiedDataFrame.append(temp)

    
    #[[0.0, 2.162857142857143], [0.0, 63.20476190476192], [3.71, 3.8890476190476186], [0.0, 16.93285714285714], [0.0, 50.36952380952381]]
    re = Results()
    #print(ClassifiedDataFrame)
    macroF1Average = re.statsSummary(ClassifiedDataFrame)
    print(macroF1Average)
    print("Program Finish")




    



