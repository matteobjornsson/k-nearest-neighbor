#Written by Matteo Bjornsson edited by Nick Stone 
#################################################################### MODULE COMMENTS ############################################################################
#The purpose of this class is to create an object that can calculate the distance between two neighbors with real valued data sets.                             # 
#################################################################### MODULE COMMENTS ############################################################################
class RealDistance:
    #On the initializing of this function 
    def __init__(self):
        #Set the p value to be 2 
        self.pValue = 2
    
    #Parameters: a list of features, a list of features 
    #Returns: Return an integer value for the distance between 2 data points 
    #Function: Take in 2 feature vectors and calculate the euclidian distance between the 2 feature vectors  
    def Distance(self, x1, x2):
        #Set the p to be the p value of the object 
        p = self.pValue
        #Set a distance variable to be 0 
        distance = 0
        #For each of the features in the feature vector x1 
        for i in range(len(x1)):
            #Add the absolute value from x1 - x2 to the power p 
            distance += (abs(x1[i] - x2[i])**p)
        #Set distance to be the distance to the 1/2 power 
        distance = distance**(1/p)
        #Return the distance value 
        return distance


####################################### UNIT TESTING #################################################
if __name__ == '__main__':
    rd = RealDistance()
    d = rd.Distance([1,2,3],[2,9,4])
    assert round(d, 9) == 7.141428429
    print(d)