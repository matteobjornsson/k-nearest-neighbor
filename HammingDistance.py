#Written by Matteo Bjornsson Edited by Nick Stone 
#################################################################### MODULE COMMENTS ############################################################################
#The purpose of this class is to calculate the hamming distance between two vectors of features, this will be used in KNN to find distance between neighbots    #
#################################################################### MODULE COMMENTS ############################################################################

class HammingDistance:

    #Parameters: a list of features, a list of features 
    #Returns: Return an integer value for the distance between 2 data points 
    #Function: Take in 2 feature vectors and calculate the hamming distance between the 2 feature vectors  
    def Distance(self, x1, x2):
        #Set the distance value to 
        distance = 0
        #Loop through all of the data points in the feature vector 
        for i in range(len(x1)):
            #If the feature vector value is equal to the same feature value in the second vector 
            if x1[i] == x2[i]:
                #Set the value to be 0 
                value = 0
            #Otherwise 
            else:
                #Set the value to 1 
                value = 1
            #Increment the distance to be the value calculated above 
            distance += value
        #Return the distance 
        return distance

####################################### UNIT TESTING #################################################
if __name__ == '__main__':
    hd = HammingDistance()
    d = hd.Distance([0,1,0],[1,1,1])
    print(d)