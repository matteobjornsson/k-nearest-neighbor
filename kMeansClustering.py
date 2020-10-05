#Written By Nick Stone and Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
#The Kmeans clustering builds off of the KNN algorithm by creating centroids and trying to group large numbers of data points together. By doing this the algo. #
#can cluster data points together and can classify data points by looking at an entire neighborhood instead of an arbitrarily defined K for the number of neighb#
#Of course the tuning data for this algorithm is the number of neighbors as well as a few other data points that should be tuned to each of the given data sets #
#################################################################### MODULE COMMENTS ############################################################################
from random import sample
import numpy as np
import DataUtility, kNN, Results
import copy, random


class kMeansClustering:
    #On initialization 
    def __init__(self,
        # number of neighbors in knn
        kNeighbors: int,
        # number of clusters
        kValue: int,
        # data to cluster
        dataSet: np.ndarray,
        # 'mixed', 'categorical', or 'real' data set
        data_type: str,
        # list of integers representing categorical feature column indices
        categorical_features: list,
        # True if the data set is a regression data set
        regression_data_set: bool,
        # weight for real value in distance metric
        alpha: int,
        # weight for categorical value in distance metric
        beta: int,
        # bin width for gaussian kernel smoother
        h: float,
        # dimensionality of data set (# features)
        d: int,
        name: str, 
        Testdata: np.ndarray):
        # create a Nearest Neighbor object to single nearest neighbor to input data point
        self.nn = kNN.kNN(1, data_type, [], regression_data_set, alpha, beta, h, d)
        self.knn = kNN.kNN(kNeighbors, data_type, [], regression_data_set, alpha, beta, h, d)
        self.categorical_features = []
        self.itermax = 5
        self.kValue = kValue
        #Convert the data such that no categorical values are in the data set  
        for j in range(len(Testdata)): 
            Testdata[j] = self.ConvertData(Testdata[j],name)
        #Set the test set value 
        self.Testdata = Testdata
        #Convert the data such that no categorical values are in the data set  
        for j in range(len(dataSet)):
            dataSet[j] = self.ConvertData(dataSet[j],name)
        #Set the data set value 
        self.dataSet = dataSet
        #Set the dimensionality 
        self.d = d
        #If the data set is machine 
        if name == "machine": 
            print(self.dataSet, self.dataSet.shape)
            self.dataSet = self.dataSet[:,2:]
            self.Testdata = self.Testdata[:,2:]
            print(self.dataSet, self.dataSet.shape)
            self.d = d-2
        # dimensionality of data set
        # save which features are real as well by deleting categorical indices from a new list
        real_features = list(range(d))
        #Remove all of the categorical values 
        for i in categorical_features:
            real_features.remove(i)
        #SEt the real features object variable 
        self.real_features = real_features

    #Parameters: take in a data set and the name of a given data set 
    #Returns:  Return the new data set with all categorical values conveted 
    #Function: Convery all of the categorical features to a integrer or real value 
    def ConvertData(self,data_set_row, Name):
        #For each of the indexes in the data_set_row 
        for i in range(len(data_set_row)): 
            #if the value is a N or an n from the vote data cast to a 1 
            if data_set_row[i] == 'N' or data_set_row[i] == 'n': 
                #Conver the value to 1 
                data_set_row[i] = 1
            #If the value that we are taking in from the vote data is a y 
            if data_set_row[i] == 'Y' or data_set_row[i] == 'y': 
                #Set the value to be a 0 
                data_set_row[i] = 0 
            #If the data from the forest fire is jan 
            if data_set_row[i] == 'jan': 
                #Set the value to 0 
                data_set_row[i] = 0/11
            #If the data from the forest fire is feb
            if data_set_row[i] == 'feb' : 
                #Set the value to be the value of the month divided by the total number of months starting from 0 
                data_set_row[i] = 1/11
            #If the data from the forest fire is mar
            if data_set_row[i] == 'mar': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 2/11
            #If the data from the forest fire is apr
            if data_set_row[i] == 'apr': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 3/11
            #If the data from the forest fire is may
            if data_set_row[i] == 'may': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 4/11
            #If the data from the forest fire is jun
            if data_set_row[i] == 'jun': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 5/11
            #If the data from the forest fire is jul
            if data_set_row[i] == 'jul': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 6 /11
            #If the data from the forest fire is aug
            if data_set_row[i] == 'aug': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 7 /11
            #If the data from the forest fire is sep
            if data_set_row[i] == 'sep': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 8 /11
            #If the data from the forest fire is oct
            if data_set_row[i] == 'oct':
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 9 /11
            #If the data from the forest fire is nov
            if data_set_row[i] == 'nov': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 10/11
            #If the data from the forest fire is dec
            if data_set_row[i] == 'dec': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 11/11
            #If the day of the week is Monday
            if data_set_row[i] == 'mon' : 
                #Set the value to be 0  
                data_set_row[i] = 0/6
            #If the day of the week is Tuesday
            if data_set_row[i] == 'tue': 
                #Set the value to be the 1st day divide by 6 days 
                data_set_row[i] = 1/6
            #If the day of the week is Wednesdayu
            if data_set_row[i] == 'wed': 
                #Set the value to be the 2nd day divide by 6 days 
                data_set_row[i] = 2/6
            #If the day of the week is Thursday
            if data_set_row[i] == 'thu': 
                #Set the value to be the 3rd day divide by 6 days 
                data_set_row[i] = 3/6
            #If the day of the week is Friday
            if data_set_row[i] == 'fri': 
                #Set the value to be the 4th day divide by 6 days 
                data_set_row[i] = 4/6
            #If the day of the week is Saturday
            if data_set_row[i] == 'sat':
                #Set the value to be the 5th day divide by 6 days  
                data_set_row[i] = 5 /6
            #If the value is sunday 
            if data_set_row[i] == 'sun':
                #Set the value to 1  
                data_set_row[i] = 6 /6
            #If the value is male 
            if data_set_row[i] == 'M':
                #Set the value to be .5
                data_set_row[i] = 1 /2
            #If the value if female 
            if data_set_row[i] == 'F':
                #Set the value to be 1  
                data_set_row[i] = 2 /2
            #if the value is infant 
            if data_set_row[i] == 'I':
                #Set the value to 0  
                data_set_row[i] = 0  /2
        #Return the updated dataset 
        return data_set_row

    #Parameters: N/a
    #Returns:  Return the centroid aray 
    #Function: randomly generate kvalue centroids by randomly generating an appropriate value per feature
    def create_random_centroids(self) -> np.ndarray:
        #Create a copy of the test data 
        sample_point = copy.deepcopy(self.Testdata[0,:])
        #Create an empty list 
        points = []
        #For each of the neighbors 
        for k in range(self.kValue):
            #Copy a new point 
            new_point = copy.deepcopy(sample_point)
            #for each of the features in the dimensionality 
            for feature in range(self.d):
                #Set the new poiunt at a specific feature value 
                new_point[feature] = random.uniform(0,1)
            #Append the new point to the list of points 
            points.append(new_point)
        #Add the points to the centroid array 
        centroid_array = np.concatenate(points).reshape(self.kValue, self.d+1)
        #Return the centroid array 
        return centroid_array

    #Parameters: Take in a given data point and the list of centroids 
    #Returns: Return the nearest centroid 
    #Function: find the nearest centroid to given sample, return the centroid index
    def closest_centroid_to_point(self, point: list, centroids: np.ndarray) -> list:
        # use the knn get_neighor class method to find the closest centroid 
        centroid = self.nn.get_k_neighbors(centroids, point, 1)
        # return the centroid index, element 1 of [distance, index, response var]
        # print(centroid)
        return centroid[0][1]
    #Parameters: Take in the list of centroids and the data 
    #Returns:  Return the centroid assignments 
    #Function: assign each data point in data set to the nearest centroid. This is stored in an array as an integer representing the centroid index at the index of the point belonging to it. 
    def assign_all_points_to_closest_centroid(self, centriods: np.ndarray, data: np.ndarray) -> list:
        centroid_assignments = [None] * len(data)
        # for each data point
        for i in range(len(data)):
            x = data[i].tolist()[:-1]
            # store the index of the centroid at the index corresponding to the sample position
            centroid_assignments[i] = self.closest_centroid_to_point(x, centriods)
        # return the list of indices
        return centroid_assignments
    #Parameters: Take in the list of centroids, the centroid assignmnets and the data 
    #Returns:  Return an updated list of centroids 
    #Function: The purpose of this function is to generate new centroid values and change the feature means to be the center of the data points assigned 
    def update_centroid_positions(self, centroids: np.ndarray, centroid_assignments: list, data: np.ndarray) -> np.ndarray:
        # centroids    
        #[c0, c1, c2, c3]
        # centroid assignments ex 1
        #[0, 2, 3, 1, 0, 1, 2, 3]
        # ex 2
        #[0, 0, 0, 0, 0, 0, 1]

        #Create a new list for mediod mean values 
        New_centroid = list() 
        #For each of the centroid
        Centroid_total = list()  
        for i in centroid_assignments: 
            if i not in Centroid_total: 
                Centroid_total.append(i)
        # iterate through centroids with data points assigned to it
        for i in Centroid_total: 
            #Loop through centroid assignments and store each index that belongs to an associated centroids 
            centroidTuples = list() 
            #For each of the centroids 
            for j in range(len(centroid_assignments)):
                #If the assignment is in a given range  
                if i == centroid_assignments[j]: 
                    #Append the value to the list 
                    centroidTuples.append(j)
            #Now we have a list of all records in the data array that belong to a specific centroid 
            #Get the total number of rows in each of the data points 
            # Rows = len(data[0])-1
            #Create a new list to store row mean 
            Mean = list()
            #For each of the features in the dataset 
            for j in range(self.d): 
                #Set the row count to 0 
                count = 0 
                #Store the total number of rows in the dataset 
                total = len(centroidTuples)
                #Loop through all of the rows in the data set 
                for z in centroidTuples: 
                    count += data[z][j]
                #Take the row count and divide by the total number of rows in the data set
                count = count / total 
                #Append the value to the list to store 
                Mean.append(count)
            new_centroid = copy.deepcopy(centroids[i])    
            for c in range(self.d):
                new_centroid[c] = Mean[c]
            #Add the entire mediods mean data to a centroid value
            centroids[i] = new_centroid
        #Return the mean values for each feature for each centroid its a lists of lists of lists         
        return centroids
    #Parameters: N/a
    #Returns:  Return the updated centroids list
    #Function: Generate all of the cluster centroids and update them 
    def generate_cluster_centroids(self):
        #Store the centroid from a random centroid value generated 
        centroids = self.create_random_centroids()
        #Store the first assignment to a given variable 
        first_assignment = self.assign_all_points_to_closest_centroid(centroids, self.dataSet)
        #Store the updated centroids for later recall 
        updated_centroids = self.update_centroid_positions(centroids, first_assignment, self.dataSet)
        #Set a counter variable to 0 
        count = 0
        #Continue to loop until we explicitly say break 
        while True:
            #Store the second assignment from the updated centroids and a given data set 
            second_assignment = self.assign_all_points_to_closest_centroid(updated_centroids, self.dataSet)
            #Store the updated centroids from the values above
            updated_centroids = self.update_centroid_positions(updated_centroids, second_assignment, self.dataSet)
            #Increment Count 
            count += 1
            #If the frist assignment is equal to the second assignment or the count is greater than the iteration limit set for a given object
            if first_assignment == second_assignment or count > self.itermax:
                #Break out of the loop
                break
            #Set the frist assignment to the second assignment 
            first_assignment = second_assignment
        #For each of th eupdated centroids 
        for i in range(len(updated_centroids)):
            #Change the classification of the centroid with the most common classes  
            updated_centroids[i][len(updated_centroids[i])-1] = self.CountCentroidClasses(i, second_assignment,self.dataSet)        
        #Return the updated centroids 
        return updated_centroids
    #Parameters: N/a
    #Returns:  Return the ground truth and the hypothesised truth 
    #Function:simple classify method that mirrors KNN, exept with the centroids as training set
    def classify(self):
        centroids = self.generate_cluster_centroids()
        test = self.assign_all_points_to_closest_centroid(centroids,self.Testdata)
        Hypothesis = list() 
        for i in range(len(test)):
            verifier = list() 
            guess = centroids[test[i]][len(centroids[test[i]])-1]
            real = self.Testdata[i][len(self.Testdata[i])-1]
            verifier.append(real)
            verifier.append(guess)
            Hypothesis.append(verifier)
        return Hypothesis
    #Parameters: Take in the centroid list, list of each data points centroid, the data set 
    #Returns:  Return the value that occured the most times 
    #Function: Find each of the classes that belong to a given centroid and return the classification value for a centroid with the most class occurences 
    def CountCentroidClasses(self,centroid, centroid_assignments,data): 
            centroidTuples = list() 
            #For each of the centroids 
            for j in range(len(centroid_assignments)):
                #If the assignment is in a given range  
                if centroid == centroid_assignments[j]: 
                    #Append the value to the list 
                    centroidTuples.append(j)
            #Now we have a list of all records in the data array that belong to a specific centroid 
            #Get the total number of rows in each of the data points 
            # Rows = len(data[0])-1
            classes = list()
            #For each of the data points in the centroid tuples 
            for z in centroidTuples: 
                #Add the classifcation value to the list 
                classes.append(data[z][len(data[0])-1])
            #If the list is empty 
            if not classes:
                #Return 0 
                return 0 
            #Return the value with the most occurrences 
            return max(classes,key=classes.count)
                




####################################### UNIT TESTING #################################################
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

    feature_data_types = {
        "segmentation": 'real',
        "vote": 'categorical',
        "glass": 'real',
        "fire": 'mixed',
        "machine": 'mixed',
        "abalone": 'mixed'
    }

    data_sets = ["segmentation", "vote", "glass", "fire", "machine", "abalone"]

    regression = [x for x in data_sets if regression_data_set[x]]

    for i in range(1):
        data_set = "vote"
        name = "vote"

        print("Data set: ", data_set)
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        # print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        test = copy.deepcopy(tenFolds[0])
        training = np.concatenate(tenFolds[1:])
        d = len(headers)-1
        kMC = kMeansClustering(kNeighbors=d,kValue=d, dataSet=training, data_type="real", categorical_features=[], regression_data_set=regression_data_set[data_set], alpha=1, beta=1, h=.5, d=d,name=name,Testdata = training)
        #kMC.classify()
        print(kMC.classify())
        #print(kMC.dataSet)

####################################### UNIT TESTING #################################################