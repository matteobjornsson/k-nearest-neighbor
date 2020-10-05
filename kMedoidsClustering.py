#Written by Matteo Bjornsson and Nick Stone  
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import copy, math, random
import kNN, DataUtility
import numpy as np


class kMedoidsClustering:

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
        # pass in the test data set at init 
        Testdata: np.ndarray):

        # create a Nearest Neighbor object to single nearest neighbor to input data point
        self.nn = kNN.kNN(1, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        self.knn = kNN.kNN(kNeighbors, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        self.categorical_features = categorical_features
        # save which features are real as well by deleting categorical indices from a new list
        real_features = list(range(d))
        for i in categorical_features:
            real_features.remove(i)
        self.real_features = real_features
        self.kValue = kValue
        self.dataSet = dataSet
        # dimensionality of data set
        self.d = d
        self.itermax = 5 
        self.Testdata = Testdata

 

    #Parameters: 
    #Returns:  
    #Function: 
    def choose_random_medoids(self):
        #Create a new empty array 
        indices = []
        #Loop through the number of neighbros we are looking at 
        for k in range(self.kValue):
            #SEt te index to be a random value within the dataset 
            index = random.randint(0, len(self.dataSet)-1)
            #make sure the inde is unique by generating a new value if the inde already exists
            while index in indices:
                #Set the index to a random value in the data set 
                index = random.randint(0, len(self.dataSet)-1)
            #Append the index to the arary 
            indices.append(index)
        #Create an empty list 
        medoids = []
        #For each of the indices 
        for i in indices:
            medoids.append(self.dataSet[i].reshape(1, self.dataSet.shape[1]))
        return np.concatenate(medoids)
    
    #Parameters: 
    #Returns:  
    #Function: 
    # find the nearest medoid to given sample, return the medoid index
    def closest_medoid_to_point(self, point: list, medoids: np.ndarray) -> list:
        # use the knn get_neighor class method to find the closest medoid 
        medoid = self.nn.get_k_neighbors(medoids, point, k=1)
        # return the medoid index, element 1 of [distance, index, response var]
        return medoid[0][1]
    
    #Parameters: 
    #Returns:  
    #Function: 
    # assign each data point in data set to the nearest medoid. This is stored in an array
    # as an integer representing the medoid index at the index of the point belonging to it. 
    def assign_all_points_to_closest_medoid(self, medoids: np.ndarray, data: np.ndarray) -> list:
        medoid_assignments = [None] * len(data)
        # for each data point
        for i in range(len(data)):
            x = data[i].tolist()[:-1]
            # store the index of the medoid at the index corresponding to the sample position
            medoid_assignments[i] = self.closest_medoid_to_point(x, medoids)
        # return the list of indices
        return medoid_assignments
    #Parameters: 
    #Returns:  
    #Function: 
    def distortion(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> float:
        #Set the distortion value to 0
        distortion = 0
        #Loop through the number of indices in the medoids array 
        for i in range(len(medoids)):
            #Store the current medoid we are looking at 
            m = medoids[i]
            points_in_cluster = []
            # for the current medoid, look up all examples x that are assigned
            # to that medoid (have a value at their index position in the medoid
            # assignment list that matches the current medoid)
            for n in range(len(medoid_assignments)):
                #store the medoid that point x is assigned to
                x_assignment = medoid_assignments[n]
                # if x is assigned to medoid i (current medoid), append the actual data point to a list
                if x_assignment == i:
                    # get the point and reshape it into a np array that can be concatenated together
                    points_in_cluster.append(data[n].reshape(1, data.shape[1]))
            if len(points_in_cluster) > 0:    
                points_in_cluster = np.concatenate(points_in_cluster)
                # use the knn method "get_k_neighbors" to calculate the distance from current medoid m to all points in the cluster
                point_distances = self.nn.get_k_neighbors(points_in_cluster, m, len(points_in_cluster))
                #For each of the points above 
                for point in point_distances:
                    distance_from_m = point[0]
                    #Add the distortion value to the variable for each of the points 
                    distortion += (distance_from_m)**2
        #Return the distortion 
        return distortion
    #Parameters: 
    #Returns:  
    #Function:    
    def update_medoids(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> np.ndarray:
        #Loop through the nunmber of indicies in the medoids array (Total number of medoids )
        for i in range(len(medoids)):
            #Store off the distortion value calculated in the above functions 
            distortion = self.distortion(medoids, medoid_assignments, data)
            #For each of the data points 
            for x in data:
                #Set a copy of the medoids above 
                new_medoids = copy.deepcopy(medoids)
                #Story off a given array to be a copy of the value x in data 
                new_medoids[i] = copy.deepcopy(x)
                #Calculate a new distortion from the copies above 
                new_distortion = self.distortion(new_medoids, medoid_assignments, data)
                #If the new distortion is less than the distortion calculated above 
                if new_distortion < distortion:
                    #Store off a deep copy of the sample x that is the new medoid
                    print("old medoid:", medoids[i])
                    print("new medoid:", x)
                    medoids[i] = copy.deepcopy(x)
            print("updating medoid for cluster", i)
        return medoids
    #Parameters: 
    #Returns:  
    #Function: 
    def generate_cluster_medoids(self):
        #Choose a random medoid and store the value 
        medoids = self.choose_random_medoids()
        #Store off the first assignment value 
        first_assignment = self.assign_all_points_to_closest_medoid(medoids, self.dataSet)
        #Store the update medoids value based on the first assignment 
        updated_medoids = self.update_medoids(medoids, first_assignment, self.dataSet)
        #Set a count to be 0
        count = 0
        print("count: ", count)
        while True:
            #Set a second assignment and store the value 
            second_assignment = self.assign_all_points_to_closest_medoid(updated_medoids, self.dataSet)
            #Store the updated medoids from the second assignment values calculated above 
            updated_medoids = self.update_medoids(updated_medoids, second_assignment, self.dataSet)
            #Increment count 

            # code for indicating if the medoid assignments are changing
            count += 1
            #Create an empty array 
            changing_assignments = []
            #For each of the values until the first assignment 
            for i in range(len(first_assignment)):
                #If the first assignment is not equal to the second assignment value 
                if first_assignment[i] != second_assignment[i]:
                    #Store thevalue off 
                    changing_assignments.append(i)
            print("medoid assignments that are changing", changing_assignments)
            #If the first is equal to the second or we are beyond the iteration limit set 
            if first_assignment == second_assignment or count > self.itermax:
                #Break 
                break
            #SEt the first assignment equal to the second assignment 
            first_assignment = second_assignment
        #Return the updated medoids 
        return updated_medoids
    #Parameters: 
    #Returns:  
    #Function: 
    def classify(self):
        medoids = self.generate_cluster_medoids()
        return self.knn.classify(medoids, self.Testdata)

####################################### UNIT TESTING #################################################
if __name__ == '__main__':
    print("program Start")
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

        print("Data set: ", data_set)
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        # print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        test = copy.deepcopy(tenFolds[0])
        training = np.concatenate(tenFolds[1:])

        d = len(headers)-1
        kMC = kMedoidsClustering(kValue=d, dataSet=training, data_type=feature_data_types[data_set], categorical_features=categorical_attribute_indices[data_set], regression_data_set=regression_data_set[data_set], alpha=1, beta=1, h=.5, d=d)
        medoids = kMC.generate_cluster_medoids()
        print("dataset medoids: ", medoids, f"(length: {len(medoids)})")
        print("original dataset: ", kMC.dataSet, f"(length: {len(kMC.dataSet)}")

    print("program end ")
    ####################################### UNIT TESTING #################################################