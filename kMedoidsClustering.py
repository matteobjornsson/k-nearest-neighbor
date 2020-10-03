#Written by Nick Stone and Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import copy, math, random
import kNN, DataUtility
import numpy as np


class kMedoidsClustering:

    def __init__(self,
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
        d: int):

        # create a Nearest Neighbor object to single nearest neighbor to input data point
        self.nn = kNN.kNN(1, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
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

 


    def choose_random_medoids(self):
        #Create a new empty array 
        indices = []
        #Loop through the number of neighbros we are looking at 
        for k in range(self.kValue):
            #SEt te index to be a random value within the dataset 
            index = random.randint(0, len(self.dataSet)-1)
            #Loop through each of the indexs in the indices list 
            while index in indices:
                #Set the index to a random value in the data set 
                index = random.randint(0, len(self.dataSet)-1)
            #Append the index to the arary 
            indices.append(index)
        #Create an empty list 
        medoids = []
        #For each of the indices 
        for i in indices:
            #Append to medoids the value of the indicies 
            medoids.append([i, self.dataSet[i]].reshape(1, self.dataSet.shape[1]))
        #Return the numpy array 
        return np.concatenate(medoids)

    # find the nearest medoid to given sample, return the medoid index
    def closest_medoid_to_point(self, point: list, medoids: np.ndarray) -> list:
        # use the knn get_neighor class method to find the closest medoid 
        medoid = self.nn.get_k_neighbors(medoids, point, k=1)
        # return the medoid index, element 1 of [distance, index, response var]
        return medoid[1]
    
    # assign each data point in data set to the nearest medoid. This is stored in an array
    # as an integer representing the medoid index at the index of the point belonging to it. 
    def assign_all_points_to_closest_medoid(self, centriods: np.ndarray, data: np.ndarray) -> list:
        medoid_assignments = []
        # for each data point
        for i in range(len(data)):
            x = data[i].tolist()[:-1]
            # store the index of the medoid at the index corresponding to the sample position
            medoid_assignments[i] = self.closest_medoid_to_point(x, centriods)
        # return the list of indices
        return medoid_assignments

    def distortion(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> float:
        #Set the distortion value to 0
        distortion = 0
        #Loop through the number of indices in the medoids array 
        for i in range(len(medoids)):
            #Store the current medoid we are looking at 
            m = medoids[i]
            #Set a value to be the NP array of the data points that are in the medoids cluster 
            points_in_cluster = np.concatenate([data[x].reshape(1, data.shape[1]) for x, j in enumerate(medoid_assignments) if j == i])
            #Set the point distances from the neighbors that the given point is looking at 
            point_distances = self.nn.get_k_neighbors(points_in_cluster, m, len(points_in_cluster))
            #For each of the points above 
            for point in point_distances:
                #Add the distortion value to the variable for each of the points 
                distortion += (point[0])**2
        #Return the distortion 
        return distortion
            
    def update_medoids(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> np.ndarray:
        for i in range(len(medoids)):
            distortion = self.distortion(medoids, medoid_assignments, data)
            for x in data:
                new_medoids = copy.deepcopy(medoids)
                new_medoids[i] = copy.deepcopy(x)
                new_distortion = self.distortion(new_medoids, medoid_assignments, data)
                if new_distortion < distortion:
                    medoids[i] = copy.deepcopy(x)
        return medoids

    def generate_cluster_medoids(self):
        medoids = self.choose_random_medoids()
        first_assignment = self.assign_all_points_to_closest_medoid(medoids, self.dataSet)
        updated_medoids = self.update_medoids(medoids, first_assignment, self.dataSet)
        count = 0
        while True:
            second_assignment = self.assign_all_points_to_closest_medoid(updated_medoids, self.dataSet)
            updated_medoids = self.update_medoids(updated_medoids, second_assignment, self.dataSet)
            count += 1
            if first_assignment == second_assignment or count > self.itermax:
                break
            first_assignment = second_assignment
        return updated_medoids




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


    print("program end ")