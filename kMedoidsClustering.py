#Written by 
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
        indices = []
        for k in range(self.kValue):
            index = random.randint(0, len(self.dataSet)-1)
            while index in indices:
                index = random.randint(0, len(self.dataSet)-1)
            indices.append(index)
        medoids = []
        for i in indices:
            medoids.append([i, self.dataSet[i]].reshape(1, self.dataSet.shape[1]))
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
        distortion = 0
        for i in range(len(medoids)):
            m = medoids[i]
            points_in_cluster = np.concatenate([data[x].reshape(1, data.shape[1]) for x, j in enumerate(medoid_assignments) if j == i])
            self.nn.get_k_neighbors(points_in_cluster, m, len(points_in_cluster))
            for point in points_in_cluster:



    def update_medoids(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> np.ndarray:
        
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