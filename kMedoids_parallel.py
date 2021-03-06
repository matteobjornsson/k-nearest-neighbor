# new code in this file from kMedoidsClustering.py Written by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
# This file is a mirror of kMedoidsClustering.py but with much of the distortion code parallelized
##################################################################### MODULE COMMENTS ############################################################################
import copy, random
import multiprocessing
import kNN, DataUtility, kMedoidsClustering
import numpy as np


class kMedoids_parallel:
    #on the creation of a given object run the following 
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
        self.itermax = 10 
        self.Testdata = Testdata
        self.initial_medoids = self.choose_random_medoids()
        self.assignments = []

 

    #Parameters: N/a
    #Returns:  Return the numpy array of medoids 
    #Function: Generate a random list of medoids 
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
        #Reprtun the numpy array of medoids 
        return np.concatenate(medoids)
    
    #Parameters: Take in a list of points and the list of medoids 
    #Returns:  Return the nearest medoid 
    #Function: find the nearest medoid to given sample, return the medoid index
    def closest_medoid_to_point(self, point: list, medoids: np.ndarray) -> list:
        # use the knn get_neighor class method to find the closest medoid 
        medoid = self.nn.get_k_neighbors(medoids, point, k=1)
        # return the medoid index, element 1 of [distance, index, response var]
        return medoid[0][1]
    
    #Parameters: Take in the list of medoids and all of the data points 
    #Returns:  Return the medoid assignment for each data point taken in 
    #Function:  assign each data point in data set to the nearest medoid. This is stored in an array as an integer representing the medoid index at the index of the point belonging to it. 
    def assign_all_points_to_closest_medoid(self, medoids: np.ndarray, data: np.ndarray) -> list:
        medoid_assignments = [None] * len(data)
        # for each data point
        for i in range(len(data)):
            x = data[i].tolist()[:-1]
            # store the index of the medoid at the index corresponding to the sample position
            medoid_assignments[i] = self.closest_medoid_to_point(x, medoids)
        # return the list of indices
        return medoid_assignments

    #Parameters: Take in the medoids, the medoid assignments and the data array 
    #Returns:  Returns the distorion value 
    #Function: Generate and return the distortion value based on the given points in the medoids 
    def distortion(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> list:
        #Set the distortion value to 0
        distortion = [0] * len(medoids)
        #Loop through the number of indices in the medoids array 
        for i in range(len(medoids)):
            #Store the current medoid we are looking at 
            m = medoids[i].tolist()[:-1]
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
                    distortion[i] += (distance_from_m)**2
        #Return the distortion 
        return distortion

    # calculate distortion piecewise. Each component of distortion is unique to the medoid
    # and can be calculated with only that medoid and the data set
    def distortion_parallel(self, medoids, medoid_assignments):

        #start some multiprocessing tools
        manager = multiprocessing.Manager()
        q = manager.Queue()
        pool = multiprocessing.Pool()

        # for every medoid, calculate the distortion of that medoid
        for i in range(len(medoids)):
            pool.apply_async(
                    self.per_medoid_distortion(i, medoids[i], medoid_assignments, q)
                    # callback=log_results
                    )
        pool.close()
        pool.join()

        # store all the calculations and return 
        distortion = [0] * len(medoids)
        while not q.empty():
            medoid_index, new_distortion = q.get()
            # medoid_index, new_distortion = res.get()
            distortion[medoid_index] = new_distortion
        return distortion

        
    #Parameters: Take in the medoids, the medoid assignments and the data 
    #Returns: return the list of updated medoid values 
    #Function: Update all of th emedoid feature values 
    def update_medoids_parallel(self, medoids: np.ndarray, medoid_assignments: list, data: np.ndarray) -> np.ndarray:
        # calculate the initial distortion in parallel
        initial_distortion = self.distortion_parallel(medoids, medoid_assignments)

        # start some multiprocessing tools
        manager = multiprocessing.Manager()
        q = manager.Queue()
        # accumulator listens for new distortion values generated by workers
        # and keeps them if they are smaller than the current value
        accumulator = multiprocessing.Process(target=self.update_processor, args=(q, medoids, initial_distortion))
        accumulator.start()

        pool = multiprocessing.Pool()
        results = []
        # for every medoid and for every data point, calculate that points distortion
        for j in range(len(medoids)):
            for i in range(len(data)):
                results.append(pool.apply_async(self.queue_new_medoid_i_distortion, args=(q, j, i, initial_distortion[j], medoid_assignments)))
                
        print()
        pool.close()
        pool.join()
        q.put('kill')
        # get the final product from the queue, placed by the accumulator
        updated_medoids = q.get()
        # print("updated medoids", updated_medoids)
        accumulator.join()
        for r in results:
            r = r.get()
        print("Medoids updated")

        return updated_medoids

    # multiprocessing worker function:
    # for a given point, calculate it's distortion and queue it track which medoid we are considering to replace
    def queue_new_medoid_i_distortion(self, q, medoid_index, data_index, initial_distortion_i, medoid_assignments):
        # calculate the distortion of the new data point
        medoid_index, new_distortion_i = self.per_medoid_distortion(medoid_index, self.dataSet[data_index], medoid_assignments)
        # if the distortion is smaller than the distortion of the medoid it is proposed to be replacing, queue the value
        if new_distortion_i < initial_distortion_i:
            # place the smaller distortion on the queue
            q.put([medoid_index, new_distortion_i, data_index])

    # function for calculating the distortion of a given point/medoid
    def per_medoid_distortion(self, medoid_index, medoid, medoid_assignments, q=None):
        medoid_position = medoid.tolist()[:-1]
        distortion_i = 0
        # collect all data points that belong to the cluster identified by the medoid index
        cluster_members = []
        for j in range(len(medoid_assignments)):
            assignment = medoid_assignments[j]
            if assignment == medoid_index:
                cluster_members.append(self.dataSet[j].reshape(1, self.dataSet.shape[1]))
        # check if there are any members
        if len(cluster_members) > 0:
            # calcluate the distances from the medoid to all the cluster members using 
            # the already written knn function get_k_neighbors
            cluster_members = np.concatenate(cluster_members)
            cluster_point_distances = self.nn.get_k_neighbors(cluster_members, medoid_position, len(cluster_members))
            # for each point in the cluster, get its distance and add it's 
            # squared value to the distortion aggregator
            for cluster_point in cluster_point_distances:
                
                distance_to_proposed_medoid = cluster_point[0]
                distortion_i += (distance_to_proposed_medoid)**2
        # for flexibility, if a queue is given, queue the distortion, otherwise return it
        if q is None:
            return[medoid_index, distortion_i]
        else:
            q.put([medoid_index, distortion_i])

    # target function for aggregating parallel computations
    # collect all computations and save them when they are smaller than the current distortion
    def update_processor(self, q, current_medoids, initial_distortion):

        medoids = copy.deepcopy(current_medoids)
        count = 0
        # listen for new computations
        while True:
            # get the next computation
            distortion_element = q.get()
            # if it is a flag to stop working, return the result and break
            if distortion_element == 'kill':
                q.put(medoids)
                break
            # for the given medoid index, check if the new distortion value is smaller
            medoid_index, new_distortion_i, new_x_index = distortion_element
            initial_distortion_i = initial_distortion[medoid_index]
            # if it is, replace it with the ne value
            if new_distortion_i < initial_distortion_i:
                initial_distortion[medoid_index] = new_distortion_i
                medoids[medoid_index] = self.dataSet[new_x_index]

    #Parameters: N/a
    #Returns:  Return the list of updated medoid values 
    #Function: Generate and update the feature mean values for each medoids 
    def generate_cluster_medoids(self):
        #Store off the first assignment value 
        first_assignment = self.assign_all_points_to_closest_medoid(self.initial_medoids, self.dataSet)
        #Store the update medoids value based on the first assignment 
        updated_medoids = self.update_medoids_parallel(self.initial_medoids, first_assignment, self.dataSet)
        #Set a count to be 0
        count = 0
        # print("count: ", count)
        while True:
            #Set a second assignment and store the value 
            second_assignment = self.assign_all_points_to_closest_medoid(updated_medoids, self.dataSet)


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
            #Store the updated medoids from the second assignment values calculated above 
            updated_medoids = self.update_medoids_parallel(updated_medoids, second_assignment, self.dataSet)
            #Increment count 
            #SEt the first assignment equal to the second assignment 
            first_assignment = second_assignment
        #Return the updated medoids 
        return updated_medoids
    #Parameters: N/a
    #Returns:  Return the classification list 
    #Function: Return the classifcation of the test data based on the medoids 
    def classify(self):
        #Store the generated random medoids 
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

    data_sets = [ "segmentation", "vote", "glass", "fire", "machine", "abalone"]


    tuned_k = {
        "segmentation": 2,
        "vote": 5,
        "glass": 2,
        "fire": 2,
        "machine": 5,
        "abalone": 12
    }
    tuned_bin_value = {
        "segmentation": .25,
        "vote": .25,
        "glass": .25,
        "fire": .1,
        "machine": .25,
        "abalone": .1
    }

    tuned_delta_value = {
        "segmentation": .25,
        "vote": .25,
        "glass": .25,
        "fire": .5,
        "machine": .1,
        "abalone": .5
    }

    tuned_error_value = {
        "fire": 1,
        "abalone": 1,
        "machine":2
    }

    tuned_cluster_number = {
        "segmentation": 80,
        "vote": 15,
        "glass": 60,
        # not sure about fire, weird behavior
        "fire": 60,
        "machine": 50,
        "abalone": 50

    }

    for i in range(1):
        data_set = "vote"

        print("Data set: ", data_set)
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        test = copy.deepcopy(tenFolds[0])
        training = np.concatenate(tenFolds[1:])

        d = len(headers)-1
        kMC_p = kMedoidsClustering_P(
            kNeighbors=tuned_k[data_set],
            kValue=5,
            dataSet=training,
            data_type=feature_data_types[data_set],
            categorical_features=categorical_attribute_indices[data_set],
            regression_data_set=regression_data_set[data_set],
            alpha=1,
            beta=1,
            h=tuned_bin_value[data_set],
            d=d,
            Testdata=test
        )
        kMC = kMedoidsClustering.kMedoidsClustering(
            kNeighbors=tuned_k[data_set],
            kValue=5,
            dataSet=training,
            data_type=feature_data_types[data_set],
            categorical_features=categorical_attribute_indices[data_set],
            regression_data_set=regression_data_set[data_set],
            alpha=1,
            beta=1,
            h=tuned_bin_value[data_set],
            d=d,
            Testdata=test
        )
        kMC.initial_medoids = kMC_p.initial_medoids
        medoids_p = kMC_p.generate_cluster_medoids()
        print(kMC_p.classify())
        # print("dataset medoids: ", medoids, f"(length: {len(medoids)})")
        # print("original dataset: ", kMC.dataSet, f"(length: {len(kMC.dataSet)}")

    print("program end ")
    ####################################### UNIT TESTING #################################################
