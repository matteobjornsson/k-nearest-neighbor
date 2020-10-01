import kNN
import Results
class CondensedKNN:

    def __init__(self, 
        error: float,
        k: int,
        data_type: str,
        categorical_features: list,
        regression_data_set: bool,
        alpha:float,
        beta:float,
        h:float,
        d:int):

        # initialize a knn object
        self.knn = kNN.kNN(k, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        # error threshhold for regression classification
        self.error = error
        # store if this data set is a regression data set (True) or not (False)
        self.regression_data_set = regression_data_set
        self.results = Results.Results()