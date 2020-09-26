import math

class GaussianKernel:

    def __init__(self, d, sigma):
        self.d = d
        self.sigma = sigma

    def estimate(self, neighborStats: list) -> float:
        # neighbors must be a list of [distance from sample, index, response variable]
        # for each of the k neighbors
        N = len(neighborStats)
        numerator = 0
        denominator = 0
        for i in range(N):
            neighbor = neighborStats[i]
            distance, responseVariable = neighbor[0], neighbor [2]
            numerator += self.kernelSmoother(distance, self.sigma) * responseVariable
            denominator += self.kernelSmoother(distance, self.sigma)
        return numerator/denominator

    def kernelSmoother(self, distance, sigma) -> float:
        return math.exp(-(distance**2)/(sigma**2))