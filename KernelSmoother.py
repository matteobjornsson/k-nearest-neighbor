import math

class KernelSmoother:

    def __init__(self, h, d):
        # bandwidth h
        self.h = h
        # dimensionality of examples 
        self.d = d

    def estimate(self, neighborStats: list) -> float:
        # neighbors must be a list of [distance from sample, index, response variable]
        # for each of the k neighbors
        N = len(neighborStats)
        numerator = 0
        denominator = 0

        for i in range(N):
            neighbor = neighborStats[i]
            distance, responseVariable = neighbor[0], neighbor [2]

            numerator += self.gaussian_kernel(distance / self.h) * responseVariable
            denominator += self.gaussian_kernel(distance / self.h)
        return numerator/denominator

    def gaussian_kernel(self, u) -> float:
        return (1/(math.sqrt(2 * math.pi)))**(self.d) * math.exp(-.5 * (u**2))