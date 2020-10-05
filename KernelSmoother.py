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
            u = distance / self.h

            numerator += self.gaussian_kernel(u) * responseVariable
            denominator += self.gaussian_kernel(u)

        return numerator/(denominator + .0000000000000000001)

    def gaussian_kernel(self, u) -> float:
        kernel_weight = (1/(math.sqrt(2 * math.pi)))**(self.d) * math.exp(-.5 * (u**2))
        return kernel_weight