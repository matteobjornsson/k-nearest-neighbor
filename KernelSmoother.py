#Written by Matteo Bjornsson Edited by Nick Stone  
#################################################################### MODULE COMMENTS ############################################################################
#The purpose of this class is to create a kernel smoother that will run a gaussian kernel to gneerate a given kernel weight for each of the value               #
#################################################################### MODULE COMMENTS ############################################################################
import math

class KernelSmoother:

    #Parameters: Bandwidth h, and dimensionality of examples 
    #Returns: N/a  set object variables 
    #Function:  Take in the bandwidth h and dimensionality and set the class variables to be these funcitons -> Initialization function 
    def __init__(self, h, d):
        # bandwidth h
        self.h = h
        # dimensionality of examples 
        self.d = d

    #Parameters: A list of Neighbor statistics 
    #Returns:  Return a float for the estimation value 
    #Function: Take in the neighbor statistcs and generate an estimation value 
    def estimate(self, neighborStats: list) -> float:
        # neighbors must be a list of [distance from sample, index, response variable]
        # for each of the k neighbors
        N = len(neighborStats)
        #Set the Numerator variable to be 0 
        numerator = 0
        #Set the denominator variable to be 0 
        denominator = 0
        #Loop through each of the neighbor statistics 
        for i in range(N):
            #Track the current neighbors stats 
            neighbor = neighborStats[i]
            #Store the disance and response variable of a given neighbor 
            distance, responseVariable = neighbor[0], neighbor [2]
            #Set the kernel input value 
            u = distance / self.h
            #Increment the numerator value with the guassian kernel value generated with u and the response variable from the neighbor 
            numerator += self.gaussian_kernel(u) * responseVariable
            #Incrmeent the denominator value with the gaussian kernel value generated with u
            denominator += self.gaussian_kernel(u)
        #Return the difference in the numerator and the denominator 
        return numerator/(denominator + .0000000000000000001)
    
    #Parameters:  Take in an input u 
    #Returns:  Return a float of the kernel weight 
    #Function:   Generate the kernel weight from the input u 
    def gaussian_kernel(self, u) -> float:
        #Run the following mathematical equation on input U to generate the kernel weight 
        kernel_weight = (1/(math.sqrt(2 * math.pi)))**(self.d) * math.exp(-.5 * (u**2))
        #Return the kernel weight 
        return kernel_weight