#Written by Matteo Bjornsson edited by Nick Stone 
#################################################################### MODULE COMMENTS ############################################################################
#The purpose of this class is to create an object that can calculate the distance between two neighbors with real valued data sets.                             # 
#################################################################### MODULE COMMENTS ############################################################################
class RealDistance:

    def __init__(self):
        self.pValue = 2

    def Distance(self, x1, x2):
        p = self.pValue
        distance = 0
        for i in range(len(x1)):
            distance += (abs(x1[i] - x2[i])**p)
        distance = distance**(1/p)
        return distance


####################################### UNIT TESTING #################################################
if __name__ == '__main__':
    rd = RealDistance()
    d = rd.Distance([1,2,3],[2,9,4])
    assert round(d, 9) == 7.141428429
    print(d)