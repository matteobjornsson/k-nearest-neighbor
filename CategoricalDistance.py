
class CategoricalDistance:

    def __init__(self, dataSet):
        self.featureDifferenceMatrix = self.calculateFDM(dataSet)

    def calculateFDM(self, dataSet):
        print("calculating feature difference matrix")
        return dataSet

    def distance(self, x1, x2):
        print("returning distance based on FDM")
        return 8.5
        