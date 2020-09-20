
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

if __name__ == '__main__':
    rd = RealDistance()
    d = rd.Distance([1,2,3],[2,3,4])
    print(d)