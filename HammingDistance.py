
class HammingDistance:

    def Distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            if x1[i] == x2[i]:
                value = 0
            else:
                value = 1
            distance += value
        return distance

if __name__ == '__main__':
    hd = HammingDistance()
    d = hd.Distance([0,1,0],[1,1,1])
    print(d)