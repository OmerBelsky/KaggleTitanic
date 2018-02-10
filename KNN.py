class KNN:

    def __init__(self, K):
        self._K = K
        self._dataSet = []

    def addPoint(self, point):
        self._dataSet.append(point)

    def classify(self, point):
        kClosest = sorted(self._dataSet, key=lambda x: point.dist(x))[0:self._K]
        classTally = 0
        for i in range(len(kClosest)):
            if kClosest[i].classValue == '0':
                classTally += 1
            else:
                classTally -= 1

        if classTally < 0:
            return '1'
        else:
            return '0'



