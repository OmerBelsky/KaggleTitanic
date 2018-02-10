import math

class Point:

    def __init__(self, features, classValue="undetermined"):
        self._features = features
        self._classValue = classValue

    @property
    def features(self):
        return self._features

    @property
    def classValue(self):
        return self._classValue

    def dist(self, point):
        dist = 0.0
        for i in range(len(self._features)):
            dist += (self._features[i] - point.features[i])**2
        return math.sqrt(dist)
