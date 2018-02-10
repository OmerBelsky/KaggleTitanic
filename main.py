import KNN
import numpy as np
import re
import Point as p
import sys
import argparse as argp
from sklearn import svm

parser = argp.ArgumentParser()
parser.add_argument('-m', default='k', dest='model', action='store_const', const='m')

knn = KNN.KNN(275)

def createDistrib():
    fares = []
    ages = []
    maxAge = 0
    maxFare = 0
    faremean = 0
    farestd = 0
    maxSibSp = 0
    maxParch = 0
    minSibSp = sys.maxsize
    minParch = sys.maxsize
    minAge = sys.maxsize
    minFare = float('inf')
    with open("./train.csv", 'r')  as reader:
        next(reader)
        for line in reader:
            line = re.split(",(?!\s)", line)
            age, sibsp, parch, fare = [line[i] for i in [5, 6, 7, 9]]
            try:
                age = int(float(age))
                fare = float(fare)
                sibsp = int(sibsp)
                parch = int(parch)
                fares.append(fare)
                ages.append(age)
                maxAge = max(maxAge, age)
                minAge = min(minAge, age)
                maxFare = max(maxFare, fare)
                minFare = min(minFare, fare)
                maxSibSp = max(maxSibSp, sibsp)
                minSibSp = min(minSibSp, sibsp)
                maxParch = max(maxParch, parch)
                minParch = min(minParch, parch)
            except Exception:
                pass
    faremean = np.mean(fares)
    farestd = np.std(fares)
    agemean = np.mean(ages)
    agestd = np.std(ages)
    return faremean, farestd, agemean, agestd, maxAge, minAge, maxFare, minFare, maxSibSp, minSibSp, maxParch, minParch

def predictWithKNN():
    for i in range(len(trainSet)):
        knn.addPoint(p.Point(trainSet[i], trainTargets[i]))
    classified = []
    for featureSet in testSet:
        classified.append(knn.classify(p.Point(featureSet)))
    with open("knn.csv", 'a') as writer:
        writer.write("PassengerId,Survived")
        for i in range(len(classified)):
            writer.write("\n{},{}".format(892 + i, classified[i]))

def predictWithSVM():
    clf = svm.SVC(kernel='rbf', C=5)
    clf.fit(trainSet, trainTargets)
    predictions = clf.predict(testSet)
    with open("svm.csv", 'a') as writer:
        writer.write("PassengerId,Survived")
        for i in range(len(predictions)):
            writer.write("\n{},{}".format(892 + i, predictions[i]))

def loadData():
    testSet = []
    trainSet = []
    trainTargets = []
    with open("./test.csv", 'r') as reader:
        next(reader)
        for line in reader:
            line = re.split(",(?!\s)", line)
            pClass, gender, age, sibsp, parch, fare = [line[i] for i in [1, 3, 4, 5, 6, 8]]
            features = createFeatures(pClass, gender, age, sibsp, parch, fare)
            testSet.append(features)
    with open("./train.csv", 'r') as reader:
        next(reader)
        for line in reader:
            line = re.split(",(?!\s)", line)
            survived, pClass, gender, age, sibsp, parch, fare = [line[i] for i in [1, 2, 4, 5, 6, 7, 9]]
            features = createFeatures(pClass, gender, age, sibsp, parch, fare)
            trainSet.append(features)
            trainTargets.append(survived)
    return trainSet, trainTargets, testSet

def createFeatures(pClass, gender, age, sibsp, parch, fare):
    predFare = np.random.normal(faremean, farestd)
    predAge = int(np.random.normal(agemean, agestd))
    predFare = min(predFare, minFare)
    predFare = max(predFare, maxFare)
    predAge = min(predAge, minAge)
    predAge = max(predAge, maxAge)
    features = [(float(pClass) - 1) / 2, 0 if gender == "male" else 1,
                int((predAge - minAge) / (maxAge - minAge)) if age == '' else int(
                    int(float(age) - minAge) / (maxAge - minAge)),
                (float(sibsp) - minSibSp) / (maxSibSp - minSibSp),
                (float(parch) - minParch) / (maxParch - minParch),
                (predFare - minFare) / (maxFare - minFare) if fare == '' else (float(fare) - minFare) / (
                    maxFare - minFare)]
    return features

faremean, farestd, agemean, agestd, maxAge, minAge, maxFare, minFare, maxSibSp, minSibSp, maxParch, minParch = createDistrib()
trainSet, trainTargets, testSet = loadData()
if parser.parse_args().model == 'k':
    predictWithKNN()
else:
    predictWithSVM()

