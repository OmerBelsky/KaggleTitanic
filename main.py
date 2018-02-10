import KNN
import numpy as np
import re
import Point as p
import sys
from sklearn import svm

knn = KNN.KNN(275)
agemean = 0
agestd = 0
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
SVMMat = []
SVMTargets = []

def createDistrib():
    global faremean
    global farestd
    global agemean
    global agestd
    global maxAge
    global maxFare
    global maxSibSp
    global maxParch
    global minAge
    global minFare
    global minSibSp
    global minParch
    fares = []
    ages = []
    with open("./train.csv", 'r')  as reader:
        for line in reader:
            line = re.split(",(?!\s)", line)
            try:
                fares.append(int(float(line[9])))
                ages.append(int(float(line[5])))
                if maxAge < int(float(line[5])):
                    maxAge = int(float(line[5]))
                if minAge > int(float(line[5])):
                    minAge = int(float(line[5]))
                if maxFare < float(line[9]):
                    maxFare = float(line[9])
                if maxSibSp < int(line[6]):
                    maxSibSp = int(line[6])
                if maxParch < int(line[7]):
                    maxParch = int(line[7])
                if minFare > float(line[9]):
                    minFare = float(line[9])
                if minSibSp > int(line[6]):
                    minSibSp = int(line[6])
                if minParch > int(line[7]):
                    minParch = int(line[7])
            except Exception:
                pass #age was null
    agemean = np.mean(fares)
    agestd = np.std(fares)
    agemean = np.mean(ages)
    agestd = np.std(ages)

def loadDataKNN():
    with open("./train.csv", 'r') as reader:
        readHead = False
        for line in reader:
            if not readHead:
                readHead = True
                continue
            line = re.split(",(?!\s)", line)
            predFare = np.random.normal(faremean, farestd)
            predAge = int(np.random.normal(agemean, agestd))
            if predFare < minFare:
                predFare = minFare
            if predFare > maxFare:
                predFare = maxFare
            if predAge < minAge:
                predAge = minAge
            if predAge > maxAge:
                predAge = maxAge
            features = [(float(line[2]) - 1) / 2, 0 if line[4] == "male" else 1,
                        int((predAge - minAge) / (maxAge - minAge)) if line[5] == '' else int(
                            int(float(line[5]) - minAge) / (maxAge - minAge)),
                        (float(line[6]) - minSibSp) / (maxSibSp - minSibSp),
                        (float(line[7]) - minParch) / (maxParch - minParch),
                        (predFare - minFare) / (maxFare - minFare) if line[9] == '' else (float(line[9]) - minFare) / (
                        maxFare - minFare)]
            knn.addPoint(p.Point(features, line[1]))

def learnStuffKNN():
    classified = []
    with open("./test.csv", 'r') as reader:
        readHead = False
        for line in reader:
            if not readHead:
                readHead = True
                continue
            line = re.split(",(?!\s)", line)
            predFare = np.random.normal(faremean, farestd)
            predAge = int(np.random.normal(agemean, agestd))
            if predFare < minFare:
                predFare = minFare
            if predFare > maxFare:
                predFare = maxFare
            if predAge < minAge:
                predAge = minAge
            if predAge > maxAge:
                predAge = maxAge
            features = [(float(line[1]) - 1) / 2, 0 if line[3] == "male" else 1,
                        int((predAge - minAge) / (maxAge - minAge)) if line[4] == '' else int(
                            int(float(line[4]) - minAge) / (maxAge - minAge)),
                        (float(line[5]) - minSibSp) / (maxSibSp - minSibSp),
                        (float(line[6]) - minParch) / (maxParch - minParch),
                        (predFare - minFare) / (maxFare - minFare) if line[8] == '' else (float(line[8]) - minFare) / (
                        maxFare - minFare)]
            point = p.Point(features)
            classified.append((line[0], knn.classify(point)))
    with open("knn.csv", 'a') as writer:
        writer.write("PassengerId,Survived")
        for i in range(len(classified)):
            writer.write("\n{},{}".format(classified[i][0], classified[i][1]))

def loadDataSVM():
    global SVMMat
    global SVMTargets
    with open("./train.csv", 'r') as reader:
        readHead = False
        for line in reader:
            if not readHead:
                readHead = True
                continue
            line = re.split(",(?!\s)", line)
            predFare = np.random.normal(faremean, farestd)
            predAge = int(np.random.normal(agemean, agestd))
            if predFare < minFare:
                predFare = minFare
            if predFare > maxFare:
                predFare = maxFare
            if predAge < minAge:
                predAge = minAge
            if predAge > maxAge:
                predAge = maxAge
            features = [(float(line[2]) - 1) / 2, 0 if line[4] == "male" else 1,
                        int((predAge - minAge) / (maxAge - minAge)) if line[5] == '' else int(
                            int(float(line[5]) - minAge) / (maxAge - minAge)),
                        (float(line[6]) - minSibSp) / (maxSibSp - minSibSp),
                        (float(line[7]) - minParch) / (maxParch - minParch),
                        (predFare - minFare) / (maxFare - minFare) if line[9] == '' else (float(line[9]) - minFare) / (
                        maxFare - minFare)]
            SVMMat.append(features)
            SVMTargets.append(line[1])

def learnStuffSVM():
    clf = svm.SVC(kernel='rbf', C=5)
    clf.fit(SVMMat, SVMTargets)
    toBeClassified = []
    with open("./test.csv", 'r') as reader:
        readHead = False
        for line in reader:
            if not readHead:
                readHead = True
                continue
            line = re.split(",(?!\s)", line)
            predFare = np.random.normal(faremean, farestd)
            predAge = int(np.random.normal(agemean, agestd))
            if predFare < minFare:
                predFare = minFare
            if predFare > maxFare:
                predFare = maxFare
            if predAge < minAge:
                predAge = minAge
            if predAge > maxAge:
                predAge = maxAge
            features = [(float(line[1]) - 1) / 2, 0 if line[3] == "male" else 1,
                        int((predAge - minAge) / (maxAge - minAge)) if line[4] == '' else int(
                            int(float(line[4]) - minAge) / (maxAge - minAge)),
                        (float(line[5]) - minSibSp) / (maxSibSp - minSibSp),
                        (float(line[6]) - minParch) / (maxParch - minParch),
                        (predFare - minFare) / (maxFare - minFare) if line[8] == '' else (float(line[8]) - minFare) / (
                        maxFare - minFare)]
            toBeClassified.append(features)
    predictions = clf.predict(toBeClassified)
    with open("svm.csv", 'a') as writer:
        writer.write("PassengerId,Survived")
        for i in range(len(predictions)):
            writer.write("\n{},{}".format(892 + i, predictions[i]))

createDistrib()
loadDataKNN()
learnStuffKNN()
#loadDataSVM()
#learnStuffSVM()

