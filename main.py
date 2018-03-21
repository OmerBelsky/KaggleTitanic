import KNN
import numpy as np
import re
import Point as p
import sys
import argparse as argp
from sklearn import svm
import tensorflow as tf

parser = argp.ArgumentParser()
parser.add_argument('-m', default='k', dest='model', action='store_const', const='m')
parser.add_argument('-d', dest='model', action='store_const', const='d')

knn = KNN.KNN(275)

def createDistrib():
    fares = []
    ages = dict([])
    maxAge = 0
    maxFare = 0
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
            honorific = line[3].split(", ")[1].split(' ')[0]
            try:
                age = int(float(age))
                fare = float(fare)
                sibsp = int(sibsp)
                parch = int(parch)
                fares.append(fare)
                if honorific not in ages.keys():
                    ages[honorific] = [age]
                else:
                    ages[honorific].append(age)
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
    agemeans = {}
    agestds = {}

    for key in ages.keys():
        agemeans[key] = np.mean(ages[key])
        agestds[key] = np.std(ages[key])
    return faremean, farestd, agemeans, agestds, maxAge, minAge, maxFare, minFare, maxSibSp, minSibSp, maxParch, minParch

def predictWithKNN():
    for i in range(len(trainSet)):
        knn.addPoint(p.Point(trainSet[i], trainTargets[i]))
    classified = []
    for featureSet in testSet:
        classified.append(knn.classify(p.Point(featureSet)))
    with open("knn2.csv", 'a') as writer:
        writer.write("PassengerId,Survived")
        for i in range(len(classified)):
            writer.write("\n{},{}".format(892 + i, classified[i]))

def predictWithSVM():
    clf = svm.SVC(kernel='rbf', C=5)
    clf.fit(trainSet, trainTargets)
    predictions = clf.predict(testSet)
    with open("svm2.csv", 'a') as writer:
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
            honorific = line[2].split(", ")[1].split(' ')[0]
            features = createFeatures(pClass, gender, age, sibsp, parch, fare, honorific)
            testSet.append(features)
    with open("./train.csv", 'r') as reader:
        next(reader)
        for line in reader:
            line = re.split(",(?!\s)", line)
            survived, pClass, gender, age, sibsp, parch, fare = [line[i] for i in [1, 2, 4, 5, 6, 7, 9]]
            honorific = line[3].split(", ")[1].split(' ')[0]
            features = createFeatures(pClass, gender, age, sibsp, parch, fare, honorific)
            trainSet.append(features)
            trainTargets.append(survived)
    return trainSet, trainTargets, testSet

def createFeatures(pClass, gender, age, sibsp, parch, fare, honorific):
    predFare = faremean
    if honorific not in agemeans.keys():
        predAge = float(sum(agemeans.values())) / len(agemeans)
    else:
        predAge = agemeans[honorific]
    predFare = min(predFare, minFare)
    predFare = max(predFare, maxFare)
    predAge = min(predAge, minAge)
    predAge = max(predAge, maxAge)
    features = [(float(pClass) - 1) / 2, 0 if gender == "male" else 1,
                (predAge - minAge) / (maxAge - minAge) if age == '' else
                    float((float(age) - minAge)) / float((maxAge - minAge)),
                    (float(sibsp) - minSibSp) / (maxSibSp - minSibSp),
                (float(parch) - minParch) / (maxParch - minParch),
                (predFare - minFare) / (maxFare - minFare) if fare == '' else (float(fare) - minFare) / (maxFare - minFare)]
    return features

def predictWithDNN():
    featureColumns = [tf.contrib.layers.real_valued_column("", dimension=6)]
    trainFile = "DNNTrain.csv"
    testFile = "DNNTest.csv"
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=featureColumns, hidden_units=[10, 20, 30, 20, 10], n_classes=2)
    trainData = tf.contrib.learn.datasets.base.load_csv_with_header(filename=trainFile, target_dtype=np.int, features_dtype=np.float)
    testData = tf.contrib.learn.datasets.base.load_csv_with_header(filename=testFile, target_dtype=np.int, features_dtype=np.float)
    classifier.fit(input_fn=lambda: (tf.constant(trainData.data), tf.constant(trainData.target)), steps=828)
    predictions = classifier.predict_classes(input_fn=lambda: (tf.constant(testData.data), tf.constant(testData.target)))
    with open("dnn.csv", 'a') as writer:
        writer.write("PassengerId,Survived")
        i = 892
        for prediction in predictions:
            writer.write('\n')
            writer.write("{},{}".format(i, prediction))
            i += 1



faremean, farestd, agemeans, agestds, maxAge, minAge, maxFare, minFare, maxSibSp, minSibSp, maxParch, minParch = createDistrib()
trainSet, trainTargets, testSet = loadData()
if parser.parse_args().model == 'k':
    predictWithKNN()
if parser.parse_args().model == 'm':
    predictWithSVM()
if parser.parse_args().model == 'd':
    predictWithDNN()
