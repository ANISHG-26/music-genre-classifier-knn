from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os 
import pickle 
import random
import operator
import math


cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
directory = os.path.join("data")
f=open("data.dat","wb")
i=0

for folder in os.listdir(directory):
    i+=1
    if i==11:
        break
    for file in os.listdir(os.path.join(directory,folder)):
        (rate,sig) = wav.read(os.path.join(directory,folder,file))
        mfcc_feat = mfcc(sig,rate,winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature,f)

f.close()

dataset = []
trainingSet = []
testSet = []


def loadDataset(filename, split , trSet, teSet):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


loadDataset("data.dat",0.66,trainingSet,testSet)

def distance(instance1, instance2, k):
    distance=0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2),cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def getNeighbors(trainingSet, instance, k):
    distances=[]
    for x in range (len(trainingSet)):
        #print(trainingSet[x])
        dist = distance(trainingSet[x],instance,k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote={}
    for x in range(len(neighbors)):
        response=neighbors[x]
        if response in classVote:
            classVote[response]+=1
        else:
            classVote[response]=1
    sorter = sorted(classVote.items(), key =operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            #print(testSet[x])
            correct+=1
    return 100.0 * correct/len(testSet)

leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet,testSet[x],5)))

accuracy1 = getAccuracy(testSet,predictions)
print(accuracy1)


