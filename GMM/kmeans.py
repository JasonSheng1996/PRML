from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
import copy as cp

################# the following code of K-means comes from the internet ####################################
################# Since I just use it as a choice of EM's initiation, I didm't bother to implement it by myself ###################
def isdistinct(means,criter):
    K = len(means)
    for i in range(K):
        for j in range(i+1,K):
            if criter > np.linalg.norm(means[i]-means[j]):
                return False
    return True

def getmeans(data,K,criter):
    means = [0]*K
    dim  = np.shape(data)[1]
    minmax = []
    for i in range(dim):
        minmax.append(np.array([min(data[:,i]),max(data[:,i])]))

    while True:
        for i in range(K):
            means[i] = []
            for j in range(dim):
                means[i].append(np.random.random()*(minmax[j][1]-minmax[j][0])+minmax[j][0])
            means[i] = np.array(means[i])

        if isdistinct(means,criter):
            break
    return means


def kmeans(data,K):
    N = np.shape(data)[0]
    dim = np.shape(data)[1]

    means = getmeans(data,K,criter=10)
    means_old = [np.zeros(dim) for k in range(K)]

    while np.sum([np.linalg.norm(means_old[k]-means[k]) for k in range(K)]) > 0.01:

        means_old = cp.deepcopy(means)

        numlog = [0]*K
        sumlog = [np.zeros(dim) for k in range(K)]
        for n in range(N):
            distlog = [np.linalg.norm(data[n]-means[k]) for k in range(K)]
            toK = distlog.index(np.min(distlog))

            numlog[toK] += 1
            sumlog[toK] += data[n]

        for k in range(K):
            means[k] = 1.0/numlog[k]*sumlog[k]
    return means