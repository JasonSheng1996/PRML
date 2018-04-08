from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
from kmeans import *


class EM_GMM(object):
    def __init__(self, K,data):
        self.data = load_data(d=data)
        self.K = K
        self.D = self.data.shape[1]
        self.dsize = self.data.shape[0]
        self.means = np.zeros((self.K,self.D))
        self.covar = np.zeros((self.K,self.D,self.D))
        self.resp = np.zeros((self.dsize,self.K))
        self.coeff = np.zeros(self.K)

        self.initiation()

    def initiation(self):
        self.coeff = np.ones(self.K)/self.K
        dmax = np.max(self.data,axis=0)
        dmin = np.min(self.data,axis=0)
        for i in range(self.K):
            self.means[i] = np.random.random(2)*(dmax-dmin)+dmin
            self.covar[i] = np.cov(self.data, rowvar=False) / 5

    ######pdf for gaussian#####
    def gaussian(self,x, mean, cov):
        #####just in case the cov are singular########
        covdet = np.linalg.det(cov + np.eye(self.D) * 1e-10)
        covinv = np.linalg.inv(cov + np.eye(self.D) * 1e-10)
        xdiff = x - mean
        prob = np.exp(-0.5 * np.dot(np.dot(xdiff, covinv), xdiff)) / (2*np.pi)**(self.D/2.) / np.sqrt(np.abs(covdet))
        return prob

    #### given mean,cov,pi calculate responsibility###
    def Estep(self):
        for i in range(self.dsize):
            for k in range(self.K):
                self.resp[i][k] = self.coeff[k] * self.gaussian(self.data[i],self.means[k],self.covar[k])

            total_prob = np.sum(self.resp[i])
            self.resp[i] = self.resp[i] / total_prob

    #### given responsibility calculate parameters of GMM#####
    def Mstep(self):
        Nk = np.sum(self.resp,axis=0)
        for k in range(self.K):
            self.means[k]=np.zeros(self.D)
            for i in range(self.dsize):
                self.means[k] += self.resp[i][k] * self.data[i]
            self.means[k] = self.means[k] / Nk[k]

            self.covar[k]=np.zeros((self.D,self.D))
            for i in range(self.dsize):
                self.covar[k] += np.array(self.resp[i][k] * np.matmul(np.matrix(self.data[i]-self.means[k]).T, np.matrix(self.data[i]-self.means[k])))
            self.covar[k] = self.covar[k] / Nk[k]

            self.coeff[k] = Nk[k] / self.dsize

    def EM(self):
        # the initiation has been done when the GMM object is instantiated
        # self.initiation()

        likehood = -8888888888888888
        t = 0
        while True:
            self.Estep()
            self.Mstep()

            #### calculate likehood ####
            #### if it has not been changed, we can stop EM####
            #### the calculation of likehood is very time-consuming so I re-calculate it very 5 EM step######
            if t%5 == 0:
                newlikehood = 0
                for i in range(self.dsize):
                    s = 0
                    for k in range(self.K):
                        s += self.coeff[k] * self.gaussian(self.data[i],self.means[k],self.covar[k])
                    newlikehood += np.log(s)

                if newlikehood - likehood < 0.00001:
                    print(newlikehood)
                    print(t)
                    break
                else:
                    likehood = newlikehood

            if t > 200:
                print("fail")
                break
            t += 1



if __name__ == '__main__':
    print("this file define the EM algorithm to test it, you can use the experiments defined in experiments.py")