# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from dataloader import *
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# import cPickle as pkl


class GenerativeClassifier(object):
    """
        meta class for generative classifiers.
    """

    def __init__(self, preprocessing="", shuffle_train=False, shuffle_test=False):
        self.x_train, self.y_train, self.x_test, self.y_test, self.feature_size = load_data(t=preprocessing,
                                                                                            verbose=False)
        self.train_size = len(self.y_train)
        self.test_size = len(self.x_test)

        # TODO: reset the prior distribution
        print("spam : train", np.count_nonzero(self.y_train), len(self.y_train))
        self.pi_1 = float(np.count_nonzero(self.y_train)) / float(len(self.y_train))

        # split the x_train
        xytrain = np.hstack((self.x_train, self.y_train))
        xytrain_sort_index = np.lexsort(xytrain.T)
        xtrain_sorted = np.delete(xytrain[xytrain_sort_index], 57, 1)
        self.xtrain_good = xtrain_sorted[0:(len(self.y_train) - np.count_nonzero(self.y_train))]
        self.xtrain_spam = xtrain_sorted[(len(self.y_train) - np.count_nonzero(self.y_train)):]

        if self.xtrain_spam.shape[0] != np.count_nonzero(self.y_train):
            print("ERROR")

        if shuffle_train:
            idx = np.arange(self.train_size)
            np.random.shuffle(idx)
            self.x_train = [self.x_train[idx[i]] for i in range(self.train_size)]
            self.y_train = [self.y_train[idx[i]] for i in range(self.train_size)]

        if shuffle_test:
            idx = np.arange(self.test_size)
            np.random.shuffle(idx)
            self.x_test = [self.x_test[idx[i]] for i in range(self.test_size)]
            self.y_test = [self.y_test[idx[i]] for i in range(self.test_size)]


"""
    this method using Beta Distribution as the conjuncture prior.
"""


class BetaNaiveBayesClassifier(GenerativeClassifier):
    def __init__(self, alpha=500, preprocessing="bi"):
        super(BetaNaiveBayesClassifier, self).__init__(preprocessing=preprocessing)
        self.alpha = alpha
        self.beta = np.full((2, self.feature_size), self.alpha, dtype=float)
        self.count = [0, 0]

    def train(self):
        # TODO
        # calculate num of x=1
        self.bb_m = np.vstack((np.count_nonzero(self.xtrain_good, axis=0), np.count_nonzero(self.xtrain_spam, axis=0)))
        self.bb_n = np.vstack((np.full((1,self.feature_size),self.xtrain_good.shape[0],dtype=float),np.full((1,self.feature_size),self.xtrain_spam.shape[0],dtype=float)))
        self.bb_u = np.divide(self.bb_m+self.alpha-1, self.bb_n+self.beta+self.alpha-2)
        # print(self.bb_u)

    def loglikelihood(self, x, cls):
        loglh = 0.0
        for i in range(len(x)):
            if x[i] == 1:
                loglh = loglh + np.log(self.bb_u[cls][i])
            elif x[i] == 0:
                loglh = loglh + np.log(1-self.bb_u[cls][i])
        return loglh

    def test(self,case=""):
        # TODO
        num_right = 0
        num_g2s = 0
        num_s2g = 0
        if case == "test":
            # print("testcase=test", self.x_test.shape[0])
            for i in range(self.x_test.shape[0]):
                logpg = self.loglikelihood(self.x_test[i], 0) + np.log(1 - self.pi_1)
                logps = self.loglikelihood(self.x_test[i], 1) + np.log(self.pi_1)
                if logpg > logps:  # classified as good
                    if self.y_test[i][0] == 0:
                        num_right += 1
                    else:
                        num_s2g += 1
                else:
                    if self.y_test[i][0] == 1:
                        num_right += 1
                    else:
                        num_g2s += 1
            print("BetaNaiveBayesClassifier: alpha",self.alpha)
            print('right:', num_right, " good to spam:", num_g2s, " spam to good:", num_s2g)
            return self.x_test.shape[0], num_right, num_g2s, num_s2g
        elif case == "train":
            # print("testcase=train", self.x_train.shape[0])
            for i in range(self.x_train.shape[0]):
                logpg = self.loglikelihood(self.x_train[i], 0) + np.log(1 - self.pi_1)
                logps = self.loglikelihood(self.x_train[i], 1) + np.log(self.pi_1)
                if logpg > logps:  # classified as good
                    if self.y_train[i][0] == 0:
                        num_right += 1
                    else:
                        num_s2g += 1
                else:
                    if self.y_train[i][0] == 1:
                        num_right += 1
                    else:
                        num_g2s += 1
            # print("BetaNaiveBayesClassifier: alpha",self.alpha)
            # print('right:', num_right, " good to spam:", num_g2s, " spam to good:", num_s2g)
            return self.x_train.shape[0], num_right, num_g2s, num_s2g

    def test_update(self):                     #update parameter after every test case
        # TODO
        num_right = 0
        num_g2s = 0
        num_s2g = 0
        # print("testcase=test", self.x_test.shape[0])
        for i in range(self.x_test.shape[0]):
            logpg = self.loglikelihood(self.x_test[i], 0) + np.log(1 - self.pi_1)
            logps = self.loglikelihood(self.x_test[i], 1) + np.log(self.pi_1)
            if logpg > logps:  # classified as good
                if self.y_test[i][0] == 0:
                    num_right += 1
                else:
                    num_s2g += 1
            else:
                if self.y_test[i][0] == 1:
                    num_right += 1
                else:
                        num_g2s += 1
            for j in range(self.feature_size):
                self.bb_m[self.y_test[i][0]][j] += self.x_test[i][j]
                self.bb_n[self.y_test[i][0]][j] += 1
                self.bb_u = np.divide(self.bb_m + self.alpha - 1, self.bb_n + self.beta + self.alpha - 2)

        print("BetaNaiveBayesClassifier: alpha",self.alpha)
        print('right:', num_right, " good to spam:", num_g2s, " spam to good:", num_s2g)
        return self.x_test.shape[0], num_right, num_g2s, num_s2g


class GaussianNaiveBayesClassifier(GenerativeClassifier):
    def __init__(self, preprocessing=""):
        super(GaussianNaiveBayesClassifier, self).__init__(preprocessing=preprocessing)
        self.ML = None

    def train(self):
        # TODO
        self.gmean = np.vstack((np.mean(self.xtrain_good, axis=0), np.mean(self.xtrain_spam, axis=0)))
        self.gstd = np.vstack((np.std(self.xtrain_good, axis=0), np.std(self.xtrain_spam, axis=0)))
        # print(self.gstd[0],self.gstd[1])

    def loglikelihood(self, x, cls):
        loglh = 0.0
        for i in range(len(x)):
            if self.gstd[cls][i] == 0:
                loglh = loglh + ss.norm.logpdf(x[i], loc=self.gmean[cls][i], scale=0.001)
            else:
                loglh = loglh + ss.norm.logpdf(x[i], loc=self.gmean[cls][i], scale=self.gstd[cls][i])
        return loglh

    def test(self, case=""):
        # TODO
        num_right = 0
        num_g2s = 0
        num_s2g = 0
        if case == "test":
            print("testcase", self.x_test.shape[0])
            for i in range(self.x_test.shape[0]):
                logpg = self.loglikelihood(self.x_test[i], 0) + np.log(1 - self.pi_1)
                logps = self.loglikelihood(self.x_test[i], 1) + np.log(self.pi_1)
                if logpg > logps:  # classified as good
                    if self.y_test[i][0] == 0:
                        num_right += 1
                    else:
                        num_s2g += 1
                else:
                    if self.y_test[i][0] == 1:
                        num_right += 1
                    else:
                        num_g2s += 1
            print("GaussianNaiveBayesClassifier:")
            print('right:', num_right, " good to spam:", num_g2s, " spam to good:", num_s2g)
            return self.x_test.shape[0], num_right, num_g2s, num_s2g
        elif case == "train":
            print("testcase", self.x_train.shape[0])
            for i in range(self.x_train.shape[0]):
                logpg = self.loglikelihood(self.x_train[i], 0) + np.log(1 - self.pi_1)
                logps = self.loglikelihood(self.x_train[i], 1) + np.log(self.pi_1)
                if logpg > logps:  # classified as good
                    if self.y_train[i][0] == 0:
                        num_right += 1
                    else:
                        num_s2g += 1
                else:
                    if self.y_train[i][0] == 1:
                        num_right += 1
                    else:
                        num_g2s += 1
            print("GaussianNaiveBayesClassifier:")
            print('right:', num_right, " good to spam:", num_g2s, " spam to good:", num_s2g)
            return self.x_train.shape[0], num_right, num_g2s, num_s2g


if __name__ == '__main__':
    # cf = GaussianNaiveBayesClassifier(preprocessing="bimean")
    cf = BetaNaiveBayesClassifier(alpha=10,preprocessing="bi")
    cf.train()
    cf.test(case="test")
    cf.test_update()

    # alpha = np.linspace(0., 100., 201)
    # alpha = np.array([1.,10.,100.])
    # er_train = np.zeros(len(alpha),dtype=float)
    # er_test = np.zeros(len(alpha),dtype=float)
    #
    # for i in range(0,len(alpha)):
    #     cf = BetaNaiveBayesClassifier(alpha=alpha[i], preprocessing="bi")
    #     cf.train()
    #     tsize,nr,gs,sg = cf.test(case="train")
    #     er_train[i] = 1.0-(float(nr)/float(tsize))
    #     tsize, nr,gs,sg = cf.test(case="test")
    #     er_test[i] = 1.0 - (float(nr) / float(tsize))
    #
    #     print("alpha: %3.2f train_error %3.3f test_error %3.3f"%(alpha[i],er_train[i],er_test[i]))
    #
    # plt.plot(alpha, er_train, 'b-', label='train error ')
    # plt.plot(alpha, er_test, 'g-', label='test error ')
    # plt.xlim((0, 100))
    # plt.ylim((0.1, 0.16))
    # plt.xlabel('alpha')
    # plt.ylabel('error rate')
    # plt.title('scatter plot')
    # plt.legend()
    # plt.savefig('alphaerror.png')