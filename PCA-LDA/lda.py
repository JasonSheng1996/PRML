from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from time import clock


class LDA(object):
    def __init__(self, dimension):
        self.x_train, self.y_train, self.x_test, self.y_test = load_data()
        self.feature_size = self.x_train.shape[1]
        self.dimension = dimension

        self.train_mean = np.mean(self.x_train, axis=0)
        self.cls = 10
        self.cls_num = np.zeros(self.cls)                                   # num of 10 different class
        self.cls_mean = np.zeros((self.cls,self.feature_size))              # mean of 10 different class
        for i in range(self.x_train.shape[0]):
            self.cls_num[self.y_train[i]] += 1
            self.cls_mean[self.y_train[i]] += self.x_train[i]
        for i in range(self.cls):
            self.cls_mean[i] = self.cls_mean[i] / self.cls_num[i]

        self.s_w = self.within_class_covariance()
        self.s_b = self.between_class_covariance()

        self.disc_space = self.discriminative_subspace()
        self.pj_train = self.project(self.x_train)
        self.pj_test = self.project(self.x_test)

        print("subspace for linear discrimination setted")

    def within_class_covariance(self):
        s = np.zeros((self.feature_size, self.feature_size))
        for i in range(self.x_train.shape[0]):
            d = np.asmatrix(self.x_train[i]-self.cls_mean[self.y_train[i]])
            s += np.matmul(d.T, d)
            if i % 1000 == 0:
                print(i)
        return s

    def between_class_covariance(self):
        s = np.zeros((self.feature_size, self.feature_size))
        for i in range(self.cls):
            d = np.asmatrix(self.cls_mean[i]-self.train_mean)
            s += self.cls_num[i] * np.matmul(d.T, d)
        return s

    def discriminative_subspace(self):
        # S_w may be singular because some pixels in pictures are always 0
        # we use pseudo_inverse matrix instead of inverse matrix to calculate the subspace
        eig, eigvect = np.linalg.eig(np.matmul(np.linalg.pinv(self.s_w),self.s_b))
        d_eig_index = np.argsort(eig)[-1:-(self.dimension + 1):-1]
        d_eig = eig[d_eig_index]
        d_eigVect = eigvect[:, d_eig_index]
        return d_eigVect

    def project(self,x):
        return np.matmul(x, self.disc_space)

    def __calc_distance(self, a, b):
        diff = a - b
        dis = np.sum(diff * diff, axis=1)
        return dis

    def nn_classification(self):
        err = 0
        right = 0
        result = np.zeros(self.y_test.shape)
        for i in range(self.pj_test.shape[0]):
            dis = self.__calc_distance(np.tile(self.pj_test[i], (self.pj_train.shape[0], 1)), self.pj_train)
            nn = np.argmin(dis, axis=0)
            tag_nn = self.y_train[nn]
            result[i] = tag_nn
            if result[i] == self.y_test[i]:
                right += 1
            # else:
            #     print("w: ", i)
            if i % 1000 == 0:
                print(i)

        err = np.sum(result != self.y_test)
        print("total/err/right", self.y_test.shape[0], err, right)
        return err / self.y_test.shape[0]

    # a much faster way to classify is using the closest means instead of closest data
    def nmean_classification(self):
        err = 0
        right = 0
        result = np.zeros(self.y_test.shape)
        pj_mean = self.project(self.cls_mean)
        for i in range(self.pj_test.shape[0]):
            dis = self.__calc_distance(np.tile(self.pj_test[i], (pj_mean.shape[0], 1)), pj_mean)
            nmean = np.argmin(dis, axis=0)
            result[i] = nmean
            if result[i] == self.y_test[i]:
                right += 1
            else:
                print("w: ", i)
            if i % 1000 == 0:
                print(i)

        err = np.sum(result != self.y_test)
        print("total/err/right", self.y_test.shape[0], err, right)
        return err / self.y_test.shape[0]


def exp_lda(exp="vs2"):
    if exp == "vs-eig":
        lda = LDA(dimension=3)
        ds = lda.disc_space.real.T
        for i in range(3):
            img = ds[i].reshape(28, 28)
            plt.imshow(img)
            plt.savefig('eign' + str(i) + ".png")

    if exp == "vs2":
        lda = LDA(dimension=2)
        pj = lda.pj_train.real
        plt.figure()
        plt.scatter(pj.T[0], pj.T[1], s=2)
        plt.title('LDA training set projected to 2 dimension')
        plt.savefig('lda2.png')

    if exp == "vs3":
        lda = LDA(dimension=3)
        pj = lda.pj_train.real
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pj.T[0],pj.T[1],pj.T[2],s=5)
        plt.title('LDA training set projected to 3 dimension')
        plt.savefig('lda3.png')

    if exp == "cls":
        dimension = np.array([2,3,9])
        for d in dimension:
            lda = LDA(dimension = d)
            err = lda.nn_classification()
            print("dimension:",d)
            print(1-err)

    if exp == "cls-m":
        dimension = np.array([2,3,9])
        for d in dimension:
            lda = LDA(dimension = d)
            err = lda.nmean_classification()
            print("dimension:",d)
            print(1-err)

if __name__ == '__main__' :
    exp_lda("cls-m")