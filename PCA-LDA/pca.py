from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


class PCA(object):
    def __init__(self, dimension, preprocess=''):
        self.x_train, self.y_train, self.x_test, self.y_test = load_data()

        # set data to [0,1] when using svm as suggested by libsvm
        # it does not truly change the distribution of data
        if preprocess == 'libsvm':
            self.x_train = self.x_train/255.0
            self.x_test = self.x_test/255.0

        self.dimension = dimension

        self.train_mean = np.mean(self.x_train, axis=0)
        self.test_mean = np.mean(self.x_test, axis=0)

        self.pdirect = self.principal_direction()

        self.pc_train = self.project(self.x_train)
        self.pc_test = self.project(self.x_test)
        print("principal component setted")

    def principal_direction(self):
        cov = np.cov(self.x_test-self.train_mean, rowvar=0)
        self.eig, self.eigvect = np.linalg.eig(cov)
        d_eig_index = np.argsort(self.eig)[-1:-(self.dimension + 1):-1]
        d_eig = self.eig[d_eig_index]
        d_eigVect = self.eigvect[:, d_eig_index]
        return d_eigVect

    # reduce data's dimension or extract features by project data to principal subspace
    def project(self, x):
        return np.matmul(x, self.pdirect)

    def __calc_distance(self, a, b):
        diff = a - b
        dis = np.sum(diff * diff, axis=1)
        return dis

    def nn_classfication(self):
        err = 0
        right = 0
        result = np.zeros(self.y_test.shape)
        for i in range(self.pc_test.shape[0]):
            dis = self.__calc_distance(np.tile(self.pc_test[i], (self.pc_train.shape[0], 1)), self.pc_train)
            nn = np.argmin(dis,axis=0)
            tag_nn = self.y_train[nn]
            result[i] = tag_nn
            if result[i] != self.y_test[i]:
                print("w: ",i)
            else:
                right+=1

        err = np.sum(result != self.y_test)
        print("total/err/right",self.y_test.shape[0],err,right)
        return err / self.y_test.shape[0]


# experiments in the report
def exp_pca(exp="vs2"):
    if exp == "vs-eig":
        pca = PCA(dimension=3)
        pc = pca.pdirect.real.T
        for i in range(3):
            img = pc[i].reshape(28,28)
            plt.imshow(img)
            plt.savefig('eign'+str(i)+".png")

    if exp == "vs2":
        pca = PCA(dimension=2)
        pc = pca.pc_train.real
        plt.figure()
        plt.scatter(pc.T[0],pc.T[1],s=3)
        # plt.show()
        plt.title('PCA training set projected to 2 dimension')
        plt.savefig('pca2.png')

    if exp == "vs3":
        pca = PCA(dimension=3)
        pc = pca.pc_train.real
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc.T[0],pc.T[1],pc.T[2],s=3)
        # plt.show()
        plt.title('PCA training set projected to 3 dimension')
        plt.savefig('pca3.png')

    if exp == "cls":
        dimension = np.array([40, 80, 200])
        for d in dimension:
            pca = PCA(dimension=d)
            err = pca.nn_classfication()
            print("dimension:",d)
            print(1-err)

    if exp == "energy":
        pca = PCA(dimension = 1)
        eig = pca.eig[np.argsort(-pca.eig)]
        total_energy =  np.sum(eig)
        sum = 0.0
        d = 0
        for i in range(len(eig)):
            sum += eig[i]
            if sum > 0.95 * total_energy:
                print(i)
                d = i+1
                break

        pca = PCA(dimension=d)
        err = pca.nn_classfication()
        print("dimension:", d)
        print(1 - err)


if __name__ == '__main__' :
    exp_pca("cls")