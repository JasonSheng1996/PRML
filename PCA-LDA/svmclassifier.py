from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from pca import *
from libsvm.python.svmutil import *
from libsvm.python.svm import *


# expriments using PCA and SVM
def exp_svm(exp):
    if exp == "linear":
        c = [0.01, 0.1, 1, 10]
        pca_r = PCA(dimension=1, preprocess='libsvm')
        x_train = pca_r.x_train.tolist()
        y_train = pca_r.y_train.tolist()
        x_test = pca_r.x_test.tolist()
        y_test = pca_r.y_test.tolist()
        for i in range(len(c)):
            m = svm_train(y_train, x_train, '-c '+str(c[i])+' -t 0 -q')
            print("raw:")
            print("c=",c[i])
            p_label, p_acc, p_val=svm_predict(y_test, x_test, m)

        d = [40, 80, 200]
        for j in range(len(d)):
            pca = PCA(dimension=d[j], preprocess='libsvm')
            x_train = pca.pc_train.real.tolist()
            y_train = pca.y_train.tolist()
            x_test = pca.pc_test.real.tolist()
            y_test = pca.y_test.tolist()
            for i in range(len(c)):
                m = svm_train(y_train, x_train, '-c ' + str(c[i]) + ' -t 0 -q')
                print("dimension =",d[j])
                print("c =",c[i])
                svm_predict(y_test, x_test, m)

    if exp == "kernel":
        c = [ 1,10,100,1000]
        g = [0.01,0.025,0.05,0.1]
        pca = PCA(dimension=40, preprocess='libsvm')
        x_train = pca.pc_train.real.tolist()
        y_train = pca.y_train.tolist()
        x_test = pca.pc_test.real.tolist()
        y_test = pca.y_test.tolist()
        for i in range(len(c)):
            for j in range(len(g)):
                m = svm_train(y_train, x_train, '-c ' + str(c[i]) + ' -g '+str(g[j])+' -t 2 -q')
                print("c =", c[i])
                print("g =", g[j])
                svm_predict(y_test, x_test, m)


if __name__ == "__main__":
    exp_svm(exp='kernel')
