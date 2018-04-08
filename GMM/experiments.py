from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
from kmeans import *
from EM import *
from Gibbs import *


def em_exp(case=""):
    if case=="OF":
        gmm = EM_GMM(K=2, data="OF")
        pt = gmm.data.T
        fig, ax = plt.subplots()
        gmm.EM()
        print(gmm.coeff)
        print(gmm.means)
        print(gmm.covar)
        for i in range(2):
            ##### to visaulize a guassian we extract the Eigenvector of cov as the direct of it#####
            v, w = np.linalg.eigh(gmm.covar[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi
            ell = matplotlib.patches.Ellipse(gmm.means[i], 1.5*v[0], 1.5*v[1], 180. + angle,facecolor="none",edgecolor="r")
            fig.gca().add_artist(ell)
        ax.set_xlim(1, 6)
        ax.set_ylim(40, 100)
        plt.scatter(pt[0], pt[1], s=15)
        plt.title('EM on Old Faithful')
        plt.savefig('exp-results/em_OF.png')

    if case == "rand4":
        KK = 2
        gmm = EM_GMM(K=KK, data="rand4")
        pt = gmm.data.T
        fig, ax = plt.subplots()

        plt.scatter(gmm.means.T[0], gmm.means.T[1], s=35,marker='x')

        gmm.EM()
        print(gmm.coeff)
        print(gmm.means)
        print(gmm.covar)
        for i in range(KK):
            v, w = np.linalg.eigh(gmm.covar[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = matplotlib.patches.Ellipse(gmm.means[i], v[0], v[1], 180. + angle, facecolor="none",
                                             edgecolor="r")
            fig.gca().add_artist(ell)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        # plt.scatter(pt[0], pt[1], s=15)

        ##### use responsibility to classify#####
        cls =  np.argmax(gmm.resp, axis=1)
        # print(cls)

        clr = ["r","g","gold","brown","black"]
        for k in range(KK):
            index_k = np.where(cls == k)[0]
            x_k = gmm.data[index_k].T
            plt.scatter(x_k[0], x_k[1], s=15,c =clr[k] )

        plt.title('em gmm')
        plt.show()


    if case=="kmeans":
        KK = 4
        gmm = EM_GMM(K=KK, data="40")
        gmm.means = kmeans(data=gmm.data,K=KK)
        pt = gmm.data.T
        fig, ax = plt.subplots()
        gmm.EM()
        print(gmm.coeff)
        print(gmm.means)
        print(gmm.covar)
        for i in range(KK):
            v, w = np.linalg.eigh(gmm.covar[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = matplotlib.patches.Ellipse(gmm.means[i], v[0], v[1], 180. + angle, facecolor="none",
                                             edgecolor="r")
            fig.gca().add_artist(ell)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        # plt.scatter(pt[0], pt[1], s=15)

        ##### use responsibility to classify#####
        cls = np.argmax(gmm.resp, axis=1)
        # print(cls)

        clr = ["r", "g", "gold", "brown", "black"]
        for k in range(KK):
            index_k = np.where(cls == k)[0]
            x_k = gmm.data[index_k].T
            plt.scatter(x_k[0], x_k[1], s=15, c=clr[k])

        plt.title('Kmeans-EM')
        plt.show()


def gibbs_exp(case=""):
    if case == "rand4":
        KK =6
        g = Gibbs(K=KK, data="rand4", maxit=50)
        g.train()
        g.recal_gmm()
        print(g.m_N)
        print(g.N_k)
        print(g.z)
        print(g.covar)

        fig, ax = plt.subplots()
        for i in range(KK):
            v, w = np.linalg.eigh(g.covar[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi
            ell = matplotlib.patches.Ellipse(g.means[i], v[0], v[1], 180. + angle, facecolor="none", edgecolor="b")
            fig.gca().add_artist(ell)

        # pt = g.data.T
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        clr = ["r","g","gold","brown"]
        for k in range(KK):
            index_k = np.where(g.z == k)[0]
            x_k = g.data[index_k].T
            plt.scatter(x_k[0], x_k[1], s=15,c =clr[k] )

        plt.title('Gibbs Gmm')
        plt.show()

    if case == "OF":
        g = Gibbs(K=2, data="OF", maxit=100)
        g.train()
        g.recal_gmm()
        print(g.m_N)
        print(g.N_k)
        print(g.covar)

        fig, ax = plt.subplots()
        for i in range(2):
            v, w = np.linalg.eigh(g.covar[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = matplotlib.patches.Ellipse(g.means[i], v[0], v[1], 180. + angle, facecolor="none", edgecolor="b")
            fig.gca().add_artist(ell)

        pt = g.data.T
        ax.set_xlim(1, 5.5)
        ax.set_ylim(40, 100)
        clr = ["r", "g", "gold"]
        for k in range(2):
            index_k = np.where(g.z == k)[0]
            x_k = g.data[index_k].T
            plt.scatter(x_k[0], x_k[1], s=15, c=clr[k])
        plt.title('Gibbs ')
        plt.show()


if __name__ == '__main__':
    em_exp(case="kmeans")
    # gibbs_exp(case="rand4")