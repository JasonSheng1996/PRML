from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
from math import gamma,log,lgamma,exp


def student_t(x, mu, sig, df):
    d = 1.0 * len(x)
    result = gamma(0.5*(df+d)) / gamma(0.5*df) / ((df*np.pi)**(0.5*d)) / (np.linalg.det(sig)**0.5)
    result = result * ((1.+ np.dot(np.dot(x-mu, np.linalg.inv(sig)),x-mu) / df) ** ((-0.5)*(df+d)))
    return result


class Gibbs(object):
    def __init__(self, K, data,maxit):
        self.data = load_data(d=data)
        self.K = K
        self.D = self.data.shape[1]
        self.N = self.data.shape[0]
        self.maxit = maxit

        self.N_k = np.zeros(self.K)
        self.z = np.zeros(self.N,dtype="int")
        self.mu = np.zeros((self.K, self.D))
        self.sig = np.zeros((self.K, self.D, self.D))
        self.Pi = np.zeros(self.K)

        self.mean = np.mean(self.data,axis=0)
        self.S_xbar = np.sum(np.outer(self.data[i]-self.mean,self.data[i]-self.mean) for i in range(self.N))
        ######the way to initiate hyper parameters are based on Machine Learning-A Probabilistic Perspective#####
        self.alpha = 1.
        self.m_0 = self.mean
        self.ka_0 = 0.01
        self.S_0 = np.diagflat(np.diag(self.S_xbar)) / self.N
        # self.S_0 = np.eye(self.D) * 0.1
        self.v_0 = self.D + 2
        ######parameters for each gaussian component########
        self.ka_N = np.zeros(self.K)
        self.m_N = np.zeros((self.K,self.D))
        self.v_N = np.zeros(self.K)
        self.S_N = np.zeros((self.K,self.D,self.D))
        ######these parameters are just use to updata m_N S_N##########
        self.sum_x = np.zeros((self.K,self.D))
        self.outer_x = np.zeros((self.K,self.D,self.D))

        ###parameters for GMM####
        self.means = np.zeros((self.K, self.D))
        self.covar = np.zeros((self.K, self.D, self.D))
        self.coeff = np.zeros(self.K)   ### pi renamed as coeff(mixing coefficient)

    def initiation(self):
        ######randomly assign each x to component k#######
        for i in range(self.N):
            self.z[i] = np.random.randint(self.K)

        for k in range(self.K):
            self.cal_para_k(k)

    def cal_para_k(self,k):
        index_k = np.where(self.z == k)[0]
        x_k = self.data[index_k]
        self.N_k[k] = x_k.shape[0]

        if self.N_k[k] != 0:
            self.sum_x[k] = np.sum(x_k,axis=0)
            self.outer_x[k] = np.sum(np.outer(x_k[i],x_k[i]) for i in range(x_k.shape[0]))

            self.ka_N[k] = self.ka_0 + self.N_k[k]
            self.v_N[k] = self.v_0 + self.N_k[k]
            self.m_N[k] = (self.ka_0*self.m_0 + self.sum_x[k])/self.ka_N[k]
            self.S_N[k] = self.S_0 + self.ka_0*np.outer(self.m_0,self.m_0) - self.ka_N[k]*np.outer(self.m_N[k],self.m_N[k]) + self.outer_x[k]

    def updata_add(self,x,k):
        self.N_k[k] += 1
        self.ka_N[k] += 1
        self.v_N[k] += 1
        self.sum_x[k] = self.sum_x[k] + x
        self.outer_x[k] = self.outer_x[k] + np.outer(x,x)
        self.m_N[k] = (self.ka_0 * self.m_0 + self.sum_x[k]) / self.ka_N[k]
        self.S_N[k] = self.S_0 + self.ka_0 * np.outer(self.m_0, self.m_0) - self.ka_N[k] * np.outer(self.m_N[k],self.m_N[k]) + self.outer_x[k]

    def updata_delete(self, x, k):
        if self.N_k[k] != 1:
            self.N_k[k] -= 1
            self.ka_N[k] -= 1
            self.v_N[k] -= 1
            self.sum_x[k] = self.sum_x[k] - x
            self.outer_x[k] = self.outer_x[k] - np.outer(x, x)
            self.m_N[k] = (self.ka_0 * self.m_0 + self.sum_x[k]) / self.ka_N[k]
            self.S_N[k] = self.S_0 + self.ka_0 * np.outer(self.m_0, self.m_0) - self.ka_N[k] * np.outer(self.m_N[k],self.m_N[k]) + self.outer_x[k]

    #####sampling based on a discrete density function#####
    def basic_sampling(self,p):
        #### normalize first #########
        p = p / np.sum(p)
        r = np.random.random(1)[0]
        for i in range(len(p)):
            if r < p[i]:
                return i
            else:
                r -= p[i]

    ###calculate marginal probability P(Xk)for ####
    def cal_marginal(self,k):
        p = - self.N_k[k] * self.D * 0.5 * log(np.pi)
        p += - self.D * 0.5 * log(self.ka_N[k])
        p += - self.v_N[k] * 0.5 * log(np.linalg.det(self.S_N[k]))

        for d in range(1, self.D + 1):
            p += lgamma(0.5 * (self.v_N[k] + 1 - d))

        return p

    ###calculate the P(zi=k|X,Z,alpha,ka,v,m,S)#####
    ### I implement 2 method to calculate it, both works porperly####
    ####to speed up the calculation, we can ignore some facters which are same in different component#####
    def cal_probability1(self,k,i):
        p1 = self.cal_marginal(k)

        self.z[i] = k
        self.cal_para_k(k=k)
        p2 = self.cal_marginal(k)
        self.z[i] = -1
        self.cal_para_k(k=k)

        p = (self.N_k[k] + self.alpha / self.K)* exp(p2-p1)
        return p

    ###another way calculate the P(zi=k|X,Z,alpha,ka,v,m,S)####
    ### this uses student t distribution#####
    ### this one is faster####
    def cal_probability2(self, k, i):
        term1 = (self.N_k[k] + self.alpha / self.K) / (self.N + self.alpha - 1)
        t_mu = self.m_N[k]
        t_sig = ((self.ka_N[k] + 1) / (self.ka_N[k] * (self.v_N[k] - self.D + 1))) * self.S_N[k]
        t_df = self.v_N[k] - self.D + 1
        term2 = student_t(x=self.data[i], mu=t_mu, sig=t_sig, df=t_df)
        return term1*term2

    def train(self):
        self.initiation()
        for it in range(self.maxit):
            for i in range(self.N):
                zi_old = self.z[i]
                self.z[i] = -1
                self.updata_delete(x=self.data[i],k=zi_old)

                pk = np.zeros(self.K)
                for k in range(self.K):
                    ########you can choose either way to calculate###########
                    # pk[k] = self.cal_probability1(k,i)
                    ######
                    pk[k] = self.cal_probability2(k,i)

                zi_new = self.basic_sampling(pk)
                self.z[i] = zi_new
                self.updata_add(x=self.data[i], k=zi_new)
        ''' Personally,I think classifying x to the component k with max probability may be better than sampling in the
            final iteration. (though I find that in most cases these two strategies give us the same results)'''
        # for i in range(self.N):
        #     zi_old = self.z[i]
        #     self.z[i] = -1
        #     self.updata_delete(x=self.data[i], k=zi_old)
        #
        #     pk = np.zeros(self.K)
        #     for k in range(self.K):
        #         pk[k] = self.cal_probability2(k,i)
        #
        #     zi_new = np.argmax(pk)
        #     self.z[i] = zi_new
        #     self.updata_add(x=self.data[i], k=zi_new)


    ####given z[] we can easily calculate the parameters for GMM####
    def recal_gmm(self):
        for k in range(self.K):
            index_k = np.where(self.z == k)[0]
            x_k = self.data[index_k]
            self.means[k] = np.mean(x_k,axis=0)
            self.covar[k] = np.sum(np.outer(x_k[i]-self.means[k],x_k[i]-self.means[k])for i in range(x_k.shape[0])) / x_k.shape[0]


if __name__=="__main__":
    print("this file define the Gibbs algorithm to test it, you can use the experiments defined in experiments.py")