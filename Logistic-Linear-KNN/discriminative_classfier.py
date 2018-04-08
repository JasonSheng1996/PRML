from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
from time import clock


class DiscriminativeClassifier(object):
    def __init__(self, preprocessing="", bias=False):
        self.x_train, self.y_train, self.x_test, self.y_test, self.feature_size = \
            load_data(t=preprocessing, bias=bias)
        self.train_size = len(self.y_train)
        self.test_size = len(self.x_test)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp((-1) * x))


class LogisticRegression(DiscriminativeClassifier):
    def __init__(self, l2_lambda=1, preprocessing="", eta=1e-4, max_epoch=500, l2_on=True):
        """
        :param l2_lamdba: l2 norm penalty
        :param preprocessing: preprocessing method
        :param eta: learning rate (step size)
        :param max_epoch: how many epochs are you going to train the regression (optional)
        :param l2_on: use l2 or not
        """
        super(LogisticRegression, self).__init__(preprocessing=preprocessing, bias=True)
        self.L2_lambda = l2_lambda
        self.eta = eta
        self.max_epoch = max_epoch
        self.L2_on = l2_on

        # initial weight
        self.weight = np.zeros((self.feature_size + 1, 1))
        # mask is to filter out the weight and discard the bias
        self.mask = np.ones_like(self.weight)
        self.mask[self.feature_size][0] = 0

        # print(self.x_train.shape)
        # print(self.y_train.shape)

    def calc_grad_hess(self, W, X, YT):
        # W: weight matrix
        # X: input
        # YT: ground true/tag of data
        # Y: y calculated by old weight
        Y = sigmoid(np.matmul(X, W))
        grad = np.matmul(X.T, Y - YT)
        if self.L2_on == True:
            grad_regularization = np.copy(self.weight)
            grad_regularization[0][0] = 0
            grad += self.L2_lambda * (grad_regularization)

        R = np.diagflat(Y * (1 - Y), 0)
        Hess = np.matmul(np.matmul(X.T, R), X)
        if self.L2_on == True:
            hess_regularization = np.eye(self.feature_size + 1)
            hess_regularization[0][0] = 0
            Hess += self.L2_lambda * hess_regularization

        return grad, Hess

    def calc_grad(self, W, X, YT):
        # W: weight matrix
        # X: input
        # YT: ground true/tag of data
        # Y: y calculated by old weight
        Y = sigmoid(np.matmul(X, W))
        grad = np.matmul(X.T, Y - YT)
        if self.L2_on == True:
            grad_regularization = np.copy(self.weight)
            grad_regularization[0][0] = 0
            grad += self.L2_lambda * grad_regularization

        return grad

    def train(self, method="Newton",eps=1e-4):
        phi0 = np.ones((self.x_train.shape[0], 1))
        Phi = np.hstack((phi0, self.x_train))
        epoch = 0
        if method == "Newton":                                       # update weight with Newton's Method
            while True:
                grad, hess = self.calc_grad_hess(self.weight, Phi, self.y_train)
                update = np.matmul(pinv(hess), grad)
                self.weight = self.weight - update

                if norm(update) < self.eta * 0.1:                   #in Newton method, let eta always be 1e-4
                    print("epoch", epoch)
                    break
                epoch += 1
                # print ("epoch", epoch)
                if epoch > self.max_epoch:
                    break
        elif method == "GD":                                            # update weight with Gradient Decrease
            while True:
                grad = self.calc_grad(self.weight, Phi, self.y_train)
                update = self.eta * grad
                self.weight = self.weight - update

                # if norm(update) < self.eta * 0.1:
                if norm(update) < 0.0005 :
                    print(epoch)
                    break
                epoch += 1
                # print ("epoch", epoch)
                if epoch > self.max_epoch:
                    print("fail to converge in max_epoch")
                    break
        else:
            print("no such method to train parameters")

    def test(self, case="test"):
        err = 0
        if case == "test":
            # print("detect spam in test data set")
            phi0 = np.ones((self.x_test.shape[0], 1))
            Phi = np.hstack((phi0, self.x_test))
            predict = (np.matmul(Phi, self.weight) > 0.0)
            for i in range(self.y_test.shape[0]):
                if predict[i][0] != self.y_test[i][0]:
                    err += 1
            return err / self.y_test.shape[0]

        if case == "train":
            # print("detect spam in training data set")
            phi0 = np.ones((self.x_train.shape[0], 1))
            Phi = np.hstack((phi0, self.x_train))
            predict = (np.matmul(Phi, self.weight) > 0.0)
            for i in range(self.y_train.shape[0]):
                if predict[i][0] != self.y_train[i][0]:
                    err += 1
            return err / self.y_train.shape[0]


class KNNClassifier(DiscriminativeClassifier):
    def __init__(self, preprocessing="", K=5):
        """

        :param preprocessing: preprocessing method
        :param K: how much neighbours you want
        """
        super(KNNClassifier, self).__init__(preprocessing=preprocessing)
        self.binary = (preprocessing == "binary")
        self.K = K

    def __calc_distance(self, a, b):
        if self.binary:                                                            #Hamming
            return np.sum(a != b, axis=1)
        else:                                                                      #Euclidean
            diff = a - b
            dis = np.sum(diff * diff, axis=1)
            return dis

    def train(self):                                            #No need for trainning
        return

    def test(self, case="test"):
        err = 0
        err_s2g = 0
        err_g2s = 0
        if case == "test":
            result = np.zeros(self.y_test.shape)
            for i in range(self.x_test.shape[0]):
                dis = self.__calc_distance(np.tile(self.x_test[i], (self.x_train.shape[0], 1)), self.x_train)
                knn = np.argsort(dis, axis=0)[:self.K]
                tag_knn = self.y_train[knn]
                num_spam_knn = np.sum(tag_knn)
                if num_spam_knn >= (self.K / 2):
                    result[i][0] = 1
            err = np.sum(result != self.y_test)
            # for i in range(self.y_test.shape[0]):
            #     if result[i][0] == 0 and self.y_test[i][0] == 1:
            #         err_s2g += 1
            #     if result[i][0] == 1 and self.y_test[i][0] == 0:
            #         err_g2s += 1
            # print(err_s2g, err_g2s)
            return err / self.y_test.shape[0]
        if case == "train":
            result = np.zeros(self.y_train.shape)
            for i in range(self.x_train.shape[0]):
                dis = self.__calc_distance(np.tile(self.x_train[i], (self.x_train.shape[0], 1)), self.x_train)
                knn = np.argsort(dis, axis=0)[:self.K]
                tag_knn = self.y_train[knn]
                num_spam_knn = np.sum(tag_knn)
                if num_spam_knn >= (self.K / 2):
                    result[i][0] = 1
                # if result[i][0] != self.y_train[i][0]:
                #     print("wrong case:",i,self.x_train[i],self.y_train[i])
                #     print("wrong NN",knn[0],self.x_train[knn[0]],self.y_train[knn[0]])

            err = np.sum(result != self.y_train)
            # for i in range(self.y_train.shape[0]):
            #     if result[i][0] == 0 and self.y_train[i][0] == 1:
            #         err_s2g += 1
            #     if result[i][0] == 1 and self.y_train[i][0] == 0:
            #         err_g2s += 1
            # print(err_s2g, err_g2s)
        return err / self.y_train.shape[0]


class LinearRegression(DiscriminativeClassifier):
    def __init__(self, l2_lambda=1, preprocessing="", eta=1e-5, mom =0.5, l2_on=True):
        super(LinearRegression, self).__init__(preprocessing=preprocessing, bias=True)
        self.L2_lambda = l2_lambda
        self.eta = eta
        self.L2_on = l2_on
        self.mom = mom

        # initial weight
        self.weight = np.zeros((self.feature_size + 1, 1))
        self.momentum = np.zeros((self.feature_size + 1, 1))

    def train(self):                                                    #train paras with least square
        phi0 = np.ones((self.x_train.shape[0], 1))
        Phi = np.hstack((phi0, self.x_train))
        if self.L2_on == True:
            l2mask =self.L2_lambda * np.eye(self.weight.shape[0])
            l2mask[0][0]
            self.weight = np.matmul(
                np.matmul(pinv((self.L2_lambda * np.eye(self.weight.shape[0]) + np.matmul(Phi.T, Phi))), Phi.T),
                self.y_train)
        # print(self.weight)

    def sgd(self):                                                           #train paras with sgd momentum
        ###shuffle the training data set
        idx = np.arange(self.train_size)
        np.random.shuffle(idx)
        self.x_train = self.x_train[idx]
        self.y_train = self.y_train[idx]

        phi0 = np.ones((self.x_train.shape[0], 1))
        Phi = np.hstack((phi0, self.x_train))
        for i in range(Phi.shape[0]):
            if i > 200:                                              #after 200 epochs,raise the importance of momentum in updating
                self.mom = 0.9
            phi_n = Phi[i].reshape(Phi.shape[1],1)
            grad_regular = self.L2_lambda * self.weight
            grad_regular[0][0] = 0
            grad = (np.matmul(self.weight.T,phi_n)-self.y_train[i][0])*phi_n + grad_regular
            self.momentum = self.mom * self.momentum + self.eta * grad
            self.weight = self.weight - self.momentum
            # self.eta *= 0.5                                          # learning rate step decay when no pre-processing

    def recall_precision_test(self):
        phi0 = np.ones((self.x_test.shape[0], 1))
        Phi = np.hstack((phi0, self.x_test))
        predict = (np.matmul(Phi, self.weight) > 0.5)
        err=0
        err_s2g=0
        err_g2s=0
        right_s=0
        right_g=0
        print(self.y_test.shape[0],np.sum(self.y_test,axis=0))
        for i in range(self.y_test.shape[0]):
            if predict[i]==0 and self.y_test[i][0]==1:
                err_s2g += 1
            elif predict[i]==1 and self.y_test[i][0]==0:
                err_g2s +=1
            elif predict[i]==1 and self.y_test[i][0]==1:
                right_s +=1
            elif predict[i]==0 and self.y_test[i][0]==0:
                right_g +=1

        print(err_s2g,err_g2s,right_s,right_g)
        if (right_s + err_g2s) == 0:
            precision = 1
        else:
            precision = right_s / (right_s + err_g2s)
        recall = right_s / (right_s + err_s2g)
        return precision,recall

    def test(self, case="test"):
        err = 0
        if case == "test":
            # print("detect spam in test data set")
            phi0 = np.ones((self.x_test.shape[0], 1))
            Phi = np.hstack((phi0, self.x_test))
            predict = (np.matmul(Phi, self.weight) > 0.5)
            for i in range(self.y_test.shape[0]):
                if predict[i][0] != self.y_test[i][0]:
                    err += 1
            return err / self.y_test.shape[0]

        if case == "train":
            # print("detect spam in training data set")
            phi0 = np.ones((self.x_train.shape[0], 1))
            Phi = np.hstack((phi0, self.x_train))
            predict = (np.matmul(Phi, self.weight) > 0.5)
            for i in range(self.y_train.shape[0]):
                if predict[i][0] != self.y_train[i][0]:
                    err += 1
            return err / self.y_train.shape[0]


def experiment_logistic(exp="lam"):
    if exp == "lam":
        lam = np.hstack((np.arange(1, 10), np.arange(10, 105, 5)))
        er_train = np.zeros(len(lam), dtype=float)
        er_test = np.zeros(len(lam), dtype=float)
        for i in range(0, len(lam)):
            lr = LogisticRegression(l2_lambda=lam[i], preprocessing="", eta=1e-4, max_epoch=50, l2_on=True)
            lr.train()
            er_train[i] = lr.test(case="train")
            er_test[i] = lr.test(case="test")
        plt.plot(lam, er_train, 'b-', label='train error ')
        plt.plot(lam, er_test, 'g-', label='test error ')
        plt.xlim((0, 100))
        plt.ylim((0.05, 0.15))
        plt.xlabel('lambda for L2 regularization')
        plt.ylabel('error rate')
        plt.title('lambda vs error rate in Logistic')
        plt.legend()
        plt.savefig('Logistic-lam.png')

    if exp == "pre":
        lam = np.array([1,10,100])
        er_train = np.zeros(len(lam), dtype=float)
        er_test = np.zeros(len(lam), dtype=float)
        for j in ["","bin","z","log"]:
            for i in range(0, len(lam)):
                lr = LogisticRegression(l2_lambda=lam[i], preprocessing=j, eta=1e-4, max_epoch=50, l2_on=True)
                lr.train()
                er_train[i] = lr.test(case="train")
                er_test[i] = lr.test(case="test")
                print(j,"lam",lam[i],"train",er_train[i],"test",er_test[i])

    if exp == "method":
        # lr = LogisticRegression(1)
        # start_time = clock()
        # lr.train()
        # end_time = clock()
        # print(end_time-start_time)
        lr = LogisticRegression(l2_lambda=1, preprocessing="z", eta=0.001, max_epoch=1e6, l2_on=True)
        start_time = clock()
        lr.train(method="GD")
        end_time = clock()
        err = lr.test(case="test")
        print(end_time-start_time,"s")
        print(err)


def experiment_linear(exp="lam"):
    if exp == "lam":
        lam = np.hstack((np.arange(1, 10), np.arange(10, 105, 5)))
        er_train = np.zeros(len(lam), dtype=float)
        er_test = np.zeros(len(lam), dtype=float)
        for i in range(0, len(lam)):
            linear = LinearRegression(l2_lambda=lam[i], preprocessing="", eta=1e-4, l2_on=True)
            linear.train()
            er_train[i] = linear.test(case="train")
            er_test[i] = linear.test(case="test")
        plt.plot(lam, er_train, 'b-', label='train error ')
        plt.plot(lam, er_test, 'g-', label='test error ')
        plt.xlim((0, 100))
        plt.ylim((0.10, 0.15))
        plt.xlabel('lambda for L2 regularization')
        plt.ylabel('error rate')
        plt.title('lambda vs error rate in Linear')
        plt.legend()
        plt.savefig('Linear-lam.png')

    if exp == "pre":
        lam = np.array([1])
        er_train = np.zeros(len(lam), dtype=float)
        er_test = np.zeros(len(lam), dtype=float)
        for j in ["log"]:
            for i in range(0, len(lam)):
                linear = LinearRegression(l2_lambda=lam[i], preprocessing=j, eta=1e-5, l2_on=True)
                linear.train()
                # for e in range(500):
                #     linear.sgd()
                er_train[i] = linear.test(case="train")
                er_test[i] = linear.test(case="test")
                print(j,"lam",lam[i],"train",er_train[i],"test",er_test[i])

    if exp == "sgd":

        # lam = np.hstack((np.arange(1, 200,5), np.arange(200, 10000, 5)))
        # er_train = np.zeros(len(lam), dtype=float)
        # er_test = np.zeros(len(lam), dtype=float)
        # for i in range(0, len(lam)):
        #     linear = LinearRegression(l2_lambda=1, preprocessing="", eta=lam[i]*1e-9,mom=0.5, l2_on=True)
        #     linear.sgd()
        #     er_train[i] = linear.test(case="train")
        #     er_test[i] = linear.test(case="test")
        # plt.plot(lam, er_train, 'b-', label='train error ')
        # plt.plot(lam, er_test, 'g-', label='test error ')
        # plt.xlim((1, 10000))
        # plt.ylim((0.1, 0.6))
        # plt.xlabel('learning rate 1e-9')
        # plt.ylabel('error rate')
        # plt.title('l-rate vs e-rate in LinearRegression with l-rate step decay')
        # plt.legend()
        # plt.savefig('Learningrate2.png')

        # lam = np.arange(1,505,5)
        # precision = np.zeros(len(lam), dtype=float)
        # recall = np.zeros(len(lam), dtype=float)
        # for i in range(0, len(lam)):
        #     linear = LinearRegression(l2_lambda=1, preprocessing="", eta=lam[i]*1e-8,mom=0.5, l2_on=True)
        #     linear.sgd()
        #     precision[i], recall[i] = linear.recall_precision_test()
        #
        # linear = LinearRegression(l2_lambda=1, preprocessing="", eta=lam[i] * 1e-8, mom=0.5, l2_on=True)
        # linear.train()
        # ls_precision, ls_recall = linear.recall_precision_test()
        # ls_line = np.array([1,500])
        # pre_line= ls_precision * np.ones(len(ls_line), dtype=float)
        # rec_line = ls_recall * np.ones(len(ls_line), dtype=float)
        #
        # plt.plot(ls_line, pre_line, 'y-', label='least square precision ')
        # plt.plot(ls_line, rec_line, 'r-', label='least square recall ')
        # plt.plot(lam, precision, 'b-', label='sgd precision ')
        # plt.plot(lam, recall, 'g-', label='sgd recall ')
        # plt.xlim((1, 500))
        # plt.ylim((0, 1))
        # plt.xlabel('learning rate 1e-8')
        # plt.ylabel('recall-precision rate')
        # plt.title('learning rate vs recall-precision rate in LinearRegression')
        # plt.legend()
        # plt.savefig('rec-pre1.png')

        lam = np.arange(0,105,5)
        lam[0]=1
        precision = np.zeros(len(lam), dtype=float)
        recall = np.zeros(len(lam), dtype=float)
        for i in range(0, len(lam)):
            linear = LinearRegression(l2_lambda=1, preprocessing="log", eta=lam[i]*1e-5,mom=0.5, l2_on=True)
            for e in range(500):
                linear.sgd()
            precision[i], recall[i] = linear.recall_precision_test()

        linear = LinearRegression(l2_lambda=1, preprocessing="log", eta=lam[i] * 1e-8, mom=0.5, l2_on=True)
        linear.train()
        ls_precision, ls_recall = linear.recall_precision_test()
        ls_line = np.array([1,100])
        pre_line= ls_precision * np.ones(len(ls_line), dtype=float)
        rec_line = ls_recall * np.ones(len(ls_line), dtype=float)

        plt.plot(ls_line, pre_line, 'y-', label='least square precision(log) ')
        plt.plot(ls_line, rec_line, 'r-', label='least square recall (log)')
        plt.plot(lam, precision, 'b-', label='sgd precision (log)')
        plt.plot(lam, recall, 'g-', label='sgd recall (log)')
        plt.xlim((1, 100))
        plt.ylim((0, 1))
        plt.xlabel('learning rate 1e-5')
        plt.ylabel('recall-precision rate')
        plt.title('learning rate vs recall-precision rate in LinearRegression(log)')
        plt.legend()
        plt.savefig('rec-pre3.png')


def experiment_knn(exp="K"):
    if exp == "K":
        lam = np.hstack((np.arange(1, 10), np.arange(10, 105, 5)))
        er_train = np.zeros(len(lam), dtype=float)
        er_test = np.zeros(len(lam), dtype=float)
        for i in range(0, len(lam)):
            kc = KNNClassifier(preprocessing="",K=lam[i])
            er_train[i] = kc.test(case="train")
            er_test[i] = kc.test(case="test")
            print(er_test[i])
        plt.plot(lam, er_train, 'b-', label='train error ')
        plt.plot(lam, er_test, 'g-', label='test error ')
        plt.xlim((1, 100))
        plt.ylim((0.1, 0.4))
        plt.xlabel('K')
        plt.ylabel('error rate')
        plt.title('K vs error rate in KNN')
        plt.legend()
        plt.savefig('KNN.png')

    if exp == "pre":
        lam = np.array([1,10,100])
        er_train = np.zeros(len(lam), dtype=float)
        er_test = np.zeros(len(lam), dtype=float)
        for j in ["","bin","z","log"]:
            for i in range(0, len(lam)):
                kc = KNNClassifier(preprocessing=j, K=lam[i])
                er_train[i] = kc.test(case="train")
                er_test[i] = kc.test(case="test")
                print(j,"lam",lam[i],"train",er_train[i],"test",er_test[i])


if __name__ == '__main__':
    # experiment_logistic(exp="method")

    experiment_linear(exp="sgd")

    # experiment_knn(exp="K")

