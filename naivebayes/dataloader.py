import scipy.io as sio
import numpy as np


def load_data(t="", preprocess=False, verbose=False, bias=False):
    data = sio.loadmat("./spamData.mat")
    feature_size = len(data["Xtrain"][1])

    def binarized(d):
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if d[i][j] > 0.0:
                    d[i][j] = 1
                else:
                    d[i][j] = 0
        return np.array(d, dtype=int)

    def binarized_mean(d):
        mean=np.mean(d, axis=0)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if d[i][j] > mean[j]:
                    d[i][j] = 1
                else:
                    d[i][j] = 0
        return np.array(d, dtype=int)

    def binarized_median(d):
        median=np.median(d, axis=0)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if d[i][j] > median[j]:
                    d[i][j] = 1
                else:
                    d[i][j] = 0
        return np.array(d, dtype=int)

    def logtransform(d):
        # TODO: Please implement me.
        return np.log(d + 0.1)

    def znormalization(d):
        # TODO: Please implement me.
        return (d-np.mean(d, axis=0)) / np.std(d, axis=0)

    def flat(d):
        return np.array(d, dtype=int).flatten()

    Xtrain = data['Xtrain']
    ytrain = data['ytrain']
    Xtest = data['Xtest']
    ytest = data['ytest']
    # Xtest = flat(data['Xtest'])
    # ytest = flat(data['ytest'])


    if verbose:
        print ("Loaded " + str(len(ytrain)) + " train samples, " + \
              "and " + str(len(ytest)) + " test samples, " + \
              "feature of X is of length " + str(feature_size) + ".")

    # TODO: explore different preprocessing methods if needed.
    if t == "bi":
        print ("Binarized preprocessing!")
        return binarized(Xtrain), ytrain, binarized(Xtest), ytest, feature_size
    elif t == "log":
        print ("log-transformation preprocessing!")
        return logtransform(Xtrain), ytrain, logtransform(Xtest), ytest, feature_size
    elif t == "z":
        print ("z-normalization preprocessing!")
        return znormalization(Xtrain), ytrain, znormalization(Xtest), ytest, feature_size
    elif t == "bimean":
        print ("Binarized mean preprocessing!")
        return binarized_mean(Xtrain), ytrain, binarized_mean(Xtest), ytest, feature_size
    elif t == "bimedian":
        print ("Binarized medianpreprocessing!")
        return binarized_median(Xtrain), ytrain, binarized_median(Xtest), ytest, feature_size
    else:
        return Xtrain, ytrain, Xtest, ytest, feature_size


if __name__ == '__main__':
    load_data(verbose=True)
