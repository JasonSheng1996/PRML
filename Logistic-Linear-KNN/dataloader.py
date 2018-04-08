import scipy.io as sio
import numpy as np


def load_data(t="", preprocess=False, verbose=False, bias=False):
    data = sio.loadmat("./spamData.mat")
    feature_size = len(data["Xtrain"][1])

    def binarized(d):
        d = [[1 if e > 0 else 0 for e in dd] for dd in d]
        return np.array(d, dtype=int)

    def logtransform(d):
        return np.log(d + 0.1)

    def znormalization(d):
        return (d-np.mean(d, axis=0)) / np.std(d, axis=0)

    def flat(d):
        return np.array(d, dtype=int).flatten()

    Xtrain = data['Xtrain']
    ytrain = data['ytrain']
    Xtest = data['Xtest']
    ytest = data['ytest']

    if verbose:
        print("Loaded " + str(len(ytrain)) + " train samples, " + \
              "and " + str(len(ytest)) + " test samples, " + \
              "feature of X is of length " + str(feature_size) + ".")

    # TODO: explore different preprocessing methods if needed.
    if t == "bin":
        print("Binarized preprocessing!")
        return binarized(Xtrain), ytrain, binarized(Xtest), ytest, feature_size
    elif t == "log":
        print("log-transformation preprocessing!")
        return logtransform(Xtrain), ytrain, logtransform(Xtest), ytest, feature_size
    elif t == "z":
        print("z-normalization preprocessing!")
        return znormalization(Xtrain), ytrain, znormalization(Xtest), ytest, feature_size
    else:
        return Xtrain, ytrain, Xtest, ytest, feature_size


if __name__ == '__main__':
    load_data(verbose=True)
