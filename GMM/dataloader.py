import numpy as np


def data_genearte():
    mean = [20, 20]
    cov = [[25, 0],
           [0, 25]]
    x1 = np.random.multivariate_normal(mean, cov, 15)
    mean = [60, 40]
    cov = [[10, 30],
           [5, 30]]
    x2 = np.random.multivariate_normal(mean, cov, 5)
    mean = [30, 60]
    cov = [[10, 5],
           [5, 10]]
    x3 = np.random.multivariate_normal(mean, cov, 10)
    mean = [40, 40]
    cov = [[10, 5],
           [5, 10]]
    x4 = np.random.multivariate_normal(mean, cov, 10)
    # data = np.vstack((np.vstack((x1, x2)), x3))
    # np.savetxt("data/rand3.txt",data)
    data = np.vstack((np.vstack((np.vstack((x1, x2)), x3)),x4))
    np.savetxt("data/rand-40.txt",data)
    return


def load_data(d = ""):
    # old faithful
    if d == "OF":
        return np.loadtxt("data/oldfaithful.txt")

    if d=="rand3":
        return np.loadtxt("data/rand3.txt")

    if d == "rand4":
        return np.loadtxt("data/rand4.txt")

    if d == "200":
        return np.loadtxt("data/rand-200.txt")

    if d == "80":
        return np.loadtxt("data/rand-80.txt")

    if d == "40":
        return np.loadtxt("data/rand-40.txt")

if __name__=="__main__":
    data_genearte()