import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import multivariate_normal as multi_gauss
from mpl_toolkits.mplot3d import Axes3D

class GMM():
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster

    def fit(self, X, iter_max=100):
        np.random.seed(0)
        # initialize
        self.N, self.D = X.shape
        self.pi = np.ones(self.n_cluster) / self.n_cluster
        self.means = np.random.uniform(X.min(), X.max(), (self.n_cluster, self.D))
        self.cov = np.empty([self.n_cluster, self.D, self.D])
        for k in range(self.n_cluster):
            self.cov[k] = 10*np.eye(self.D)
        record = []
        # EM algorithm
        for step in range(iter_max):
            gamma = self.E_step(X)
            self.M_step(X, gamma)
            logL = self.loglikelihood(X)
            print("iter: %d, log likelihood: %f" % (step, logL))
            #if step % 10 ==0:
            record.append([step, logL])
            if step == 0:
                oldL = logL
            else:
                if logL - oldL < 1e-5:
                    break
                else:
                    oldL = logL
        return np.array(record)

    def E_step(self, X):
        gamma = np.empty([self.N, self.n_cluster])
        for n in range(self.N):
            for k in range(self.n_cluster):
                gamma[n, k] = self._gauss(X[n].reshape(self.D, 1), self.means[k].reshape(self.D, 1), self.cov[k])
        return gamma/np.sum(gamma, axis=1, keepdims=True)

    def M_step(self, X, gamma):
        Nk = np.sum(gamma, axis=0)
        # pi update
        self.pi = Nk / np.sum(Nk)
        # means update
        self.means = gamma.T @ X / Nk[:, None]
        # covariance mat update
        cov = np.empty([self.n_cluster, self.n_cluster])
        dx = np.empty([self.n_cluster, self.N, self.D])
        for k in range(self.n_cluster):
            for n in range(self.N):
                dx[k, n] = X[n] - self.means[k]
        self.cov = np.einsum("nk, kni, knj -> kij", gamma, dx, dx)/Nk[:, None, None]

    def loglikelihood(self, X):
        # compute log likelohood
        logL = 0
        for n in range(self.N):
            L = 0
            for k in range(self.n_cluster):
                L += self.pi[k] * self._gauss(X[n].reshape(self.D, 1), self.means[k].reshape(self.D, 1), self.cov[k])
            logL += np.log(L)
        return logL

    def _gauss(self, x, mean, cov):
        D = len(x)
        return   np.exp(-(x-mean).T @np.linalg.inv(cov)@(x-mean))/(np.linalg.det(cov) * (2 * math.pi)**(D/2))

    def classify(self, X):
        N = len(X)
        posterior = np.empty([N, self.n_cluster])
        for n in range(N):
            for k in range(self.n_cluster):
                posterior[n, k] = self.pi[k] * self._gauss(X[n].reshape(self.D, 1), self.means[k].reshape(self.D, 1), self.cov[k])
        return posterior/np.sum(posterior, axis=1, keepdims=True), np.argmax(posterior, axis=1)

if __name__ == "__main__":
    X = np.loadtxt("x.csv", delimiter=",")
    gmm = GMM(4)
    record = gmm.fit(X, 50)
    posterior, labels = gmm.classify(X)
    # save data
    np.savetxt("z.csv", posterior, delimiter=",")
    with open("params.dat", "w") as f:
        f.write("pi:\n")
        for k in range(gmm.n_cluster):
            f.write("cluster %d: %f\n" % (k, gmm.pi[k]))
        f.write("\nmeans:\n")
        for k in range(gmm.n_cluster):
            f.write("cluster %d: %s\n" % (k, gmm.means[k]))
        f.write("\nprecison matrix:\n")
        for k in range(gmm.n_cluster):
            f.write("cluster %d\n" % k)
            f.write("%s\n" % np.linalg.inv(gmm.cov[k]))
    with open("em_likelihood.txt", "w") as f:
        f.write("step\tlog-likelihood\n")
        for i in range(len(record)):
            f.write("%d\t%f\n" % (record[i, 0], record[i, 1]))
    # plot
    colors = ["red", "lightblue", "lightgreen", "orange"]
    label_color = [colors[int(label)] for label in labels]
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        ax.plot([X[i, 0]], [X[i, 1]], [X[i, 2]], "o", color=label_color[i])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    #plt.show()
    plt.savefig("em.png")
    plt.close()
