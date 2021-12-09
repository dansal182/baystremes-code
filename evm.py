from abc import ABC
from math import *
import numpy as np


class EVM(object):
    def __init__(self, alpha=1, theta=1, gamma=1, xi=1, threshold=1):
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.gamma = float(gamma)
        self.xi = float(xi)
        self.threshold = float(threshold)

    def density(self, x):
        raise NotImplementedError("No density function for model: {0}".format(self.__class__.__name__))

    def distribution(self, x):
        raise NotImplementedError("No density function for model: {0}".format(self.__class__.__name__))

    def quantile(self, q):
        raise NotImplementedError("No density function for model: {0}".format(self.__class__.__name__))

    def percentiles(self, percentiles=None):
        if percentiles is None:
            percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        percs = []
        [percs.append(t) for t in percentiles if t not in percs]
        out = np.linspace(0, 10, len(percs))
        for i in range(len(percs)):
            out[i] = self.quantile(percs[i])
        return out

    def vec_density(self, xvec):
        out = np.linspace(1, 20, len(xvec))
        for i in range(len(xvec)):
            out[i] = self.density(xvec[i])
        return out

    def vec_quantile(self, xvec):
        out = np.linspace(1, 20, len(xvec))
        for i in range(len(xvec)):
            out[i] = self.quantile(xvec[i])
        return out

    def vec_distribution(self, xvec):
        out = np.linspace(1, 20, len(xvec))
        for i in range(len(xvec)):
            out[i] = self.distribution(xvec[i])
        return out

    def likelihood(self, xsample):
        xlk = self.vec_density(xsample)
        return np.exp(np.log(xlk).sum())

    def log_like(self, xsample):
        return np.log(self.likelihood(xsample))

    def tail(self, x):
        return 1 - self.distribution(x)


class Frechet(EVM):
    def __init__(self, alpha, theta, gamma):
        EVM.__init__(self, alpha, theta, gamma, xi=0, threshold=0)

    def distribution(self, x):
        a, th, g = self.alpha, self.theta, self.gamma
        if (x - g) / th > 0.0:
            return exp(-(exp(-a * log((x - g) / th))))
        else:
            return 0.0

    def quantile(self, q):
        if q > 0.0:
            a, th, g = self.alpha, self.theta, self.gamma
            if q < 1.0:
                return th * exp(-1 / a * log((-log(q)))) + g
        else:
            return None

    def density(self, x):
        a, th, g = self.alpha, self.theta, self.gamma
        aux = (x - g) / th
        if aux > 0.0:
            return a / th * aux ** (-(a + 1)) * self.distribution(x)
        else:
            return 0.0


class Pareto(EVM):
    def __init__(self, alpha, threshold):
        EVM.__init__(self, alpha, 0, 0, 0, threshold)

    def distribution(self, x):
        a, u = self.alpha, self.threshold
        if x > u:
            return 1 - (x/u)**(-a)
        else:
            return 0.0

    def density(self, x):
        a, u = self.alpha, self.threshold
        if x > u:
            return a / u * (x/u)**(-(a+1))
        else:
            return 0.0

    def quantile(self, q):
        if q > 0.0:
            if q < 1.0:
                a, u = self.alpha, self.threshold
                return u * np.exp(-1/a*np.log(1-q))
        else:
            return None


class Gumbel(EVM):
    def __init__(self, theta, gamma):
        EVM.__init__(self, 0, theta, gamma, 0, 0)

    def distribution(self, x):
        th, g = self.theta, self.gamma
        if x > g:
            return np.exp(-np.exp(-(x - g)/th))
        else:
            return 0.0

    def density(self, x):
        th, g = self.theta, self.gamma
        if x > g:
            return np.exp(-(np.exp(-(x - g)/th) + (x - g)/th)) * 1/th
        else:
            return 0.0

    def quantile(self, q):
        th, g = self.theta, self.gamma
        return -th*(np.log(-np.log(q))) + g


class PoissonPointProcess(EVM, ABC):
    def __init__(self, alpha, theta, gamma, threshold, nperiods):
        EVM.__init__(self, alpha, theta, gamma, 0, threshold)
        self.nperiods = int(nperiods)

    def density(self, x):
        a, th, g, u, mp = self.alpha, self.theta, self.gamma, self.threshold, self.nperiods
        if isinstance(x, int):
            nu = (u - g)/th
            if nu > 0.0:
                return np.exp(-mp*nu**(-a))*(nu**(-a))**x/factorial(x)
            else:
                return 0.0
        else:
            return 0.0

    def likelihood(self, xsample):
        a, th, g, u, mp = self.alpha, self.theta, self.gamma, self.threshold, self.nperiods
        if (u - g)/th > 0.0:
            val = 1
            for x in xsample:
                val *= a/th*((x-g)/th)**(-(a+1))
            return np.exp(-mp*((u - g)/th)**(-a))*val
