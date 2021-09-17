import numpy as np
import pandas as pd


def sim_gamma(alpha, lmbda, size):
    return np.random.gamma(alpha, 1 / lmbda, size)


def support_check(xsam, theta, gamma, threshold):
    check = (xsam - gamma) / theta
    if len(check[check > 0.0]) == len(xsam) & (threshold - gamma) / theta > 0.0:
        return True
    else:
        return False


def ppp_log_like(extremes, alpha, theta, gamma, threshold, nperiods):
    xsam, a, th, g, u, mp = extremes, alpha, theta, gamma, threshold, nperiods
    if support_check(xsam, th, g, u):
        bx = np.log((xsam - g) / th).sum()
        return len(xsam) * np.log(a) - (mp * ((u - g) / th) ** (-a) + (a + 1) * bx)
    else:
        return -np.Inf


def post_alpha(param, xsam, threshold, hyper, mp):
    a, th, g = param[0], param[1], param[2]
    u = threshold
    a0, b0 = hyper[0], hyper[1]
    alpha_prior = (a0 - 1) * np.log(a) - a * b0
    return ppp_log_like(xsam, a, th, g, u, mp) + alpha_prior


def post_theta(param, xsam, threshold, hyper, mp):
    a, th, g = param[0], param[1], param[2]
    u = threshold
    p0, q0 = hyper[0], hyper[1]
    if support_check(xsam, th, g, u):
        return np.log(th) - mp * ((u - g) / th) ** (-a) + (p0 - 1) * np.log(th) - th * q0
    else:
        return -np.Inf


def post_gamma(param, xsam, threshold, hyper, mp):
    a, th, g = param[0], param[1], param[2]
    u = threshold
    c = hyper[0]
    return ppp_log_like(xsam, a, th, g, u, mp) - g ** 2 / (2 * c)


def jump_alpha(ax, bx):
    return sim_gamma(ax, bx, 1)


def jump_theta(theta, hyper_std):
    return np.random.normal(theta, hyper_std, 1)


def jump_gamma(gamma, hyper_scale, threshold):
    g = np.random.gumbel(gamma, hyper_scale, 1)
    if g > threshold:
        return gamma
    else:
        return g


def dens_alpha(alpha, ax, bx):
    return (ax - 1) * np.log(alpha) - alpha * bx


def dens_gamma(gamma, hyper):
    mu = hyper[0]
    s0 = hyper[1]
    return -(np.exp(-(gamma - mu) / s0) + (gamma - mu) / s0)


def ratio_alpha(param, xsam, threshold, proposal, mp, hyp_prior, hyp_post):
    r = np.exp(post_alpha([proposal, param[1], param[2]], xsam, threshold, hyp_prior, mp) +
               dens_alpha(param[0], hyp_post[0], hyp_post[1]) -
               post_alpha(param, xsam, threshold, hyp_prior, mp) -
               dens_alpha(proposal, hyp_post[0], hyp_post[1]))
    r = 0.0 if r != r else r
    return min(r, 1)


def ratio_theta(param, xsam, threshold, proposal, mp):
    r = np.exp(post_theta(xsam, param[0], proposal, param[2], threshold, [1, 1], mp) -
               post_theta(param, xsam, threshold, [1, 1], mp))
    r = 0.0 if r != r else r
    return min(r, 1)


def ratio_gamma(param, xsam, threshold, proposal, mp, hyp_prior, hyp_post):
    r = np.exp(post_gamma([param[0], param[1], proposal], xsam, threshold, hyp_prior, mp) +
               dens_gamma(param[2], hyp_post) -
               post_gamma(param, xsam, threshold, hyp_prior, mp) -
               dens_gamma(proposal, hyp_post))
    r = 0.0 if r != r else r
    return min(r, 1)


def acceptance(old, props, ratio):
    if np.random.random(1) < ratio:
        return props
    else:
        return old


def quasi_conjugate_sampling(iterations, initial, xsam, threshold, hyper_prior, mp):
    if len(xsam[xsam > threshold]) != len(xsam):
        xsam = xsam[xsam > threshold]
    chain = np.array(np.linspace(0, 10000, (iterations + 1) * 3)).reshape(iterations + 1, 3)
    probabilities = chain
    chain[0,] = initial
    ax = hyper_prior[0] + len(xsam)
    bx = hyper_prior[1] + np.log(xsam / threshold).sum()
    for i in np.arange(iterations):
        pro1 = jump_alpha(ax, bx)
        probabilities[i, 0] = ratio_alpha(chain[i,], xsam, threshold, pro1, mp, hyper_prior, [ax, bx])
        chain[i + 1, 0] = acceptance(chain[i, 0], pro1, probabilities[i, 0])
        pro2 = jump_theta(chain[i, 1], hyper_prior[2])
        probabilities[i, 1] = ratio_theta(chain[i,], xsam, threshold, pro2, mp)
        chain[i + 1, 1] = acceptance(chain[i, 1], pro2, probabilities[i, 1])
        pro3 = jump_gamma(chain[i, 2], hyper_prior[3], threshold)
        probabilities[i, 2] = ratio_gamma(chain[i,], xsam, threshold, pro3, mp, hyper_prior[4], hyper_prior[5, 6])
        chain[i + 1, 2] = acceptance(chain[i, 2], pro3, probabilities[i, 2])
        if not support_check(xsam, chain[i + 1, 1], chain[i + 1, 2], threshold):
            chain[i + 1, 1] = 1.0
            chain[i + 1, 2] = 0.0
    probabilities[iterations + 1, ] = np.zeros(3)
    out = pd.DataFrame(chain, columns={"Alpha", "Theta", "Gamma"})
    out['ind'] = np.arange(len(out))
    pout = pd.DataFrame(probabilities, columns={"P(Alpha)", "P(Theta)", "P(Gamma)"})
    pout['ind'] = np.arange(len(pout))
    return pd.merge(out, pout, on='ind')
