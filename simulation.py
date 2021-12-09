import numpy as np
import pandas as pd


def sim_gamma(alpha, lmbda, size):
    return np.random.gamma(alpha, 1 / lmbda, size)


def sample_integer(probabilities, int_values):
    p_test = np.random.random(1)
    p_dist = 0.0
    out = int_values[0]
    for i, p in enumerate(probabilities):
        p_dist += p
        if p_test <= p_dist:
            out = int_values[i]
            break
    return out


def sample_integers(probabilities, int_values, size):
    out = np.arange(size)
    for i in range(size):
        out[i] = sample_integer(probabilities, int_values)
    return out


def support_check(xsam, theta, gamma, threshold):
    if theta < 1e-6:
        out = False
    else:
        check = (xsam - gamma) / theta
        if (len(check[check > 0.0]) == len(xsam)) & ((threshold - gamma) / theta > 0.0):
            out = True
        else:
            out = False
    return out


def ppp_log_like(extremes, alpha, theta, gamma, threshold, nperiods):
    xsam, a, th, g, u, mp = extremes, alpha, theta, gamma, threshold, nperiods
    if support_check(xsam, th, g, u):
        bx = np.log((xsam - g) / th).sum()
        return len(xsam) * np.log(a / th) - (mp * ((u - g) / th) ** (-a) + (a + 1) * bx)
    else:
        return -np.Inf


def post_alpha(param, xsam, threshold, hyper, mp):
    a, th, g = param[0], param[1], param[2]
    u = threshold
    a0, b0 = hyper[0], hyper[1]
    alpha_prior = (a0 - 1) * np.log(a) - a * b0
    if np.isinf(-ppp_log_like(xsam, a, th, g, u, mp)):
        return 0.0
    else:
        return ppp_log_like(xsam, a, th, g, u, mp) + alpha_prior


def post_theta(param, xsam, threshold, hyper, mp):
    a, th, g = param[0], param[1], param[2]
    u = threshold
    p0, q0 = hyper[0], hyper[1]
    if support_check(xsam, th, g, u):
        out = np.log(th) - mp * ((u - g) / th) ** (-a) + (p0 - 1) * np.log(th) - th * q0
    else:
        out = 0.0
    return out


def post_gamma(param, xsam, threshold, hyper, mp):
    a, th, g = param[0], param[1], param[2]
    u = threshold
    c = hyper[0]
    if np.isinf(-ppp_log_like(xsam, a, th, g, u, mp)):
        return 0.0
    else:
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
    r = (post_alpha([proposal, param[1], param[2]], xsam, threshold, hyp_prior, mp) +
               dens_alpha(param[0], hyp_post[0], hyp_post[1]) -
               post_alpha(param, xsam, threshold, hyp_prior, mp) -
               dens_alpha(proposal, hyp_post[0], hyp_post[1]))
    if r < 709:
        ra = np.exp(r)
    else:
        ra = 1
    if np.isnan(ra):
        ra = 0.0
    return min(ra, 1)


def ratio_theta(param, xsam, threshold, proposal, mp):
    if np.isinf(post_theta([param[0], proposal, param[2]], xsam, threshold, [1, 1], mp)) | np.isinf(
            post_theta(param, xsam, threshold, [1, 1], mp)):
        r = 0.0
    else:
        r = np.exp(post_theta([param[0], proposal, param[2]], xsam, threshold, [1, 1], mp) -
                   post_theta(param, xsam, threshold, [1, 1], mp))
    if np.isnan(r):
        r = 0.0
    return min(r, 1)


def ratio_gamma(param, xsam, threshold, proposal, mp, hyp_prior, hyp_post):
    if np.isinf(post_gamma([param[0], param[1], proposal], xsam, threshold, hyp_prior, mp)) | np.isinf(
            post_gamma(param, xsam, threshold, hyp_prior, mp)):
        r = 0.0
    else:
        r = np.exp(post_gamma([param[0], param[1], proposal], xsam, threshold, hyp_prior, mp) +
                   dens_gamma(param[2], hyp_post) -
                   post_gamma(param, xsam, threshold, hyp_prior, mp) -
                   dens_gamma(proposal, hyp_post))
    if np.isnan(r):
        r = 0.0
    return min(r, 1)


def acceptance(old, props, ratio):
    rat = max(1, ratio)
    if np.random.random(1) < rat:
        return props
    else:
        return old
