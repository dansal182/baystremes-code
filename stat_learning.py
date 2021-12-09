import matplotlib.pyplot as plt
from bayestremes.simulation import *
import bayestremes.evm as evm


def stand_var(x):
    return (x - x.mean()) / x.std()


def clt_sim(nsum, nsim):
    y = np.linspace(0, 1, nsim)
    for i in range(nsum):
        y += np.random.random(nsim)
    plt.figure(figsize=(6, 6))
    plt.hist(x=stand_var(y), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.8)
    plt.show()
    # plt.close('all')
    return y


def get_stds(x0, x1):
    y0 = x0.var()
    y1 = x1.var()
    out = ((len(x0) - 1) * y0 + (len(x1) - 1) * y1) / (len(x0) + len(x1) - 2)
    return out


def alpha_bayes_est(xsam, threshold, a0, b0):
    if len(xsam[xsam >= threshold]) != 0:
        out = (len(xsam[xsam >= threshold]) + a0) / (np.log(xsam[xsam >= threshold] / threshold).sum() + b0)
    else:
        print('Threshold larger than sample maxima')
        out = 0
    return out


def alphas_vec(xsam, thresholds, a0, b0):
    out = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        out[i] = alpha_bayes_est(xsam, t, a0, b0)
    return out


def get_subset(array0, array1, criteria):
    out = []
    for i, x in enumerate(array0):
        if array1[i] == criteria:
            out.append(x)
        return np.array(out)


def gibbs_threshold_search_alg(xsam, thresholds, probs_initial, iter):
    xworking = xsam[xsam >= min(thresholds)]
    n = len(xworking)
    latent = np.zeros(n)
    K = len(thresholds)
    probs_matrix = np.empty(shape=(iter, K))
    alphas_matrix = np.empty(shape=(iter, K))
    probs_matrix[0, :] = probs_initial
    alphas_matrix[0, :] = alphas_vec(xsam=xworking, thresholds=thresholds, a0=0.1, b0=0.1)
    for i in range(1, iter):
        betas = probs_matrix[i - 1, :]
        alphas = alphas_matrix[i - 1, :]
        for j in range(n):
            bets = np.zeros(K)
            for t in range(K):
                bets[t] = betas[t] * evm.Pareto(alpha=alphas[t], threshold=thresholds[t]).density(xworking[j])
            latent[j] = sample_integer(bets / bets.sum(), np.arange(1, K + 1))
        m = np.zeros(K)
        a = np.zeros(K)
        b = np.zeros(K)
        for k in range(K):
            m[k] = (np.array(latent) == k + 1).sum()
            a[k] = 1 + m[k]
            x_K = get_subset(xworking, latent, k + 1)
            b[k] = 1 + np.log(x_K[x_K >= thresholds[k]] / thresholds[k]).sum()
        for l in range(K):
            alphas_matrix[i, l] = sim_gamma(alpha=a[l], lmbda=b[l], size=1)
        probs_matrix[i, :] = np.random.dirichlet(m + 1, 1)
    return probs_matrix


def quasi_conjugate_sampling(iterations, initial, xsam, threshold, hyper_prior, hyp_gamma,  mp):
    if len(xsam[xsam > threshold]) != len(xsam):
        xsam = xsam[xsam > threshold]
    chain = np.empty(shape=(iterations + 1, 3))
    probabilities = np.empty(shape=(iterations + 1, 3))
    chain[0, :] = initial
    probabilities[0, :] = np.zeros(3)
    ax = hyper_prior[0] + len(xsam)
    bx = hyper_prior[1] + np.log(xsam / threshold).sum()
    for i in np.arange(iterations):
        pro1 = jump_alpha(ax, bx)
        probabilities[i + 1, 0] = ratio_alpha(chain[i, :], xsam, threshold, pro1, mp, hyper_prior, [ax, bx])
        chain[i + 1, 0] = acceptance(chain[i, 0], pro1, probabilities[i, 0])
        pro2 = jump_theta(chain[i, 1], hyper_prior[2])
        if pro2 < 1e-6:
            pro2 = 1.0
        probabilities[i + 1, 1] = ratio_theta(chain[i, :], xsam, threshold, pro2, mp)
        chain[i + 1, 1] = acceptance(chain[i, 1], pro2, probabilities[i, 1])
        pro3 = jump_gamma(chain[i, 2], hyper_prior[3], threshold)
        if pro3 > threshold:
            pro3 = 0.0
        probabilities[i + 1, 2] = ratio_gamma(chain[i, :], xsam, threshold, pro3, mp, [hyper_prior[3]], hyp_gamma)
        chain[i + 1, 2] = acceptance(chain[i, 2], pro3, probabilities[i, 2])
        if not support_check(xsam, chain[i + 1, 1], chain[i + 1, 2], threshold):
            chain[i + 1, 1] = 1.0
            chain[i + 1, 2] = 0.0
    out = pd.DataFrame(chain, columns=["Alpha", "Theta", "Gamma"])
    out['ind'] = np.arange(len(out))
    pout = pd.DataFrame(probabilities, columns=["P(Alpha)", "P(Theta)", "P(Gamma)"])
    pout['ind'] = np.arange(len(pout))
    return pd.merge(out, pout, on='ind').drop(columns='ind')
