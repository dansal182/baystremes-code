from bayestremes.simulation import *
import bayestremes.evm as evm


def max_tail_test(xsam, prior_odds, a0, b0, nsim, bf_size):
    if isinstance(xsam, np.ndarray):
        x = xsam.max()
        alphas = sim_gamma(a0, b0, nsim)
        h0_vals = np.linspace(0, 1, nsim)
        for i, val in enumerate(h0_vals):
            h0_vals[i] = evm.Frechet(alphas[i], theta=1, gamma=0).density(x)
        h0_val = h0_vals.mean()
        if 1.0 - h0_val < 1.0:
            h1 = evm.Gumbel(theta=1, gamma=0)
            aux = h1.density(x) / h0_val * prior_odds
            if aux < bf_size:
                print('Fail to reject H0 no empirical evidence against a Heavy-Tail, with BF equal to:')
                print(str(aux))
            else:
                print('Reject H0, with BF equal to:')
                print(str(aux))
        else:
            print('No posterior likelihood for those hyperparameters, changing a0 and b0 could help.')
    else:
        print('Test for numpy.ndarrays')


def bayes_factor(x, alpha, prior_odds):
    h0 = evm.Frechet(alpha=alpha, theta=1, gamma=0)
    h1 = evm.Gumbel(theta=1, gamma=0)
    return h1.density(x) / h0.density(x) * prior_odds


def support_check_gev(xsam, xi):
    aux = 1.0 + xi * xsam
    if len(aux[aux > 0.0]) == len(xsam):
        return True
    else:
        return False


def gev_likelihood(xsam, xi):
    if support_check_gev(xsam, xi):
        if abs(xi) > 1e-6:
            a = np.exp(-1 / xi * np.log(1 + xi * xsam)).sum()
            b = (1 / xi + 1) * np.log(1 + xi * xsam).sum()
            loglike = - (a + b)
        else:
            a = (np.exp(-xsam) + xsam).sum()
            loglike = -a
    else:
        loglike = -np.Inf
    return loglike


def gev_gradient_loglike(xsam, xi):
    if support_check_gev(xsam, xi):
        if abs(xi) > 1e-6:
            a = np.exp(-1 / xi * np.log(1 + xi * xsam).sum())
            b = (np.log(1 + xi * xsam) * (1 + a)).sum()
            c = (xsam / (1 + xi * xsam) * (1 + a)).sum()
            d = (xsam / (1 + xi * xsam)).sum()
            out = 1 / (xi ** 2) * b - (1 / xi * c + d)
        else:
            out = 0.0
    else:
        out = np.NAN
    return out


def hypo_interval_ratio(xsam, xi_old, xi_proposal, momenta, momenta_aux):
    if support_check_gev(xsam, xi_old) & support_check_gev(xsam, xi_proposal):
        ratio = np.exp(gev_likelihood(xsam, xi_proposal) + 1 / 2 * momenta * momenta -
                       (gev_likelihood(xsam, xi_old) + 1 / 2 * momenta_aux * momenta_aux))
    else:
        ratio = 0
    if np.isnan(ratio):
        ratio = 0.0
    return min(1.0, ratio)


def hypo_interval_chain(iterations, xsam, initial, c, steps, momenta_std):
    chain = np.empty(shape=iterations + 1)
    probs = np.empty(shape=iterations + 1)
    chain[0] = initial
    probs[0] = 0.0
    for i in range(1, iterations + 1):
        momenta = np.random.normal(0, momenta_std, 1)
        proposal = chain[i - 1]
        momenta_aux = momenta + c * gev_gradient_loglike(xsam, proposal) / 2.0
        props = np.empty(shape=steps + 1)
        momentas = np.empty(shape=steps + 1)
        props[0] = proposal
        momentas[0] = momenta_aux
        for l in range(1, steps + 1):
            momentas[l] = momentas[l - 1] + c / 2 * gev_gradient_loglike(xsam, props[l - 1])
            props[l] = props[l - 1] + c / momenta_std * momentas[l]
            if np.isnan(gev_gradient_loglike(xsam, props[l])):
                props[l] = 0.0
        momenta_aux = momentas[steps]
        proposal = props[steps]
        momenta_aux += c * gev_gradient_loglike(xsam, proposal) / 2.0
        probs[i] = hypo_interval_ratio(xsam, chain[i - 1], proposal,
                                       momenta, momenta_aux)
        chain[i] = acceptance(chain[i - 1], proposal, probs[i])
    print('Average acceptance probability: ' + str(probs.mean()))
    return chain[round(iterations / 2):iterations + 1]
