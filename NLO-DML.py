# ============================================================================
# NLO CROSS-FITTING
# ============================================================================

def block_partition(n, K=None):
    """The K adjacent blocks M_k of size T_b = floor(n/K), remainder in M_K."""
    K = K_FOLDS if K is None else K
    T_b = n // K
    assert T_b >= 2, f"n={n} is too small for K={K} blocks"
    blocks = [np.arange((k) * T_b, (k + 1) * T_b) for k in range(K)]
    blocks[-1] = np.arange((K - 1) * T_b, n)      # remainder appended to M_K
    return blocks, T_b


def fit_fold_indices(blocks, k, scheme):
    """Indices of the fold on which eta_hat_k is estimated.
    'nlo' (see paper) or 'standard' 
    """
    K = len(blocks)
    if scheme == 'nlo':
        excluded = {j for j in (k - 1, k, k + 1) if 0 <= j < K}
    elif scheme == 'standard':
        excluded = {k}
    else:
        raise ValueError(f"unknown cross-fitting scheme {scheme!r}")
    idx = np.concatenate([blocks[j] for j in range(K) if j not in excluded])
    assert idx.size > 0, "empty fit fold; increase K or n"
    return np.sort(idx)


def describe_folds(blocks, scheme, T_b, verbose=True):
    """Report each fold's fit/eval sizes and the realised temporal separation.
    """
    rows = []
    for k in range(len(blocks)):
        ev = blocks[k]
        ft = fit_fold_indices(blocks, k, scheme)
        gap = int(np.min(np.abs(ft[:, None] - ev[None, :]))) if ft.size else -1
        rows.append({'fold': k + 1, 'eval_rows': f"{ev[0]}-{ev[-1]}",
                     'n_eval': ev.size, 'n_fit': ft.size, 'min_gap': gap})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\n  Fold construction ({scheme}, K={len(blocks)}, T_b={T_b}):")
        print(df.to_string(index=False))
        if scheme == 'nlo':
            worst = df['min_gap'].min()
            print(f"    Minimum separation between fit and evaluation folds: "
                  f"{worst} periods  (T_b = {T_b})")
            assert worst >= T_b, (
                f"NLO separation {worst} < T_b {T_b}: neighbour removal failed")
    return df


# ============================================================================
# DML ESTIMATOR 
# ============================================================================


def hac_bandwidth(n, rule=None):
    """Newey-West bandwidth b_n, satisfying b_n -> inf and b_n/n -> 0."""
    rule = HAC_BANDWIDTH if rule is None else rule
    if isinstance(rule, (int, np.integer)):
        return max(0, int(rule))
    if rule == 'newey_west':
        return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    if rule in ('cube_root', 'auto'):
        return int(np.floor(n ** (1.0 / 3.0)))
    raise ValueError(f"unknown HAC_BANDWIDTH {rule!r}")


def newey_west_lrv(psi, bandwidth=None, demean=False):
    """Bartlett-kernel long-run variance of a scalar series.
    Returns sigma_psi^2 = gamma_0 + 2 sum_j (1 - j/(b+1)) gamma_j
    """
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.size
    b = hac_bandwidth(n) if bandwidth is None else int(bandwidth)
    b = min(b, n - 1)
    x = psi - psi.mean() if demean else psi
    lrv = float(np.mean(x ** 2))
    for j in range(1, b + 1):
        w = 1.0 - j / (b + 1.0)
        lrv += 2.0 * w * float(np.mean(x[j:] * x[:-j]))
    return max(lrv, 1e-30), b


def dml_estimate(Y, D, g_hat, m_hat, alpha_0=None, label=''):
    """alpha_hat and its HAC standard error from cross-fitted nuisances.

    Partialling-out (Robinson) score, i.e. Chernozhukov et al. (2018) score 2:

        psi^a = -(D_t - m(W_t))^2,     psi^b = (Y_t - g(W_t))(D_t - m(W_t))
        J_hat = mean(psi^a),           alpha_hat = -J_hat^{-1} mean(psi^b)
    """
    Y = np.asarray(Y, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    g_hat = np.asarray(g_hat, dtype=np.float64)
    m_hat = np.asarray(m_hat, dtype=np.float64)
    n = Y.size

    V_hat = D - m_hat
    psi_b = (Y - g_hat) * V_hat
    J_hat = -float(np.mean(V_hat ** 2))
    assert abs(J_hat) > 1e-12, "E[V^2] is numerically zero; the treatment is fully predictable"
    alpha_hat = -float(np.mean(psi_b)) / J_hat

    U_hat = (Y - g_hat) - alpha_hat * V_hat  # clean residual = U at the truth
    psi = V_hat * U_hat                      # cross-fitted score at alpha_hat
    lrv, b_n = newey_west_lrv(psi)
    var_hac = lrv / (J_hat ** 2) / n
    se_hac = float(np.sqrt(var_hac))

    var_iid = float(np.mean(psi ** 2)) / (J_hat ** 2) / n
    se_iid = float(np.sqrt(var_iid))

    z = norm.ppf(0.5 + CI_LEVEL / 2.0)
    ci = (alpha_hat - z * se_hac, alpha_hat + z * se_hac)
    ci_iid = (alpha_hat - z * se_iid, alpha_hat + z * se_iid)

    out = {'alpha_hat': alpha_hat, 'J_hat': J_hat, 'se_hac': se_hac,
           'se_iid': se_iid, 'ci_low': ci[0], 'ci_high': ci[1],
           'ci_low_iid': ci_iid[0], 'ci_high_iid': ci_iid[1],
           'hac_bandwidth': b_n, 'lrv_psi': lrv,
           'lrv_ratio': lrv / max(float(np.mean(psi ** 2)), 1e-30),
           'score_autocorr_1': _ar1_autocorr(psi, 1),
           'mean_V2': float(np.mean(V_hat ** 2)), 'n': n}

    J_dv = float(np.mean(-D * V_hat))
    out['alpha_hat_v2denom'] = (-float(np.mean(psi_b)) / J_dv
                                if abs(J_dv) > 1e-12 else np.nan)

    if alpha_0 is not None:
        out['bias'] = alpha_hat - alpha_0
        out['t_stat'] = (alpha_hat - alpha_0) / se_hac
        out['covered'] = bool(ci[0] <= alpha_0 <= ci[1])
        out['covered_iid'] = bool(ci_iid[0] <= alpha_0 <= ci_iid[1])
    return out


def nuisance_diagnostics(g_hat, m_hat, l0_true, m0_true, n):
    """L2 errors of the cross-fitted nuisances and the product-rate check.

    Proposition 3's condition 3(ii) is sqrt(n) r_{g,n} r_{m,n} = o(1).  With the
    true nuisances in hand this is directly measurable, which turns an
    assumption into a diagnostic: if the product blows up, any coverage failure
    is a nuisance-rate problem, not a cross-fitting one.
    """
    r_g = float(np.sqrt(np.mean((g_hat - l0_true) ** 2)))
    r_m = float(np.sqrt(np.mean((m_hat - m0_true) ** 2)))
    return {'r_g_L2': r_g, 'r_m_L2': r_m,
            'sqrt_n_product': float(np.sqrt(n) * r_g * r_m),
            'r_g_rate_n14': r_g * n ** 0.25,     # o_P(1) iff r_g = o(n^{-1/4})
            'r_m_rate_n14': r_m * n ** 0.25,
            'r2_g': float(r2_score(l0_true, g_hat)),
            'r2_m': float(r2_score(m0_true, m_hat))}
