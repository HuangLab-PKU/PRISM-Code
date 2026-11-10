"""Select separated Cy3 modes before locating inter-mode valleys."""
from __future__ import annotations
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
DEFAULT_BW = 0.06

def choose_g_layer_cuts(values, glayer, bw=DEFAULT_BW, n_grid=500, q_lo=5.0, q_hi=95.0, max_points=120000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo = float(np.percentile(values, q_lo))
    hi = float(np.percentile(values, q_hi))
    diag = {'q_lo': lo, 'q_hi': hi, 'peaks': [], 'valleys': [], 'min_separation': 0.0}
    if not hi > lo:
        return ([], diag)
    sub = values[::max(1, len(values) // max_points)]
    xs = np.linspace(lo, hi, n_grid)
    dens = gaussian_kde(sub, bw_method=bw)(xs)
    diag['valleys'] = [float(v) for v in xs[argrelextrema(dens, np.less)[0]] if lo < v < hi]
    min_sep = (hi - lo) / (2.0 * glayer)
    diag['min_separation'] = float(min_sep)
    peak_idx = argrelextrema(dens, np.greater)[0]
    step = max(1, n_grid // 50)
    if dens[0] > dens[step]:
        peak_idx = np.append(peak_idx, 0)
    if dens[-1] > dens[-1 - step]:
        peak_idx = np.append(peak_idx, len(xs) - 1)
    peak_idx = np.unique(peak_idx)
    chosen: list[int] = []
    for i in peak_idx[np.argsort(dens[peak_idx])[::-1]]:
        if all((abs(xs[i] - xs[j]) >= min_sep for j in chosen)):
            chosen.append(int(i))
        if len(chosen) == glayer:
            break
    chosen.sort()
    diag['peaks'] = [float(xs[i]) for i in chosen]
    if len(chosen) < glayer:
        return ([], diag)
    cuts = []
    for a, b in zip(chosen[:-1], chosen[1:]):
        seg = dens[a:b + 1]
        cuts.append(float(xs[a + int(np.argmin(seg))]))
    return (cuts, diag)
