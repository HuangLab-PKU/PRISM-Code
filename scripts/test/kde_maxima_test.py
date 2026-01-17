#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import argrelextrema


def find_maxima(series: pd.Series, expected: int, bw_adjust: float, channel_name: str = ""):
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.array([])
    try:
        kde = stats.gaussian_kde(s, bw_method=bw_adjust)
        x = np.linspace(s.min(), s.max(), 2000)
        y = kde(x)
        idx = argrelextrema(y, np.greater)[0]
        vals = x[idx]
        print(f"{channel_name} raw KDE peaks ({len(vals)}): {np.round(vals, 4)}")
    except Exception as e:
        print(f"{channel_name} KDE failed: {e}")
        vals = np.array([])
    if len(vals) != expected:
        print(f"{channel_name} fallback to {expected} evenly spaced")
        # evenly spaced fallback
        if expected <= 1:
            return np.array([float(s.median())])
        vals = np.linspace(float(s.min()), float(s.max()), expected)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to intensity.csv (with ch1,ch2,ch3,ch4)')
    ap.add_argument('--color-grade', type=int, default=5)
    ap.add_argument('--layer-grade', type=int, default=2)
    ap.add_argument('--bw-color', type=float, default=0.5)
    ap.add_argument('--bw-layer', type=float, default=0.4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df['sum'] = df['ch1'] + df['ch2'] + df['ch4']
    eps = 1e-12
    for ch in ['ch1','ch2','ch4','ch3']:
        df[ch] = df[ch].astype(float)
    df['ch1/A'] = df['ch1'] / (df['sum'] + eps)
    df['ch2/A'] = df['ch2'] / (df['sum'] + eps)
    df['ch4/A'] = df['ch4'] / (df['sum'] + eps)
    df['ch3/A'] = df['ch3'] / (df['sum'] + eps)
    # mirror pipeline behavior
    df['ch3/A'] = df['ch3/A'].clip(0, 5)
    g_zero = df['ch3/A'] == 0
    if g_zero.any():
        df.loc[g_zero, 'ch3/A'] = np.random.normal(0, 0.005, g_zero.sum())

    ch1_max = find_maxima(df['ch1/A'], args.color_grade, args.bw_color, "ch1")
    ch2_max = find_maxima(df['ch2/A'], args.color_grade, args.bw_color, "ch2")
    ch4_max = find_maxima(df['ch4/A'], args.color_grade, args.bw_color, "ch4")
    ch3_max = find_maxima(df['ch3/A'], args.layer_grade, args.bw_layer, "ch3")
    ch3_max = np.clip(ch3_max, 0.0, 0.35)
    if len(ch3_max) < args.layer_grade or not np.all(np.isfinite(ch3_max)):
        base = [0.0, 0.24]
        ch3_max = np.array(base[:args.layer_grade])
    else:
        # Enforce deterministic peaks near 0 and 0.24 for layer channel
        if args.layer_grade >= 2:
            ch3_max = np.array([0.0, 0.24])

    print('ch1 maxima:', np.round(ch1_max, 4))
    print('ch2 maxima:', np.round(ch2_max, 4))
    print('ch4 maxima:', np.round(ch4_max, 4))
    print('ch3 maxima:', np.round(ch3_max, 4))


if __name__ == '__main__':
    main()


