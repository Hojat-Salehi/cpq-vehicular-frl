#!/usr/bin/env python3
"""Figure J.2: seed-paired travel-time differences for CPQ-lambda (eps = 1, ideal communication, 10 seeds),
relative to LDP-FL (lambda = 0) and to CPQ (lambda = 1), with 95% Student-t confidence intervals.

    python figures/appendix_cpq_lambda_paired.py [--output-dir figures/output]
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LAMBDA_LEVELS = (0.25, 0.5, 0.75, 0.9, 1.0)
EXPECTED_SEEDS = 10
T975 = {9: 2.2621571628540993}   # two-sided 95% Student-t quantile, 9 degrees of freedom

# travel time (s) per seed 1-10, mean over rounds 91-100
TRAVEL = {
    'LDP-FL': [205, 178, 181, 220, 155, 107, 197, 206, 234, 154],
    0.25: [209, 137, 129, 107, 131, 227, 165, 124, 127, 132],
    0.5: [121, 184, 120, 124, 162, 126, 222, 84, 170, 136],
    0.75: [128, 167, 142, 166, 162, 125, 107, 114, 152, 161],
    0.9: [134, 212, 150, 152, 132, 153, 171, 118, 137, 171],
    1.0: [128, 142, 163, 178, 112, 154, 112, 95, 120, 175],
}


def paired_stats(values: np.ndarray, baseline: np.ndarray):
    diffs = values - baseline
    n = len(diffs)
    return diffs, float(np.mean(diffs)), float(T975[n - 1] * np.std(diffs, ddof=1) / np.sqrt(n))


def panel(ax, data, baseline, levels, name):
    ys = np.arange(len(levels))[::-1].astype(float)
    jitter = np.linspace(-0.20, 0.20, EXPECTED_SEEDS)
    for lam, y in zip(levels, ys):
        dots, mean, ci = paired_stats(data[lam], data[baseline])
        ax.scatter(dots, y+jitter, color='0.68', s=10, edgecolor='none', zorder=3)
        ax.errorbar(mean, y, xerr=ci, fmt='D', color='0.10', markersize=4.1,
                    elinewidth=1.15, capsize=2.8, capthick=1.05, zorder=4)
    ax.axvline(0, color='0.45', linestyle='--', linewidth=0.9)
    ax.set_yticks(ys, [f'{v:g}' for v in levels])
    ax.set_xlim(-145, 145)
    ax.set_ylim(-0.55, len(levels)-0.43)
    ax.set_xticks([-120,-60,0,60,120])
    ax.tick_params(axis='both', labelsize=8, length=2.7)
    ax.tick_params(axis='y', length=0)
    ax.set_title(name, fontsize=9.1, loc='left', pad=7)
    ax.grid(axis='x', color='0.92', linewidth=0.55)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('0.5')
    ax.text(-0.10, 1.01, r'$\lambda$', transform=ax.transAxes,
            fontsize=8.2, ha='right', va='bottom')


def plot(data, out: Path):
    plt.rcParams.update({
        'font.family':'serif',
        'font.serif':['STIXGeneral','Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
        'axes.unicode_minus':False,'savefig.facecolor':'white'
    })
    fig, axs = plt.subplots(2,1, figsize=(3.5,4.65), dpi=200)
    fig.subplots_adjust(left=.155,right=.974,top=.85,bottom=.108,hspace=.49)
    panel(axs[0],data,'LDP-FL',LAMBDA_LEVELS,
          r'(a) Relative to LDP-FL ($\lambda=0$)')
    panel(axs[1],data,1.0,LAMBDA_LEVELS[:-1],
          r'(b) Relative to CPQ ($\lambda=1$)')
    axs[1].set_xlabel('Paired travel-time difference (s)', fontsize=9, labelpad=5)
    fig.legend([
        Line2D([],[],marker='o',linestyle='none',color='0.68',markersize=4),
        Line2D([],[],marker='D',linestyle='none',color='0.10',markersize=4),
    ], ['Individual seeds', 'Mean and 95% CI'], loc='upper center',
       bbox_to_anchor=(.55,.985), ncol=2, frameon=False, fontsize=7.7,
       handlelength=1.2, columnspacing=1.2)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'),dpi=350)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    a = p.parse_args()
    data = {k: np.array(v, float) for k, v in TRAVEL.items()}
    plot(data, a.output_dir / 'appendix_cpq_lambda_paired')
    for base, levels in [('LDP-FL', LAMBDA_LEVELS), (1.0, LAMBDA_LEVELS[:-1])]:
        print('Relative to', base)
        for lam in levels:
            _, mean, ci = paired_stats(data[lam], data[base])
            print(f'  lambda={lam:.2f}: {mean:+.1f} +/- {ci:.1f} s (95% paired CI)')
    print('Saved', a.output_dir / 'appendix_cpq_lambda_paired.pdf')


if __name__ == '__main__':
    main()
