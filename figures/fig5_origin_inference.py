#!/usr/bin/env python3
"""Figure 5: vehicle-origin inference accuracy against the number of observed rounds (eps = 0.3, ns-3,
10 origins, 3 seeds). Accuracy (%) with 95% cluster-bootstrap confidence intervals.

    python figures/fig5_origin_inference.py                    # shadow-run attack (Figure 5)
    python figures/fig5_origin_inference.py --scenario known   # attacker knows one vehicle per origin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROUNDS = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)

# Each row: accuracy, lower CI, upper CI; all expressed as percentages.
DATA = {
    'shadow': {
        'No privacy': (
            [11, 12, 31, 46, 57, 77, 91],
            [5, 5, 21, 36, 49, 67, 87],
            [17, 20, 42, 55, 67, 85, 96],
        ),
        'CPQ': (
            [6, 11, 9, 13, 13, 15, 30],
            [2, 7, 5, 7, 7, 8, 19],
            [11, 17, 14, 20, 19, 21, 44],
        ),
        'LDP-FL': (
            [11, 11, 12, 11, 17, 11, 22],
            [6, 6, 7, 6, 11, 5, 15],
            [17, 17, 18, 17, 25, 18, 30],
        ),
    },
    'known': {
        'No privacy': (
            [11, 16, 23, 34, 40, 64, 69],
            [8, 10, 17, 26, 32, 56, 59],
            [15, 21, 30, 41, 48, 71, 77],
        ),
        'CPQ': (
            [10, 11, 14, 11, 10, 12, 13],
            [6, 7, 11, 8, 7, 8, 9],
            [14, 15, 18, 14, 14, 17, 18],
        ),
        'LDP-FL': (
            [10, 10, 12, 10, 10, 10, 12],
            [6, 7, 8, 6, 6, 6, 7],
            [15, 14, 15, 13, 14, 14, 18],
        ),
    },
}

STYLE = {
    'CPQ': {'color': '#0072B2', 'marker': 'o', 'ls': '-', 'zorder': 4},
    'LDP-FL': {'color': '#CC79A7', 'marker': 'D', 'ls': '-.', 'zorder': 3},
    'No privacy': {'color': '#4B5563', 'marker': 's', 'ls': '--', 'zorder': 2},
}


def plot_attack(scenario: str, output_dir: Path) -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8.1,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7.7,
        'ytick.labelsize': 7.7,
        'legend.fontsize': 7.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.65,
        'savefig.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.65), dpi=200)
    # Room for a compact above-plot legend and the x-axis label.
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.20, top=0.83)

    # Draw intervals first so that lines, markers remain visible on top.
    for name in ('No privacy', 'LDP-FL', 'CPQ'):
        avg, lo, hi = (np.asarray(a, dtype=float) for a in DATA[scenario][name])
        st = STYLE[name]
        ax.fill_between(ROUNDS, lo, hi, color=st['color'], alpha=0.105,
                        linewidth=0, zorder=1)
        ax.plot(ROUNDS, avg, color=st['color'], linestyle=st['ls'],
                linewidth=1.55, marker=st['marker'], markersize=3.55,
                markerfacecolor='white', markeredgewidth=0.95,
                label=name, zorder=st['zorder'])

    ax.axhline(10, color='#8A939C', linewidth=0.9,
               linestyle=(0, (1.2, 2.3)), zorder=0)
    # Keep chance-level label below the horizontal line, away from the curves.
    ax.text(99, 5.2, 'Chance (10%)', ha='right', va='center',
            color='#58616A', fontsize=7.2)

    # A log axis separates the early (1, 2, 5) measurement points without
    # falsely implying that we collected results at intervening rounds.
    ax.set_xscale('log', base=10)
    ax.set_xlim(0.9, 115)
    ax.xaxis.set_major_locator(FixedLocator(ROUNDS))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '2', '5', '10', '20', '50', '100']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Federated rounds observed (log scale)', labelpad=5)
    ax.set_ylabel('Origin inference accuracy (%)', labelpad=4)

    ax.grid(axis='y', color='#DEE2E6', linestyle='-', linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', direction='out', length=2.6,
                   width=0.65, pad=2.2)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(0.65)

    # Consistent ordering and a single-row legend.
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ('CPQ', 'LDP-FL', 'No privacy')]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='lower center', bbox_to_anchor=(0.5, 1.04),
              ncol=3, frameon=False, handletextpad=0.35,
              columnspacing=1.15, handlelength=2.2, borderaxespad=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f'fig_origin_inference_{scenario}'
    fig.savefig(base.with_suffix('.pdf'), dpi=300)
    fig.savefig(base.with_suffix('.png'), dpi=450)
    plt.close(fig)
    print('Saved:', base.with_suffix('.pdf'))
    print('Saved:', base.with_suffix('.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario', choices=['shadow', 'known'], default='shadow',
                        help='shadow: no known target vehicles (main paper); '
                             'known: one known vehicle per origin (appendix)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'output')
    args = parser.parse_args()
    plot_attack(args.scenario, args.output_dir)
