#!/usr/bin/env python3
"""Figure 2 (E1, ideal communication) and Figure 4 (E2, 802.11p with ML-DSA-44, eps = 1).

Each value is the mean over 5 seeds of one block of 10 rounds (rounds 1-10, ..., 91-100).

    python figures/fig2_fig4_learning.py [--output-dir figures/output] [--formats pdf png]
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EPSILONS = (0.3, 0.5, 1.0, 2.0, 5.0)

# E1: mean evaluation travel time (s) per 10-round block
E1_TRAVEL = {
    'No privacy': [347.3, 289.1, 203.4, 204.6, 185.4, 172.3, 150.5, 137.6, 127.7, 111.8],
    'CPQ': {
        0.3: [354.9, 304.5, 201.8, 204.2, 212.8, 219.7, 239.6, 190.0, 175.0, 165.3],
        0.5: [351.2, 300.7, 224.1, 202.1, 174.7, 162.6, 166.2, 167.0, 169.7, 137.9],
        1.0: [350.1, 297.1, 194.9, 184.1, 183.3, 201.2, 197.4, 187.3, 169.7, 144.7],
        2.0: [346.8, 309.5, 195.5, 184.2, 183.2, 190.6, 177.1, 151.3, 133.2, 118.6],
        5.0: [343.3, 313.3, 185.2, 187.9, 191.9, 186.8, 205.6, 186.3, 146.2, 125.7]},
    'CPQ-λ': {
        0.3: [346.0, 360.1, 262.2, 212.3, 205.2, 191.8, 204.2, 186.7, 195.1, 191.0],
        0.5: [346.3, 328.6, 237.2, 216.1, 191.3, 212.8, 207.7, 184.4, 182.5, 178.8],
        1.0: [347.3, 309.8, 204.0, 207.5, 182.6, 192.9, 203.4, 181.4, 167.9, 152.8],
        2.0: [337.9, 304.4, 187.5, 211.5, 195.8, 207.4, 206.7, 180.3, 160.4, 141.1],
        5.0: [327.9, 316.3, 204.1, 192.8, 184.0, 187.8, 205.5, 180.4, 150.8, 127.8]},
    'LDP-FL': {
        0.3: [358.3, 360.9, 318.0, 261.8, 268.7, 247.1, 259.2, 230.6, 209.2, 206.7],
        0.5: [345.5, 336.0, 276.8, 227.0, 225.1, 214.2, 227.7, 201.5, 206.1, 199.5],
        1.0: [368.0, 309.4, 226.4, 218.8, 218.7, 211.9, 183.8, 192.5, 186.7, 187.7],
        2.0: [361.4, 312.5, 208.6, 207.8, 199.9, 213.7, 185.8, 166.8, 147.3, 126.7],
        5.0: [363.0, 310.0, 199.6, 192.2, 193.5, 197.6, 187.8, 178.0, 154.7, 138.3]},
    'Gaussian': {
        0.3: [334.5, 367.2, 360.5, 352.3, 364.0, 390.6, 389.5, 385.7, 398.8, 412.8],
        0.5: [360.3, 378.2, 378.7, 359.4, 336.3, 357.2, 312.5, 331.3, 303.0, 323.8],
        1.0: [369.9, 342.9, 325.6, 293.3, 275.0, 254.1, 225.4, 224.8, 234.7, 226.2],
        2.0: [375.9, 359.9, 289.5, 224.9, 198.7, 197.9, 206.4, 209.4, 221.0, 186.1],
        5.0: [361.0, 321.3, 214.8, 210.8, 205.2, 226.2, 193.3, 200.6, 195.5, 164.6]},
    'Laplace': {
        0.3: [367.6, 385.3, 347.1, 301.9, 275.8, 280.7, 265.2, 270.7, 282.9, 257.1],
        0.5: [367.8, 338.2, 323.7, 236.9, 213.1, 213.0, 244.6, 241.8, 204.3, 215.1],
        1.0: [366.0, 326.8, 222.6, 204.5, 203.6, 206.6, 213.1, 226.8, 193.2, 196.9],
        2.0: [355.1, 319.9, 194.6, 192.6, 192.1, 196.2, 194.8, 212.8, 186.4, 148.4],
        5.0: [344.0, 306.8, 190.1, 201.6, 187.2, 186.5, 172.6, 147.4, 134.0, 121.5]},
}

# E2 (eps = 1): travel time (s) and verified updates aggregated per round (of 50) per 10-round block
E2_TRAVEL = {
    'No privacy': [368.7, 253.3, 168.6, 172.7, 156.2, 140.7, 129.6, 106.1, 123.9, 152.6],
    'CPQ':        [350.6, 304.1, 216.3, 179.9, 189.4, 164.6, 166.9, 147.7, 158.7, 167.0],
    'CPQ-λ':      [349.4, 281.7, 224.9, 207.1, 192.1, 198.1, 176.9, 186.9, 179.1, 177.2],
    'LDP-FL':     [355.2, 305.5, 231.9, 181.9, 189.8, 192.0, 185.2, 199.9, 189.4, 194.9],
    'Gaussian':   [347.7, 369.2, 350.9, 324.7, 310.5, 325.3, 286.7, 290.6, 302.7, 290.3],
    'Laplace':    [373.1, 327.5, 249.8, 225.0, 213.1, 217.1, 212.7, 195.7, 220.4, 218.4],
}
E2_AGGREGATED = {
    'No privacy': [24.3, 19.5, 20.4, 21.8, 20.6, 19.2, 18.2, 18.4, 17.0, 16.7],
    'CPQ':        [31.9, 25.9, 25.8, 27.3, 27.3, 26.2, 26.0, 25.5, 25.1, 24.0],
    'CPQ-λ':      [30.8, 25.1, 25.9, 27.1, 26.5, 25.5, 26.2, 25.4, 24.9, 23.0],
    'LDP-FL':     [31.3, 25.9, 27.2, 28.6, 27.7, 27.0, 24.9, 24.3, 25.1, 23.3],
    'Gaussian':   [24.9, 21.6, 19.8, 19.3, 18.8, 19.2, 21.3, 19.5, 21.2, 20.5],
    'Laplace':    [24.6, 19.9, 17.7, 20.1, 20.9, 21.5, 20.6, 19.7, 20.1, 19.2],
}

METHODS = ('CPQ', 'CPQ-λ', 'LDP-FL', 'Gaussian', 'Laplace')
ALL_METHODS = ('No privacy',) + METHODS
COLORS = {'No privacy': '#383E48', 'CPQ': '#0072B2', 'CPQ-λ': '#7851A9',
          'LDP-FL': '#CC79A7', 'Gaussian': '#D55E00', 'Laplace': '#009E73'}
MARKERS = {'No privacy': 's', 'CPQ': 'o', 'CPQ-λ': 'v', 'LDP-FL': 'D', 'Gaussian': 's', 'Laplace': '^'}
LINESTYLES = {'No privacy': (0, (4, 2)), 'CPQ': '-', 'CPQ-λ': (0, (4, 1.5, 1, 1.5)),
              'LDP-FL': '--', 'Gaussian': '-.', 'Laplace': (0, (1.2, 1.4))}


def load_data(path):
    """Replace the embedded numbers by those computed from runs (tools/paper_results.py)."""
    d = json.load(open(path))
    if 'fig2' in d:
        E1_TRAVEL.clear()
        E1_TRAVEL.update({m: v if m == 'No privacy' else {float(e): x for e, x in v.items()} for m, v in d['fig2'].items()})
    if 'fig4' in d:
        E2_TRAVEL.update(d['fig4']['travel'])
        E2_AGGREGATED.update(d['fig4']['aggregated'])


def setup_plotting():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'font.size': 8,
        'axes.titlesize': 8.7, 'axes.labelsize': 8.2,
        'axes.linewidth': 0.7, 'xtick.labelsize': 7.2,
        'ytick.labelsize': 7.2, 'legend.fontsize': 7.2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 360, 'lines.linewidth': 1.6,
    })


def legend_handles(methods):
    return [Line2D([0], [0], color=COLORS[m], ls=LINESTYLES[m], marker=MARKERS[m], lw=1.65, ms=4.3,
                   mfc='white' if m != 'No privacy' else COLORS[m], mew=0.8, label=m) for m in methods]


def tidy_axes(ax):
    ax.grid(axis='y', color='#DADDE2', lw=.52)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6, width=.6)


def figure2():
    fig, axs = plt.subplots(1, 5, figsize=(7.35, 2.55), sharex=True, sharey=True)
    x = np.arange(10) * 10 + 5.5          # centre of rounds 1-10, 11-20, ...
    for i, (ax, eps) in enumerate(zip(axs, EPSILONS)):
        ax.axvspan(90, 100, color='#DCE8F1', alpha=.65, zorder=0)
        ax.plot(x, E1_TRAVEL['No privacy'], color=COLORS['No privacy'],
                ls=LINESTYLES['No privacy'], lw=1.1, alpha=.85, zorder=2)
        for method in METHODS:
            ax.plot(x, E1_TRAVEL[method][eps], color=COLORS[method], ls=LINESTYLES[method],
                    lw=1.6, marker=MARKERS[method], ms=3.2, markevery=2,
                    markerfacecolor='white', markeredgewidth=.72, zorder=3)
        ax.set_title(f'({"abcde"[i]})  $\\epsilon_p={eps:g}$', loc='left', pad=6)
        ax.set_xlim(0, 101)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(60, 460)
        ax.set_yticks([100, 200, 300, 400])
        tidy_axes(ax)
    fig.legend(handles=legend_handles(METHODS + ('No privacy',)), loc='upper center',
               bbox_to_anchor=(.53, .995), ncol=6, frameon=False, fontsize=7.0,
               columnspacing=.68, handletextpad=.32, handlelength=2.0)
    fig.text(.52, .055, 'Federated aggregation round', ha='center', fontsize=8.2)
    fig.text(.012, .48, 'Mean trip duration (s)  [lower is better]',
             rotation=90, ha='center', va='center', fontsize=8.1)
    fig.subplots_adjust(left=.08, right=.992, bottom=.23, top=.77, wspace=.16)
    return fig


def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.67))
    fig.subplots_adjust(left=.083, right=.98, bottom=.21, top=.80, wspace=.27)
    x = np.arange(10) * 10 + 10           # end of each 10-round block
    panels = [(E2_TRAVEL, 'Mean travel time (s) ↓', '(a) Greedy routing evaluation'),
              (E2_AGGREGATED, 'Updates aggregated per round (of 50) ↑', '(b) Effective model contributions')]
    for ax, (data, ylabel, title) in zip(axes, panels):
        for method in ALL_METHODS:
            ax.plot(x, data[method], color=COLORS[method], marker=MARKERS[method], markersize=3.5,
                    linestyle=LINESTYLES[method], linewidth=1.8, zorder=3)
        ax.axvspan(90.5, 100.6, color='#E7EBEF', alpha=.80, zorder=0)
        ax.set_xlim(1, 101)
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.set_xlabel('Federated round')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', pad=7)
        tidy_axes(ax)
        ax.text(.98, .97, 'Final 10 rounds', ha='right', va='top',
                transform=ax.transAxes, fontsize=7, color='#626D77')
    axes[0].set_ylim(75, 405)
    axes[0].set_yticks(np.arange(100, 405, 50))
    axes[1].set_ylim(0, 42)
    axes[1].set_yticks(np.arange(0, 42, 10))
    fig.legend(handles=legend_handles(ALL_METHODS), loc='upper center', bbox_to_anchor=(.52, .99),
               ncol=6, frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=.75, handletextpad=.35)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    p.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf', 'png'])
    p.add_argument('--data', type=Path, help='figure_data.json from tools/paper_results.py (default: the numbers of the paper)')
    args = p.parse_args()
    if args.data:
        load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    for name, make in [('E1_learning_dynamics', figure2), ('fig4_e2_learning_and_participation', figure4)]:
        fig = make()
        for ext in args.formats:
            path = args.output_dir / f'{name}.{ext}'
            fig.savefig(path, bbox_inches='tight', pad_inches=.09, dpi=360, facecolor='white')
            print(path)
        plt.close(fig)


if __name__ == '__main__':
    main()
