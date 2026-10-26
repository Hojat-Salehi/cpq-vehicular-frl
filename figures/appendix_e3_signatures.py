#!/usr/bin/env python3
"""Figures I.1 and K.1 (E3: ML-DSA-44, Falcon-512 and SLH-DSA-SHA2-128f over ns-3, eps = 1, one seed, 30 rounds).

I.1  uplink attempt success by distance to the RSU at the start of the attempt
K.1  vehicles that completed the CPQ group setup (verified SETUP_OK) per communication drive

    python figures/appendix_e3_signatures.py [--output-dir figures/output]
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SCHEMES = ['ML-DSA-44', 'Falcon-512', 'SLH-DSA-SHA2-128f']
SHORT = {'ML-DSA-44': 'ML-DSA-44', 'Falcon-512': 'Falcon-512', 'SLH-DSA-SHA2-128f': 'SLH-DSA-128f'}
COLORS = {'ML-DSA-44': '#24517A', 'Falcon-512': '#7B5450', 'SLH-DSA-SHA2-128f': '#427563'}
STYLES = {'ML-DSA-44': ('-', 'o'), 'Falcon-512': ('--', 's'), 'SLH-DSA-SHA2-128f': (':', '^')}

# uplink attempt success (%) in the distance bins 75-100, 100-125, 125-150, 150-175 m (attempts in brackets in
# the comments)
DIST_LABELS = ['75–100', '100–125', '125–150', '150–175']
UPLINK_SUCCESS = {
    ('ML-DSA-44', 'No privacy'): [83, 52, 3, 0],            # 209, 645, 632, 267
    ('ML-DSA-44', 'CPQ'): [100, 100, 100, 77],              # 13, 423, 384, 95
    ('Falcon-512', 'No privacy'): [90, 54, 4, 0],           # 221, 670, 653, 319
    ('Falcon-512', 'CPQ'): [100, 100, 100, 94],             # 15, 372, 359, 90
    ('SLH-DSA-SHA2-128f', 'No privacy'): [83, 47, 3, 0],    # 240, 674, 717, 334
    ('SLH-DSA-SHA2-128f', 'CPQ'): [100, 100, 25, 0],        # 78, 477, 689, 251
}

# vehicles (of 49) with a verified SETUP_OK after each drive
SETUP_OK = {
    'ML-DSA-44': [0, 0, 0, 0, 0, 0, 0, 40, 49],
    'Falcon-512': [0, 0, 0, 0, 0, 39, 49],
    'SLH-DSA-SHA2-128f': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 48, 48, 49],
}

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'axes.linewidth': 0.65,
    'xtick.labelsize': 7.4, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def save(fig, out, stem, formats):
    for ext in formats:
        path = out / f'{stem}.{ext}'
        fig.savefig(path, bbox_inches='tight', pad_inches=0.045, dpi=300)
        print(path)
    plt.close(fig)


def distance_success():
    x = np.arange(len(DIST_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.48), sharey=True)
    for ax, method, title in zip(axes, ['No privacy', 'CPQ'],
                                 ['(a) Full-precision uplink', '(b) One-bit CPQ uplink']):
        for scheme in SCHEMES:
            ls, marker = STYLES[scheme]
            ax.plot(x, UPLINK_SUCCESS[(scheme, method)], linestyle=ls, marker=marker, linewidth=1.55,
                    markersize=4.5, color=COLORS[scheme], markeredgecolor='white', markeredgewidth=.45,
                    label=SHORT[scheme], zorder=3)
        ax.set_title(title, fontsize=8.8, pad=8)
        ax.set_xticks(x, DIST_LABELS)
        ax.set_xlim(-.16, 3.16)
        ax.set_ylim(-4, 106)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.grid(axis='y', alpha=.50, linewidth=.5, color='#b8bfc6')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=2.7, width=.55)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Distance at transfer start (m)', labelpad=6)
    axes[0].set_ylabel('Successful uplink attempts (%)', labelpad=6)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.045),
               ncol=3, columnspacing=1.75, handlelength=2.35, frameon=False)
    fig.subplots_adjust(left=.089, right=.99, bottom=.23, top=.85, wspace=.13)
    return fig


def setup_progress():
    fig, ax = plt.subplots(figsize=(3.48, 2.62))
    for scheme in SCHEMES:
        counts = [0] + SETUP_OK[scheme]
        drives = list(range(len(counts)))
        ls, marker = STYLES[scheme]
        ax.step(drives, counts, where='post', linestyle=ls, linewidth=1.55,
                color=COLORS[scheme], label=f'{SHORT[scheme]}: {drives[-1]}')
        changed = [i for i in range(1, len(counts)) if counts[i] != counts[i - 1]]
        ax.scatter([drives[i] for i in changed], [counts[i] for i in changed], s=24, marker=marker,
                   color=COLORS[scheme], edgecolor='white', linewidth=.4, zorder=5)
    ax.axhline(49, color='#9099a1', linestyle=':', linewidth=.75)
    ax.axvline(20, color='#777777', linestyle=(0, (2, 3)), linewidth=.8)
    ax.text(19.68, 6, 'Abort limit', rotation=90, va='bottom', ha='right', fontsize=7, color='#656565')
    ax.set_xlim(0, 20.7)
    ax.set_ylim(-2, 54.5)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_yticks([0, 10, 20, 30, 40, 49])
    ax.set_ylabel('Vehicles completing setup (of 49)', labelpad=6)
    ax.set_xlabel('Communication drives before training', labelpad=6)
    ax.grid(axis='y', color='#d4d9df', linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center',
              bbox_to_anchor=(.52, 1.24), frameon=False, fontsize=7.05, labelspacing=.40,
              handlelength=1.8, ncol=2, columnspacing=.9)
    fig.subplots_adjust(left=.175, right=.97, bottom=.20, top=.79)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    p.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf', 'png'])
    p.add_argument('--data', type=Path, help='figure_data.json from tools/paper_results.py (default: the numbers of the paper)')
    args = p.parse_args()
    if args.data and 'appendix_e3' in json.load(open(args.data)):
        d = json.load(open(args.data))['appendix_e3']
        UPLINK_SUCCESS.update({tuple(k.split('|')): v for k, v in d['UPLINK_SUCCESS'].items()})
        SETUP_OK.update(d['SETUP_OK'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(distance_success(), args.output_dir, 'fig_e3_success_distance', args.formats)
    save(setup_progress(), args.output_dir, 'fig_e3_setup_progress', args.formats)


if __name__ == '__main__':
    main()
