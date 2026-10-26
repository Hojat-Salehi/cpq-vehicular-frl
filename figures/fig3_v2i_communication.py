#!/usr/bin/env python3
"""Figure 3: authenticated V2I communication in E2 (802.11p, ML-DSA-44, eps = 1, 5 seeds).

(a) downlink and uplink traffic per round, (b) uplink attempt success by distance to the RSU at the
start of the attempt (attempts pooled over the seeds; bins with fewer than 30 attempts omitted),
(c) median and 90th percentile of the delivered uplink latency.

    python figures/fig3_v2i_communication.py [--output-dir figures/output]
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ['No privacy', 'CPQ', 'LDP-FL', 'Gaussian', 'Laplace']

# mean over seeds: traffic per round (KB), delivered uplink latency median / p90 (s)
TRAFFIC_KB = {  # (downlink, uplink)
    'No privacy': (12396.0, 6852.8), 'CPQ': (12482.0, 238.3), 'LDP-FL': (12514.0, 238.4),
    'Gaussian': (13010.0, 7183.6), 'Laplace': (12498.0, 6956.6)}
UPLINK_LATENCY_S = {  # (median, p90)
    'No privacy': (2.541, 4.477), 'CPQ': (0.247, 0.596), 'LDP-FL': (0.249, 0.597),
    'Gaussian': (2.683, 4.552), 'Laplace': (2.571, 4.576)}

# uplink attempts by distance at the start of the attempt: bins (m), success (%) and attempts
DIST_BINS = [(0, 50), (50, 75), (75, 100), (100, 125), (125, 150), (150, 175), (175, 200)]
UPLINK_SUCCESS = {
    'No privacy': ([100.0, 100.0, 87.7, 56.6, 2.7, 0.0, 0.0], [116, 678, 3349, 10236, 10189, 5228, 344]),
    'CPQ':        ([None, None, 100.0, 100.0, 99.4, 78.0, 10.3], [0, 0, 269, 5625, 6094, 1651, 126]),
    'LDP-FL':     ([None, 100.0, 100.0, 100.0, 99.6, 76.1, 12.7], [0, 2, 238, 5653, 6111, 1657, 110]),
    'Gaussian':   ([100.0, 100.0, 84.9, 53.0, 3.0, 0.0, 0.0], [131, 759, 3541, 11490, 10419, 4878, 316]),
    'Laplace':    ([100.0, 100.0, 86.5, 55.3, 3.2, 0.0, 0.0], [160, 779, 3443, 10791, 10291, 4732, 315]),
}

STYLE = {
    'No privacy': dict(color='#4D4D4D', marker='s', linestyle='--'),
    'CPQ': dict(color='#0072B2', marker='o', linestyle='-'),
    'LDP-FL': dict(color='#CC79A7', marker='D', linestyle='--'),
    'Gaussian': dict(color='#D55E00', marker='s', linestyle='-.'),
    'Laplace': dict(color='#009E73', marker='^', linestyle=':'),
}


def setup_plotting():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 8.2, 'axes.titlesize': 9,
        'axes.labelsize': 8.3, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.4, 'axes.linewidth': 0.7, 'lines.linewidth': 1.75,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def cleanup(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='out', length=2.5, width=0.65)
    ax.grid(axis='y', color='#E3E6E9', linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def legend_methods():
    return [Line2D([0], [0], color=STYLE[m]['color'], linestyle=STYLE[m]['linestyle'],
                   marker=STYLE[m]['marker'], markersize=4, markerfacecolor='white',
                   linewidth=1.6, label=m) for m in METHODS]


def figure3():
    fig = plt.figure(figsize=(7.15, 2.73))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.16, 1.62, 1.08], wspace=.49,
                          left=.073, right=.985, bottom=.275, top=.84)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    xx = np.arange(len(METHODS))

    for i, m in enumerate(METHODS):
        down, up = (v / 1000 for v in TRAFFIC_KB[m])
        ax1.plot([i, i], [up, down], color='#BBC1C7', linewidth=1.2, zorder=2)
        ax1.scatter([i], [down], marker='s', s=26, edgecolor='#374151', facecolor='white', linewidth=.8, zorder=3)
        ax1.scatter([i], [up], marker='o', s=27, edgecolor=STYLE[m]['color'], facecolor=STYLE[m]['color'], zorder=4)
    ax1.set_yscale('log')
    ax1.set_ylim(.15, 24.5)
    ax1.set_xlim(-.48, 4.48)
    ax1.set_xticks(xx, METHODS, rotation=36, ha='right')
    ax1.set_ylabel('Traffic per round (MB)')
    ax1.set_yticks([.2, .5, 1, 2, 5, 10, 20])
    ax1.set_yticklabels(['0.2', '0.5', '1', '2', '5', '10', '20'])
    ax1.set_title('(a) Wireless traffic', loc='left', pad=7)
    cleanup(ax1)
    ax1.text(.03, .98, '□ downlink     ● uplink', transform=ax1.transAxes, ha='left', va='top',
             fontsize=7.2, color='#535963')

    x = np.array([(lo + hi) / 2 for lo, hi in DIST_BINS], float)
    for m in METHODS:
        pct, n = UPLINK_SUCCESS[m]
        y = np.array([np.nan if p is None or k < 30 else p for p, k in zip(pct, n)], float)
        s = STYLE[m]
        ax2.plot(x, y, markersize=4, marker=s['marker'], color=s['color'], linestyle=s['linestyle'], zorder=3)
    ax2.set_title('(b) Uplink attempt success', loc='left', pad=7)
    ax2.set_xlabel('Distance at transfer start (m)')
    ax2.set_ylabel('Delivered attempts (%)')
    ax2.set_ylim(-3, 106)
    ax2.set_xlim(20, 205)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_xticks([50, 100, 150, 200])
    cleanup(ax2)

    for i, m in enumerate(METHODS):
        med, p90 = UPLINK_LATENCY_S[m]
        c = STYLE[m]['color']
        ax3.plot([i, i], [med, p90], color=c, linewidth=2.0, zorder=3)
        ax3.scatter([i], [med], marker='o', s=32, facecolor='white', edgecolor=c, linewidth=1.2, zorder=4)
        ax3.scatter([i], [p90], marker='^', s=30, facecolor=c, edgecolor=c, zorder=4)
    ax3.set_xticks(xx, METHODS, rotation=36, ha='right')
    ax3.set_xlim(-.48, 4.48)
    ax3.set_yscale('log')
    ax3.set_ylim(.15, 7)
    ax3.set_yticks([.2, .5, 1, 2, 5])
    ax3.set_yticklabels(['0.2', '0.5', '1', '2', '5'])
    ax3.set_ylabel('Delivered uplink latency (s)')
    ax3.set_title('(c) Transfer latency', loc='left', pad=7)
    ax3.text(.04, .98, '○ median     ▲ p90', transform=ax3.transAxes, va='top', fontsize=7.1, color='#535963')
    cleanup(ax3)

    fig.legend(handles=legend_methods(), loc='upper center', bbox_to_anchor=(.51, .995), ncol=5,
               frameon=False, handlelength=1.8, columnspacing=1.1, handletextpad=.4)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    p.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf', 'png'])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    fig = figure3()
    for ext in args.formats:
        path = args.output_dir / f'fig3_e2_communication.{ext}'
        fig.savefig(path, bbox_inches='tight', pad_inches=0.065, dpi=280)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
