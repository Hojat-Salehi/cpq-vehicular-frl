#!/usr/bin/env python3
"""Figures J.1, J.3 and J.4 (E1, ideal communication, 5 seeds).

Values per seed are each run's mean over rounds 91-100 (travel time and SUMO delays in s, arrival in %).

    python figures/appendix_e1_seeds.py [--figure paired seeds secondary] [--output-dir figures/output]
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

# per-seed values (seeds 1-5), key 'method|eps'
DATA = {
    'travel': {
        'No privacy|none': [78.6, 138.5, 140.9, 135.7, 65.3],
        'CPQ|0.3': [187.3, 134.1, 174.0, 177.7, 153.3],
        'LDP-FL|0.3': [173.5, 257.6, 185.2, 197.6, 219.3],
        'Laplace|0.3': [252.3, 254.3, 242.6, 245.0, 291.2],
        'Gaussian|0.3': [440.0, 437.8, 397.8, 367.4, 421.0],
        'CPQ|0.5': [235.4, 118.9, 73.5, 139.5, 122.0],
        'LDP-FL|0.5': [233.4, 193.3, 205.6, 175.3, 190.1],
        'Laplace|0.5': [202.6, 229.2, 189.3, 216.7, 237.8],
        'Gaussian|0.5': [286.1, 417.5, 325.0, 274.0, 316.4],
        'CPQ|1': [128.4, 141.7, 162.8, 178.5, 112.0],
        'LDP-FL|1': [205.2, 177.9, 180.7, 219.9, 155.0],
        'Laplace|1': [138.7, 187.0, 218.6, 201.6, 238.6],
        'Gaussian|1': [202.6, 250.9, 192.5, 288.1, 196.8],
        'CPQ|2': [85.4, 111.1, 115.7, 136.6, 144.1],
        'LDP-FL|2': [85.4, 102.7, 201.3, 136.0, 108.3],
        'Laplace|2': [146.0, 165.9, 163.3, 129.8, 137.0],
        'Gaussian|2': [203.0, 186.9, 212.1, 195.0, 133.3],
        'CPQ|5': [102.1, 142.8, 98.3, 153.3, 132.0],
        'LDP-FL|5': [96.7, 187.3, 133.3, 108.0, 166.2],
        'Laplace|5': [85.9, 131.8, 128.0, 133.2, 128.7],
        'Gaussian|5': [108.7, 139.9, 208.4, 206.4, 159.5],
    },
    'arrival': {
        'No privacy|none': [100.0, 83.2, 81.7, 84.0, 100.0],
        'CPQ|0.3': [77.1, 89.2, 70.7, 69.8, 86.6],
        'LDP-FL|0.3': [73.1, 65.7, 69.6, 78.0, 68.9],
        'Laplace|0.3': [67.1, 61.7, 63.1, 61.5, 50.2],
        'Gaussian|0.3': [18.3, 15.5, 27.5, 32.7, 19.9],
        'CPQ|0.5': [57.7, 92.7, 100.0, 86.5, 87.7],
        'LDP-FL|0.5': [62.5, 79.4, 67.3, 80.1, 67.1],
        'Laplace|0.5': [73.9, 69.9, 67.2, 67.5, 61.5],
        'Gaussian|0.5': [56.3, 23.7, 45.5, 57.9, 45.3],
        'CPQ|1': [96.6, 81.0, 77.3, 85.6, 95.0],
        'LDP-FL|1': [78.0, 85.9, 75.4, 69.1, 83.1],
        'Laplace|1': [90.9, 77.3, 64.3, 75.1, 66.4],
        'Gaussian|1': [69.7, 63.5, 80.3, 51.3, 71.7],
        'CPQ|2': [100.0, 94.1, 94.7, 81.9, 89.8],
        'LDP-FL|2': [98.7, 97.7, 77.5, 95.6, 89.7],
        'Laplace|2': [87.1, 79.3, 76.5, 86.7, 88.6],
        'Gaussian|2': [75.5, 80.2, 72.7, 78.1, 86.3],
        'CPQ|5': [99.7, 80.7, 97.1, 74.7, 86.9],
        'LDP-FL|5': [98.9, 89.1, 82.7, 99.8, 83.5],
        'Laplace|5': [100.0, 87.3, 82.4, 88.4, 88.0],
        'Gaussian|5': [94.9, 88.9, 77.9, 70.7, 79.7],
    },
    'loss': {
        'No privacy|none': [27.1, 46.9, 48.8, 46.4, 27.0],
        'CPQ|0.3': [64.0, 46.1, 44.8, 45.5, 56.6],
        'LDP-FL|0.3': [52.3, 134.2, 58.8, 88.5, 99.6],
        'Laplace|0.3': [128.3, 124.9, 108.1, 125.1, 145.7],
        'Gaussian|0.3': [277.2, 268.5, 255.3, 206.3, 252.0],
        'CPQ|0.5': [68.9, 42.9, 31.8, 44.2, 39.4],
        'LDP-FL|0.5': [91.3, 91.1, 67.0, 70.5, 56.7],
        'Laplace|0.5': [83.0, 94.0, 58.2, 97.4, 86.9],
        'Gaussian|0.5': [150.1, 233.6, 177.4, 144.4, 177.4],
        'CPQ|1': [56.1, 49.8, 48.3, 65.8, 44.5],
        'LDP-FL|1': [92.1, 79.1, 52.9, 79.5, 48.8],
        'Laplace|1': [52.6, 81.3, 77.8, 82.7, 105.3],
        'Gaussian|1': [106.1, 111.8, 78.2, 130.7, 93.0],
        'CPQ|2': [38.9, 40.5, 43.3, 44.8, 53.0],
        'LDP-FL|2': [35.8, 39.6, 67.4, 58.5, 37.4],
        'Laplace|2': [56.6, 54.0, 47.8, 43.6, 49.4],
        'Gaussian|2': [75.8, 92.8, 91.3, 82.8, 52.7],
        'CPQ|5': [48.6, 47.6, 34.0, 43.2, 46.4],
        'LDP-FL|5': [39.2, 85.8, 45.0, 51.0, 61.3],
        'Laplace|5': [40.2, 43.6, 38.5, 46.6, 44.8],
        'Gaussian|5': [40.0, 52.0, 81.5, 63.2, 45.8],
    },
    'waiting': {
        'No privacy|none': [11.1, 19.3, 20.4, 19.9, 13.0],
        'CPQ|0.3': [27.4, 18.9, 13.9, 13.7, 27.0],
        'LDP-FL|0.3': [17.4, 77.1, 24.2, 42.2, 49.9],
        'Laplace|0.3': [65.5, 65.8, 54.2, 67.8, 89.2],
        'Gaussian|0.3': [180.7, 178.4, 151.5, 128.5, 160.2],
        'CPQ|0.5': [26.5, 18.3, 15.3, 17.9, 15.3],
        'LDP-FL|0.5': [46.4, 49.6, 27.8, 29.3, 20.5],
        'Laplace|0.5': [41.6, 45.0, 21.1, 54.6, 34.0],
        'Gaussian|0.5': [91.5, 147.7, 105.6, 75.6, 99.1],
        'CPQ|1': [28.1, 21.3, 18.4, 30.2, 20.7],
        'LDP-FL|1': [48.1, 40.0, 20.0, 36.1, 19.4],
        'Laplace|1': [23.6, 43.6, 35.5, 35.2, 49.3],
        'Gaussian|1': [51.4, 59.6, 37.1, 74.0, 51.2],
        'CPQ|2': [19.7, 17.6, 18.9, 17.6, 24.1],
        'LDP-FL|2': [16.9, 18.0, 29.6, 28.6, 15.9],
        'Laplace|2': [25.7, 22.0, 17.8, 18.0, 22.0],
        'Gaussian|2': [33.2, 50.1, 46.6, 40.9, 22.9],
        'CPQ|5': [24.9, 19.4, 13.2, 14.6, 20.7],
        'LDP-FL|5': [18.2, 40.5, 18.8, 27.1, 28.6],
        'Laplace|5': [20.4, 16.9, 13.9, 19.7, 18.7],
        'Gaussian|5': [18.1, 21.9, 38.1, 23.6, 15.9],
    },
}

# mean and sample SD over the seeds (Table 2)
REPORTED = {
    'travel': {
        'CPQ|0.3': [165.3, 21.4],
        'LDP-FL|0.3': [206.6, 33.2],
        'Laplace|0.3': [257.1, 19.7],
        'Gaussian|0.3': [412.8, 30.5],
        'CPQ|0.5': [137.9, 59.7],
        'LDP-FL|0.5': [199.5, 21.8],
        'Laplace|0.5': [215.1, 19.6],
        'Gaussian|0.5': [323.8, 56.4],
        'CPQ|1': [144.7, 26.5],
        'LDP-FL|1': [187.7, 25.3],
        'Laplace|1': [196.9, 37.8],
        'Gaussian|1': [226.2, 41.8],
        'CPQ|2': [118.6, 23.1],
        'LDP-FL|2': [126.7, 45.5],
        'Laplace|2': [148.4, 15.9],
        'Gaussian|2': [186.1, 30.9],
        'CPQ|5': [125.7, 24.5],
        'LDP-FL|5': [138.3, 38.3],
        'Laplace|5': [121.5, 20.0],
        'Gaussian|5': [164.6, 43.1],
        'No privacy|none': [111.8, 36.7],
    },
    'arrival': {
        'CPQ|0.3': [78.7, 8.9],
        'LDP-FL|0.3': [71.1, 4.7],
        'Laplace|0.3': [60.7, 6.3],
        'Gaussian|0.3': [22.8, 7.1],
        'CPQ|0.5': [84.9, 16.1],
        'LDP-FL|0.5': [71.3, 8.0],
        'Laplace|0.5': [68.0, 4.5],
        'Gaussian|0.5': [45.7, 13.6],
        'CPQ|1': [87.1, 8.5],
        'LDP-FL|1': [78.3, 6.6],
        'Laplace|1': [74.8, 10.5],
        'Gaussian|1': [67.3, 10.8],
        'CPQ|2': [92.1, 6.7],
        'LDP-FL|2': [91.9, 8.7],
        'Laplace|2': [83.6, 5.4],
        'Gaussian|2': [78.5, 5.1],
        'CPQ|5': [87.8, 10.6],
        'LDP-FL|5': [90.8, 8.2],
        'Laplace|5': [89.2, 6.5],
        'Gaussian|5': [82.4, 9.5],
        'No privacy|none': [89.8, 9.4],
    },
    'loss': {
        'CPQ|0.3': [51.4, 8.6],
        'LDP-FL|0.3': [86.7, 33.1],
        'Laplace|0.3': [126.4, 13.4],
        'Gaussian|0.3': [251.9, 27.4],
        'CPQ|0.5': [45.4, 14.0],
        'LDP-FL|0.5': [75.3, 15.4],
        'Laplace|0.5': [83.9, 15.5],
        'Gaussian|0.5': [176.6, 35.3],
        'CPQ|1': [52.9, 8.3],
        'LDP-FL|1': [70.5, 18.7],
        'Laplace|1': [79.9, 18.8],
        'Gaussian|1': [104.0, 19.8],
        'CPQ|2': [44.1, 5.5],
        'LDP-FL|2': [47.7, 14.3],
        'Laplace|2': [50.3, 5.1],
        'Gaussian|2': [79.1, 16.3],
        'CPQ|5': [44.0, 5.9],
        'LDP-FL|5': [56.5, 18.4],
        'Laplace|5': [42.7, 3.3],
        'Gaussian|5': [56.5, 16.4],
        'No privacy|none': [39.2, 11.2],
    },
    'waiting': {
        'CPQ|0.3': [20.2, 6.7],
        'LDP-FL|0.3': [42.2, 23.5],
        'Laplace|0.3': [68.5, 12.7],
        'Gaussian|0.3': [159.9, 21.4],
        'CPQ|0.5': [18.7, 4.6],
        'LDP-FL|0.5': [34.7, 12.6],
        'Laplace|0.5': [39.2, 12.5],
        'Gaussian|0.5': [103.9, 26.9],
        'CPQ|1': [23.8, 5.1],
        'LDP-FL|1': [32.7, 12.7],
        'Laplace|1': [37.4, 9.7],
        'Gaussian|1': [54.7, 13.5],
        'CPQ|2': [19.6, 2.7],
        'LDP-FL|2': [21.8, 6.7],
        'Laplace|2': [21.1, 3.3],
        'Gaussian|2': [38.7, 10.9],
        'CPQ|5': [18.6, 4.7],
        'LDP-FL|5': [26.6, 9.0],
        'Laplace|5': [17.9, 2.6],
        'Gaussian|5': [23.5, 8.7],
        'No privacy|none': [16.7, 4.4],
    },
}

EPS = ('0.3','0.5','1','2','5')
METHODS = ('CPQ','LDP-FL','Gaussian','Laplace')
COLORS = {'CPQ':'#0072B2', 'LDP-FL':'#CC79A7', 'Gaussian':'#D55E00', 'Laplace':'#009E73'}
MARKERS = {'CPQ':'o','LDP-FL':'D','Gaussian':'s','Laplace':'^'}
LSTYLES = {'CPQ':'-', 'LDP-FL':'--', 'Gaussian':'-.', 'Laplace':(0,(1.2,1.4))}
REFERENCE_COLOR = '#383E48'
TCRIT4 = 2.7764451051977987


def style():
    plt.rcParams.update({
      'font.family':'DejaVu Serif', 'font.size':8,
      'axes.titlesize':8.7, 'axes.labelsize':8.2, 'axes.linewidth':.7,
      'xtick.labelsize':7.2, 'ytick.labelsize':7.2, 'legend.fontsize':7.4,
      'axes.spines.top':False, 'axes.spines.right':False,
      'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.dpi':360,
      'lines.linewidth':1.6,
    })


def series(metric, method, ep):
    return np.array(DATA[metric][f'{method}|{ep}'], float)


def meansd(metric, method, ep):
    # mean and sample SD as reported in Table 2 (computed from unrounded seed values)
    return tuple(REPORTED[metric][f'{method}|{ep}'])


def handles(with_reference=False):
    v=[Line2D([0],[0],color=COLORS[m],ls=LSTYLES[m],marker=MARKERS[m],
              lw=1.65,ms=4.5,mfc='white',mew=.9,label=m) for m in METHODS]
    if with_reference:
        v += [Line2D([0],[0],color=REFERENCE_COLOR,ls=(0,(4,2)),lw=1.2,label='No privacy')]
    return v


def grid(ax, axis='y'):
    ax.grid(axis=axis,color='#DADDE2',lw=.52)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6,width=.6)


def paired_statistics(metric, ep):
    cpq=series(metric,'CPQ',ep)
    independent=series(metric,'LDP-FL',ep)
    # A positive difference consistently means CPQ is better in either panel.
    d = independent-cpq if metric=='travel' else cpq-independent
    return d,d.mean(),TCRIT4*d.std(ddof=1)/np.sqrt(len(d))


def plot_paired():
    """Figure J.1: seed-paired CPQ benefit over LDP-FL with 95% t confidence intervals (n = 5)."""
    style()
    fig,axs=plt.subplots(1,2,figsize=(7.35,3.05),sharey=True)
    y=np.arange(len(EPS))[::-1]
    setups=[('travel','(a)  Trip-duration benefit', 'LDP-FL $-$ CPQ  (seconds)', (-110,140),[-100,-50,0,50,100]),
            ('arrival','(b)  Arrival-rate benefit','CPQ $-$ LDP-FL  (percentage points)',(-45,50),[-40,-20,0,20,40])]
    for ax,(metric,title,xlabel,lims,ticks) in zip(axs,setups):
        ax.axvspan(0,lims[1],color='#F0F6FA',zorder=0)
        ax.axvline(0,color='#67707C',lw=.9,ls=(0,(3,2)),zorder=2)
        for k,ep in enumerate(EPS):
            d,mu,ci=paired_statistics(metric,ep)
            yy=y[k]
            # Individual paired seed effects, not unpaired samples.
            ax.scatter(d,yy+np.array([-.16,-.08,0,.08,.16]),
                       s=15,color=COLORS['CPQ'],alpha=.38,linewidths=0,zorder=3)
            ax.errorbar(mu,yy,xerr=ci,fmt='o',mfc='white',mec=COLORS['CPQ'],
                        mew=1.2,color=COLORS['CPQ'],ms=6.5,
                        capsize=3,elinewidth=1.35,capthick=1,zorder=5)
        ax.set_title(title,loc='left',pad=12,fontweight='semibold')
        ax.set_xlabel(xlabel,labelpad=6)
        ax.set_xlim(*lims);ax.set_xticks(ticks)
        ax.set_ylim(-.55,4.7)
        ax.set_yticks(y,list(EPS))
        grid(ax,'x')
        ax.tick_params(axis='y',length=0)
        ax.text(.97,.97,'CPQ better  $\\rightarrow$',transform=ax.transAxes,ha='right',
                va='top',fontsize=7.1,color='#4B6172')
    axs[0].set_ylabel('Privacy budget $\\epsilon_p$')
    fig.text(.5,.955,'Dots: five seed-paired differences   |   Open circle: mean   |   Whiskers: 95% $t$ CI',
             ha='center',va='center',fontsize=7.2,color='#3F4750')
    fig.subplots_adjust(left=.085,right=.987,bottom=.22,top=.825,wspace=.23)
    return fig


def plot_seeds():
    """Figure J.3: per-seed travel time and arrival rate, with the no-privacy reference."""
    style()
    fig,axs=plt.subplots(1,2,figsize=(7.35,3.25))
    xi=np.arange(len(EPS),dtype=float)
    dx={'CPQ':-.26, 'LDP-FL':-.087, 'Gaussian':.087, 'Laplace':.26}
    for ax,metric,title,ylabel,lims in [
        (axs[0],'travel','(a)  Final trip duration','Mean trip duration (s)',(30,485)),
        (axs[1],'arrival','(b)  Successful arrivals','Arrival rate (%)',(5,105))]:
        b=series(metric,'No privacy','none')
        ax.axhspan(b.mean()-b.std(ddof=1),b.mean()+b.std(ddof=1),
                   color='#768493',alpha=.09,zorder=0)
        ax.axhline(b.mean(),color=REFERENCE_COLOR,lw=1.1,ls=(0,(4,2)),zorder=1)
        for m in METHODS:
            x=xi+dx[m]
            mu=[];sd=[]
            for j,ep in enumerate(EPS):
                obs=series(metric,m,ep)
                official_mean, official_sd=meansd(metric,m,ep)
                mu.append(official_mean);sd.append(official_sd)
                ax.scatter(x[j]+np.linspace(-.043,.043,5),obs,
                           color=COLORS[m],alpha=.33,s=11,edgecolors='none',zorder=4)
            ax.errorbar(x,mu,yerr=sd,fmt=MARKERS[m],linestyle='none',
                        mfc='white',mec=COLORS[m],mew=.95,color=COLORS[m],
                        ms=5,elinewidth=1.05,capsize=2.2,zorder=5)
        ax.set_xlim(-.55,4.55);ax.set_ylim(*lims)
        ax.set_xticks(xi,EPS)
        ax.set_xlabel('Privacy budget $\\epsilon_p$')
        ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',pad=9)
        grid(ax)
    leg=handles(True)
    fig.legend(handles=leg,loc='upper center',bbox_to_anchor=(.51,.995),
               ncol=5,frameon=False,columnspacing=1.2,handletextpad=.4,handlelength=2.2)
    fig.text(.5,.025,'Open markers: mean ± sample SD   |   Small dots: individual seed summaries   |   Gray band: no-privacy mean ± SD',
             ha='center',fontsize=7,color='#4F5760')
    fig.subplots_adjust(left=.093,right=.99,bottom=.19,top=.82,wspace=.28)
    return fig


def plot_secondary():
    """Figure J.4: SUMO time loss and waiting time, mean over seeds."""
    style()
    fig,axs=plt.subplots(1,2,figsize=(7.35,2.92))
    for i,(ax,metric,title) in enumerate(zip(axs,['loss','waiting'],
                     ['(a)  SUMO time loss (s)','(b)  SUMO waiting time (s)'])):
        a=np.array([[meansd(metric,m,ep)[0] for ep in EPS] for m in METHODS])
        norm=Normalize(vmin=0,vmax=np.ceil(a.max()/20)*20)
        im=ax.imshow(a,aspect='auto',cmap='YlGnBu',norm=norm)
        for r in range(4):
            for c in range(5):
                val=a[r,c]
                # Explicit unit shown in title and caption; no fabricated precision.
                ax.text(c,r,f'{val:.1f}',ha='center',va='center',fontsize=8.2,
                        color='white' if norm(val)>.59 else '#1E2936')
        ax.set_xticks(range(5),EPS)
        ax.set_yticks(range(4),METHODS)
        ax.set_title(title,loc='left',pad=9)
        ax.set_xlabel('Privacy budget $\\epsilon_p$')
        ax.set_xticks(np.arange(-.5,5,1),minor=True)
        ax.set_yticks(np.arange(-.5,4,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.1)
        ax.tick_params(axis='both',which='both',length=0)
        cb=fig.colorbar(im,ax=ax,fraction=.036,pad=.025)
        cb.ax.tick_params(labelsize=6.9,length=2)
        cb.outline.set_linewidth(.5)
    fig.text(.50,.047,'Cells show means over five seeds, rounds 91–100. Smaller values are better.',
             ha='center',fontsize=7.2,color='#4F5760')
    fig.subplots_adjust(left=.115,right=.98,bottom=.23,top=.86,wspace=.29)
    return fig

FIGURES={
  'paired':('E1_paired_advantage',plot_paired),
  'seeds':('E1_seed_variability',plot_seeds),
  'secondary':('E1_secondary_delays',plot_secondary),
}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--figure',nargs='+',choices=list(FIGURES),default=list(FIGURES))
    p.add_argument('--formats',nargs='+',choices=['pdf','svg','png'],default=['pdf','png'])
    p.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/'output'))
    args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    for id in args.figure:
        name,fn=FIGURES[id]
        fig=fn()
        for ext in args.formats:
            path=out/f'{name}.{ext}'
            fig.savefig(path,bbox_inches='tight',pad_inches=.09,
                        dpi=360,facecolor='white')
            print(path)
        plt.close(fig)

if __name__=='__main__':
    main()
