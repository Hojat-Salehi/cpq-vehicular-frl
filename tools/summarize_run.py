#!/usr/bin/env python
# Text summary of one run folder: python tools/summarize_run.py <run folder> [--window 10]

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd


def _read(run, name):
    path = os.path.join(run, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _vehicle_csvs(run, kind):
    files = sorted(f for f in os.listdir(run) if f.startswith('veh') and f.endswith(f'_{kind}.csv'))
    frames = [pd.read_csv(os.path.join(run, f)) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else None


def section(out, title):
    out.write(f'\n== {title} ' + '=' * max(0, 70 - len(title)) + '\n')


def setup(out, run):
    cfg_path = os.path.join(run, 'config.yaml')
    if not os.path.exists(cfg_path):
        return
    import yaml
    c = yaml.safe_load(open(cfg_path))
    s, e, n = c.get('sumo', {}), c.get('experiment', {}), c.get('ns3', {})
    section(out, 'Setup')
    out.write(f"method {e.get('method')} | vehicles {s.get('num_clients')} | rounds {e.get('true_rounds')} x "
              f"{c.get('common', {}).get('total_episodes')} episodes | epsilon decay over "
              f"{e.get('epsilon_decay_rounds')} rounds | state {e.get('state_encoding', 'raw')}\n")
    out.write(f"traffic 1 vehicle / {s.get('background_period')} s | warm-up {s.get('warmup')} s, departures within "
              f"{s.get('depart_window')} s | trip limit {e.get('max_trip_time')} s | privacy epsilon "
              f"{c.get('privacy', {}).get('epsilon')}\n")
    if n.get('use_ns3'):
        out.write(f"ns-3: RSU {n.get('rsu_junction')}, discovery {n.get('discovery')} "
                  f"({n.get('beacon_min')} of beacons in {n.get('beacon_timeout')} s), {n.get('pqc_scheme')}, "
                  f"channel args '{n.get('fl_sim_args', '')}'\n")
    else:
        out.write('communication: ideal (no ns-3)\n')


def performance(out, run, window):
    ev = _read(run, 'global_eval.csv')
    if ev is None or ev.empty:
        return None
    base = _read(run, 'baselines.csv')
    section(out, 'Model performance (greedy global model, evaluation scenarios)')
    last = ev.tail(window)
    best = ev.loc[ev.travel_time_mean.idxmin()]
    out.write(f"rounds evaluated: {len(ev)}\n")
    out.write(f"travel time mean: round 1 {ev.travel_time_mean.iloc[0]:.1f} s -> last {window} rounds "
              f"{last.travel_time_mean.mean():.1f} s (std across rounds {last.travel_time_mean.std():.1f}); "
              f"best {best.travel_time_mean:.1f} s in round {int(best['round'])}\n")
    out.write(f"arrival rate: round 1 {ev.arrival_rate.iloc[0]:.0%} -> last {window} rounds "
              f"{last.arrival_rate.mean():.0%} (min {last.arrival_rate.min():.0%})\n")
    out.write(f"time loss {last.time_loss_mean.mean():.1f} s, waiting {last.waiting_time_mean.mean():.1f} s "
              f"(last {window} rounds)\n")
    if base is not None:
        for _, b in base.iterrows():
            gap = last.travel_time_mean.mean() / b.travel_time_mean - 1
            out.write(f"baseline {b['round']:8s}: travel {b.travel_time_mean:.1f} s, arrival {b.arrival_rate:.0%} "
                      f"-> DQN (last {window}) is {gap:+.0%}\n")
    return ev, base


def convergence(out, ev, base, window):
    section(out, 'Convergence (evaluation travel time, rolling mean over rounds)')
    roll = ev.travel_time_mean.rolling(window, min_periods=1).mean()
    final = roll.iloc[-1]
    for tol in (0.25, 0.10):
        hit = np.where(np.abs(roll - final) <= tol * final)[0]
        # first round after which the rolling mean stays within the band
        stay = next((int(ev['round'].iloc[i]) for i in hit if np.all(np.abs(roll.iloc[i:] - final) <= tol * final)), None)
        out.write(f"within {tol:.0%} of the final level ({final:.1f} s) from round {stay} on\n")
    if base is not None:
        for _, b in base.iterrows():
            beat = ev[ev.travel_time_mean <= b.travel_time_mean]
            out.write(f"rounds at or below baseline {b['round']} ({b.travel_time_mean:.1f} s): {len(beat)} of "
                      f"{len(ev)}" + (f", first in round {int(beat['round'].iloc[0])}" if len(beat) else '') + '\n')
    marks = sorted(set(np.linspace(1, len(ev), min(len(ev), 8)).astype(int)))
    out.write('rolling mean at round: ' + ', '.join(f"{int(ev['round'].iloc[i - 1])}: {roll.iloc[i - 1]:.0f} s"
                                                   for i in marks) + '\n')
    drops = ev[ev.arrival_rate < 0.9]
    out.write(f"rounds with eval arrival < 90%: {len(drops)}"
              + (f" (last {int(drops['round'].iloc[-1])})" if len(drops) else '') + '\n')


def training(out, run, window):
    tr = _vehicle_csvs(run, 'train')
    if tr is None:
        return
    section(out, 'Training episodes (exploring policies, all vehicles)')
    g = tr.groupby('FL_Round')
    per = pd.DataFrame({'arrival': g.Arrived.mean(), 'travel_med': g.Travel_Time.median(),
                        'revisits': g.Revisits.mean(), 'eps': g.Epsilon.first()})
    out.write(f"episodes logged: {len(tr)} ({tr.Vehicle.nunique()} vehicles)\n")
    out.write(f"arrival: first round {per.arrival.iloc[0]:.0%} -> last {window} rounds {per.arrival.tail(window).mean():.0%}\n")
    out.write(f"travel median: first round {per.travel_med.iloc[0]:.0f} s -> last {window} rounds "
              f"{per.travel_med.tail(window).mean():.0f} s; revisits per trip {per.revisits.iloc[0]:.1f} -> "
              f"{per.revisits.tail(window).mean():.1f}; epsilon now {per.eps.iloc[-1]:.3f}\n")
    out.write('end reasons: ' + ', '.join(f'{k} {v}' for k, v in tr.End_Reason.value_counts().items()) + '\n')


def communication(out, run, window):
    c, p = _read(run, 'comm.csv'), _read(run, 'participation.csv')
    if c is None or p is None:
        return
    section(out, 'Communication (ns-3 802.11p, PQC-signed)')
    out.write(f"rounds: {len(p)} | aggregated updates per round: mean {p.aggregated.mean():.1f}, "
              f"min {p.aggregated.min()}, max {p.aggregated.max()} | rounds with no update: "
              f"{int((p.aggregated == 0).sum())}\n")
    out.write(f"outdated vehicles (missed the downlink) per round: mean {p.outdated_vehicles.mean():.1f}\n")
    for d, sent, got in (('downlink', 'downlink_sent', 'downlink_delivered'),
                         ('uplink', 'uplink_sent', 'uplink_delivered')):
        g = c[c.Direction == d]
        ok = g[g.Outcome == 'delivered']
        if g.empty:
            continue
        goodput = ok.Wire_Bytes * 8 / ok.Duration / 1e6
        out.write(f"{d:8s}: delivery {p[got].sum()}/{p[sent].sum()} vehicles ({p[got].sum() / max(p[sent].sum(), 1):.0%}) | "
                  f"per attempt {len(ok)}/{len(g)} ({len(ok) / len(g):.0%}) | {g.Wire_Bytes.median() / 1000:.1f} KB | "
                  f"latency median {ok.Duration.median():.2f} s, p90 {ok.Duration.quantile(.9):.2f} s | "
                  f"goodput median {goodput.median():.2f} Mbit/s | queue wait mean {g.Queue_Wait.mean():.2f} s\n")
    c = c.copy()
    c['band'] = pd.cut(c.Dist_Start, [0, 50, 75, 100, 125, 150, 200, 1e9],
                       labels=['<50', '50-75', '75-100', '100-125', '125-150', '150-200', '>200'])
    t = c.groupby(['Direction', 'band'], observed=True).agg(
        attempts=('Outcome', 'size'), success=('Outcome', lambda o: (o == 'delivered').mean()),
        latency=('Duration', lambda d: d[c.loc[d.index, 'Outcome'] == 'delivered'].median()))
    out.write('by distance to the RSU at the start of the attempt (m):\n')
    for (d, b), r in t.iterrows():
        out.write(f"  {d:8s} {b:>8s}: {int(r.attempts):5d} attempts, success {r.success:4.0%}, "
                  f"latency median {r.latency:.2f} s\n")
    up = c[(c.Direction == 'uplink') & (c.Outcome == 'delivered')].groupby('Vehicle').size()
    vehs = sorted(set(c.Vehicle), key=lambda v: int(v[3:]))
    out.write('rounds aggregated per vehicle: ' + ', '.join(f'{v} {int(up.get(v, 0))}' for v in vehs) + '\n')
    out.write('outcomes: ' + ', '.join(f'{k} {v}' for k, v in c.Outcome.value_counts().items()) + '\n')


def background(out, run):
    b = _read(run, 'background.csv')
    if b is None or b.empty:
        return
    section(out, 'Background traffic in the evaluation window (non-agent vehicles)')
    labels = [l for l in b.FL_Round.unique()]
    base = b[b.FL_Round.astype(str) == '0']
    if not base.empty:
        out.write(f"no agents: {base.Vehicles.mean():.0f} vehicles, time loss {base.Time_Loss_Mean.mean():.1f} s, "
                  f"waiting {base.Waiting_Time_Mean.mean():.1f} s\n")
    for lab in [l for l in labels if not str(l).isdigit()]:
        r = b[b.FL_Round == lab]
        out.write(f"agents on {lab:8s} routes: time loss {r.Time_Loss_Mean.mean():.1f} s, waiting {r.Waiting_Time_Mean.mean():.1f} s\n")
    rounds = b[b.FL_Round.astype(str).str.isdigit() & (b.FL_Round.astype(str) != '0')]
    if not rounds.empty:
        r = rounds.assign(FL_Round=rounds.FL_Round.astype(int)).groupby('FL_Round').Time_Loss_Mean.mean()
        out.write(f"agents on DQN routes: time loss first round {r.iloc[0]:.1f} s, last rounds {r.tail(10).mean():.1f} s\n")


def timing(out, run):
    t = _read(run, 'round_times.csv')
    if t is None or t.empty:
        return
    section(out, 'Wall-clock time')
    out.write(f"{len(t)} rounds in {t.round_s.sum() / 3600:.2f} h; per round mean {t.round_s.mean():.1f} s "
              f"(training+downlink {t.train_s.mean():.1f}, uplink+aggregation {t.aggregate_s.mean():.1f}, "
              f"evaluation {t.eval_s.mean():.1f})\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', help='run directory (written by run_shared.py)')
    ap.add_argument('--window', type=int, default=10, help='rounds for "last rounds" averages')
    a = ap.parse_args()
    pd.set_option('mode.chained_assignment', None)
    out = io.StringIO()
    out.write(f'Run: {os.path.abspath(a.run)}\n')
    setup(out, a.run)
    res = performance(out, a.run, a.window)
    if res is not None:
        convergence(out, res[0], res[1], a.window)
    training(out, a.run, a.window)
    communication(out, a.run, a.window)
    background(out, a.run)
    timing(out, a.run)
    text = out.getvalue()
    print(text)
    with open(os.path.join(a.run, 'summary.txt'), 'w') as f:
        f.write(text)


if __name__ == '__main__':
    sys.exit(main())
