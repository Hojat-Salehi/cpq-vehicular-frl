# SUMO scenario

| File | Content |
|---|---|
| `net/dqn.net.xml` | Road network: 10 intersections and 30 directed edges (5.37 km) |
| `add/dqn.add.xml` | Vehicle types of the background traffic |
| `add/dqn.det.xml` | Lane-area detectors (regenerated at the start of each run) |
| `dqn.sumocfg` | SUMO configuration used by the runs |
| `rou/` | Route files; evaluation scenarios are generated here on first use |

The learning vehicles start on edges E0, E1, E3, E11 and E8 (10 each) and drive to E15. Background
traffic is generated with SUMO's `randomTrips.py` (one departure every 1.75 s).

## Installing SUMO (1.26.0)

The simplest way is the pip package, installed by `requirements.txt`:

```bash
pip install eclipse-sumo==1.26.0
export SUMO_HOME="$(python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')"
sumo --version
```

A system installation also works (see https://sumo.dlr.de/docs/Installing/), as long as `SUMO_HOME` points to
it and `sumo` is on the `PATH`. To watch a run, set `sumo.use_gui: true` in the config (needs `sumo-gui`).
