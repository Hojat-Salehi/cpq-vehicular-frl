# Communication-Efficient Private Federated Reinforcement Learning with Post-Quantum Authentication for Vehicular Networks

Code for the paper of the same title. Vehicles learn a routing policy with federated deep Q-learning in
SUMO and exchange their models with a road-side unit (RSU). The model updates are protected with
one-bit private quantization (CPQ, CPQ-λ, LDP-FL) or with Gaussian / Laplace noise, signed with
post-quantum signatures (ML-DSA-44, Falcon-512, SLH-DSA-SHA2-128f) and, in the mobility-coupled
experiments, carried over a packet-level IEEE 802.11p channel simulated in ns-3.

![Architecture](docs/architecture.png)

## Repository layout

```
cpq-vehicular-frl/
├── run_shared.py        one federated run (entry point)
├── configs/base.yaml    configuration shared by all experiments
├── frl/                 Python package
│   ├── learning/        DQN network and agent
│   ├── env/             SUMO environment, episodes, evaluation
│   ├── privacy/         CPQ, CPQ-λ, LDP-FL (onebit.py), pairing and group-key setup, Gaussian / Laplace
│   ├── federated/       aggregation strategies and the vehicle client
│   ├── comm/            model exchange over ns-3 and PQC signatures
│   └── monitoring/      optional logs (aggregation MSE, attack data)
├── sumo/                road network, detectors, routes          (see sumo/README.md)
├── ns3/                 ns-3 program: 802.11p channel, RSU, transport (see ns3/README.md)
├── attack/              vehicle-origin inference attack
├── tools/               run_experiment.py, analyze_experiment.py
├── figures/             scripts for the figures of the paper
└── tests/
```

All commands are run from the repository root.

## Installation

Tested on Ubuntu 20.04 with Python 3.10, SUMO 1.26.0, liboqs 0.15.0 and ns-3.46.

```bash
conda create -n cpq-frl python=3.10
conda activate cpq-frl
pip install -r requirements.txt
export SUMO_HOME="$(python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')"
```

liboqs-python needs the liboqs C library:

```bash
git clone --branch 0.15.0 --depth 1 https://github.com/open-quantum-safe/liboqs
cmake -S liboqs -B liboqs/build -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$HOME/oqs_install
cmake --build liboqs/build --parallel 8 && cmake --install liboqs/build
export OQS_INSTALL_PATH=$HOME/oqs_install
export LD_LIBRARY_PATH=$HOME/oqs_install/lib:$LD_LIBRARY_PATH
```

ns-3 is only needed for the experiments over 802.11p: see [ns3/README.md](ns3/README.md). The attack
analysis and the figures use `requirements-analysis.txt` (a separate environment is fine).

Check the installation with the tests:

```bash
for t in tests/check_*.py; do python "$t" || echo "FAILED: $t"; done
```

## Reproducing the experiments

Every run is `configs/base.yaml` plus the changes listed for it in `tools/run_experiment.py`. The launcher
writes configs, logs and outputs to `experiments/final/<experiment>/`, runs 3 runs in parallel and skips
finished runs, so it can be restarted at any time.

```bash
python tools/run_experiment.py E1 --dry-run        # list the runs
python tools/run_experiment.py E1                  # run them
python tools/analyze_experiment.py experiments/final/E1_privacy_utility    # summary tables
```

| Command | In the paper | Runs | Time per run |
|---|---|---|---|
| `E1 --eps 0.3 0.5 1 2 5` | Fig. 2, Table 2: five mechanisms at ε = 0.3, 0.5, 1, 2, 5 (seeds 1–5) | 105 | 1.5 h |
| `E1L --seeds 1 2 3 4 5` | Table 2: CPQ-λ (λ = 0.75) at ε ≠ 1 (at ε = 1 it is part of `A1`) | 20 | 1.5 h |
| `A1`, `A1X` | Table 3, Fig. J.2: CPQ-λ ablation at ε = 1 with aggregation MSE | 20 + 40 | 1.5 h |
| `A2` | Table J.1: number of learning vehicles | 30 | 0.7 h |
| `E2`, `E2L` | Figs. 3–4, Table 5: 802.11p with ML-DSA-44 at ε = 1 | 25 + 5 | 3 h |
| `E3` | Table 4, Figs. I.1, K.1: signature schemes (one seed, 30 rounds) | 6 | 0.8 h |
| `ATKN` | Fig. 5: origin inference attack data (10 origins, ns-3) | 15 | 4 h |

The attack itself runs on the logged data:
`python -m attack.run --stage all --config attack/config_ns3.yaml`.

A single run can be started directly:

```bash
FL_SEED=1 FL_OUT_DIR=results/test python run_shared.py --config configs/base.yaml
```

## Figures

`figures/` holds one script per figure of the paper, with the numbers of the paper embedded:

```bash
bash figures/make_all.sh          # writes figures/output/*.pdf and *.png
```

| Script | Figures |
|---|---|
| `fig2_fig4_learning.py` | 2, 4 |
| `fig3_v2i_communication.py` | 3 |
| `fig5_origin_inference.py` | 5 |
| `appendix_e1_seeds.py` | J.1, J.3, J.4 |
| `appendix_cpq_lambda_paired.py` | J.2 |
| `appendix_e3_signatures.py` | I.1, K.1 |

## Main results

Mean travel time (s) in rounds 91–100 under ideal communication, mean ± standard deviation over 5 seeds
(no privacy: 111.8 ± 36.7):

| Mechanism | ε = 0.3 | ε = 0.5 | ε = 1 | ε = 2 | ε = 5 |
|---|---|---|---|---|---|
| CPQ | **165.3 ± 21.4** | **137.9 ± 59.7** | **144.7 ± 26.5** | **118.6 ± 23.1** | 125.7 ± 24.5 |
| CPQ-λ (λ = 0.75) | 191.0 ± 36.4 | 178.8 ± 42.7 | 152.8 ± 17.5 | 141.1 ± 41.7 | 127.8 ± 11.9 |
| LDP-FL | 206.6 ± 33.2 | 199.5 ± 21.8 | 187.7 ± 25.3 | 126.7 ± 45.5 | 138.3 ± 38.3 |
| Gaussian | 412.8 ± 30.5 | 323.8 ± 56.4 | 226.2 ± 41.8 | 186.1 ± 30.9 | 164.6 ± 43.1 |
| Laplace | 257.1 ± 19.7 | 215.1 ± 19.6 | 196.9 ± 37.8 | 148.4 ± 15.9 | **121.5 ± 20.0** |

![Travel time during training, ideal communication](docs/fig2_learning_ideal.png)

Empirical aggregation MSE of CPQ-λ at ε = 1 (×10⁻⁵): 3.717 for LDP-FL (λ = 0), 2.825 at λ = 0.5,
1.708 at λ = 0.75 and 0.145 for CPQ (λ = 1).

Over 802.11p with ML-DSA-44 at ε = 1 (5 seeds):

| Method | Travel time (s) | Arrival (%) | Uplink delivered (%) | Verified updates per round |
|---|---|---|---|---|
| No privacy | 152.6 | 80.4 | 75.7 | 19.6 |
| CPQ | 167.0 | 79.8 | 99.4 | 26.5 |
| CPQ-λ (λ = 0.75) | 177.2 | 75.8 | 99.5 | 26.1 |
| LDP-FL | 194.9 | 72.8 | 99.4 | 26.5 |
| Gaussian | 290.3 | 51.8 | 74.0 | 20.6 |
| Laplace | 218.4 | 66.4 | 75.9 | 20.4 |

![Travel time and verified updates over 802.11p](docs/fig4_learning_80211p.png)

One-bit updates reduce the signed uplink from 113.76 KB to 8.65 KB (ML-DSA-44); with Falcon-512 the CPQ
uplink is 6.89 KB and with SLH-DSA-SHA2-128f 23.34 KB. At ε = 0.3, a vehicle-origin inference attack
trained on shadow runs reaches 30% accuracy against CPQ and 22% against LDP-FL after 100 rounds, versus
91% without privacy (chance: 10%).

## Results archive

The processed results behind the paper and the raw run outputs are archived on Zenodo: DOI to be added.

## License

MIT, see [LICENSE](LICENSE).
