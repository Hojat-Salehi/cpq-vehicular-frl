# ns-3 Oracle Integration

This folder contains the ns-3 side of the FLDQN + PQC oracle experiment
(802.11p V2I channel + SR-ARQ fragmentation, see `fl-sim.cc`).

Tested with ns-3.46, RealtimeSimulatorImpl, 802.11p ad-hoc, gcc 15 from the `sumo_env` conda env.

## Build

```bash
conda activate sumo_env
cd ~ && wget https://www.nsnam.org/releases/ns-3.46.tar.bz2 && tar xjf ns-3.46.tar.bz2
cp <project>/ns3/fl-sim.cc <project>/ns3/CMakeLists.txt ~/ns-3.46/scratch/
cd ~/ns-3.46
./ns3 configure --build-profile=debug \
  --enable-modules="core;network;mobility;internet;wifi;applications" \
  -- -DNS3_WARNINGS_AS_ERRORS=OFF
./ns3 build fl-sim
```

`CMakeLists.txt` links zlib (`z`), which `fl-sim.cc` needs for chunk CRC-32.

## Run

```bash
./ns3 run "fl-sim --numVehicles=10"
```

Start `pqc_oracle.py` (port 23456) first; Python clients talk to the control plane on port 12345.

## Channel options

| Flag | Default | Meaning |
|---|---|---|
| `--propModel` | `logdist_nakagami` | `logdist_nakagami`, `logdist`, `friis`, or `none` (no attenuation, sanity check) |
| `--pathLossExp` / `--refLoss` | 2.7 / 46.6777 dB | log-distance parameters |
| `--preambleMinRssi` | -82 dBm | ns-3 preamble-detection floor (frames below this are not received) |
| `--rxSensitivity` | -101 dBm | PHY RX sensitivity |
| `--areaHalf` | 250 m | random-waypoint area is `[-a, a]^2` around the RSU |
| `--populateArp` | true | pre-filled ARP caches (see below) |
| `--logMobility` | true | per-second `[MOB]` position lines |

Two channel profiles were used for the PQC analysis:

- **Edge-limited** (defaults): coverage ends around 150-200 m, so packet size decides delivery.
- **Good coverage**: `--preambleMinRssi=-92 --rxSensitivity=-92` (typical 802.11p receiver
  sensitivity at 6 Mbps); almost everything is delivered, so packet size shows up as latency.

`--populateArp=true` is on by default because 802.11p V2X messaging has no ARP, and ns-3's ARP
marks an unreachable RSU as dead for 100 s. Without it, a vehicle that once failed far from
the RSU kept failing even at 30 m.

`ns3_wave_metrics.csv` (written under `pqc_logs/<FL_RUN_ID>/` relative to the ns-3 working dir)
includes `dist_m`, and `GET_STATUS,<veh>` returns `OK,<distance_m>`.

## Transport results (2026-10-08, 10 vehicles x 5 uplinks per cell, no FL training)

Real PQC-signed packets sent through `fl-sim` + `pqc_oracle.py`. `cpq` = bit-packed one-bit
update, `fp32` = full-precision float32 model. RTT is for delivered packets.

| Profile | Scheme | Update | Wire bytes | Chunks | Delivery | Median RTT | P90 RTT |
|---|---|---|---|---|---|---|---|
| Edge-limited | Falcon-512 | cpq | 6,891 | 6 | 92% | 103 ms | 584 ms |
| Edge-limited | ML-DSA-44 | cpq | 8,658 | 8 | 82% | 119 ms | 480 ms |
| Edge-limited | SLH-DSA-SHA2-128f | cpq | 23,339 | 20 | 56% | 162 ms | 813 ms |
| Edge-limited | Falcon-512 | fp32 | 111,951 | 94 | 32% | 919 ms | 3,110 ms |
| Edge-limited | ML-DSA-44 | fp32 | 113,718 | 95 | 34% | 906 ms | 3,315 ms |
| Edge-limited | SLH-DSA-SHA2-128f | fp32 | 128,397 | 107 | 30% | 827 ms | 2,809 ms |
| Good coverage | Falcon-512 | cpq | 6,896 | 6 | 100% | 36 ms | 82 ms |
| Good coverage | ML-DSA-44 | cpq | 8,658 | 8 | 100% | 42 ms | 76 ms |
| Good coverage | SLH-DSA-SHA2-128f | cpq | 23,339 | 20 | 100% | 89 ms | 242 ms |
| Good coverage | Falcon-512 | fp32 | 111,953 | 94 | 100% | 624 ms | 1,369 ms |
| Good coverage | ML-DSA-44 | fp32 | 113,717 | 95 | 100% | 655 ms | 1,537 ms |
| Good coverage | SLH-DSA-SHA2-128f | fp32 | 128,399 | 107 | 96% | 663 ms | 1,411 ms |

With the lossless `--propModel=none` check, ML-DSA-44 cpq was delivered 50/50 (median 28 ms),
so the losses above come from the channel, not the SR-ARQ transport.
