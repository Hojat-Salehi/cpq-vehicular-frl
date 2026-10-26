# ns-3 program (`fl-sim`)

`fl-sim.cc` simulates the IEEE 802.11p link between the vehicles and the RSU: channel, beacons, and an
SR-ARQ transport that splits each signed model into 1,200-byte chunks. It runs in lockstep with SUMO:
`frl/comm/ns3_link.py` starts it, sends it the vehicle positions after every SUMO step and collects the
completed transfers. It is only needed for the experiments with `ns3.use_ns3: true` (E2, E3 and the attack).

## Build (ns-3.46)

```bash
cd ~
wget https://www.nsnam.org/releases/ns-3.46.tar.bz2
tar xjf ns-3.46.tar.bz2
cp <repo>/ns3/fl-sim.cc <repo>/ns3/CMakeLists.txt ~/ns-3.46/scratch/
cd ~/ns-3.46
./ns3 configure --build-profile=debug \
    --enable-modules="core;network;mobility;internet;wifi;applications" \
    -- -DNS3_WARNINGS_AS_ERRORS=OFF
./ns3 build fl-sim
```

The binary is `~/ns-3.46/build/scratch/ns3.46-fl-sim-debug`. If ns-3 is installed elsewhere, set
`ns3.ns3_dir` in `configs/base.yaml`. `CMakeLists.txt` links zlib, which `fl-sim.cc` uses for the chunk
CRC-32. The paper used the debug profile, built with the C++ compiler of the conda environment.

Check the build with `python tests/check_ns3_link.py` from the repository root.

## Channel

Defaults (used in the paper): 6 Mbps OFDM on a 10 MHz channel at 5.9 GHz, 20 dBm transmit power,
log-distance path loss (exponent 2.7, 46.68 dB at 1 m) with Nakagami fading, receiver sensitivity
-101 dBm and preamble-detection floor -82 dBm. Coverage ends at about 150-200 m from the RSU.

| Option | Default | Meaning |
|---|---|---|
| `--propModel` | `logdist_nakagami` | `logdist_nakagami`, `logdist`, `friis`, or `none` (lossless) |
| `--pathLossExp` / `--refLoss` | 2.7 / 46.6777 dB | log-distance parameters |
| `--preambleMinRssi` | -82 dBm | preamble-detection floor |
| `--rxSensitivity` | -101 dBm | receiver sensitivity |

Extra options are passed through `ns3.fl_sim_args` in the config.
