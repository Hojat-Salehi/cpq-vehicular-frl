import flwr as fl
from typing import Dict, Optional, Tuple
import tensorflow as tf
from frl.learning.dqnagent import dqnAgent
from frl.learning.dqn import DQN
import os
import time
import csv
import numpy as np

from frl.comm.pqc_oracle import get_decrypted_update
from frl.comm.pqc_crypto import PQCCrypto
from frl.comm.oracle_transport import OracleTransport
from frl.utils import require_run_id, client_veh_id, ns3_drop_undelivered

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RUN_ID = require_run_id()
print(f"[SERVER] FL_RUN_ID={RUN_ID}", flush=True)
PQC_LOG_DIR = os.path.join("pqc_logs", RUN_ID)
os.makedirs(PQC_LOG_DIR, exist_ok=True)

E2E_LOG = os.path.join(PQC_LOG_DIR, "fl_round_e2e.csv")
_round_start_ts: Dict[int, float] = {}
_round_e2e_written: Dict[int, bool] = {}
if not os.path.exists(E2E_LOG):
    with open(E2E_LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "round",
                "t_round_start",
                "t_round_end",
                "round_wall_ms",
                "num_results",
            ]
        )

model = DQN(3)

# --- Strategy helpers ---
def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {"round": server_round}

def get_eval_fn():
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        print(f"[server] evaluate at round {server_round}")
        return 0.0, {"accuracy": 0.0}
    return evaluate


# PQC helpers
def _resolve_veh_id(client, fit_res, idx: int) -> str:
    metrics = getattr(fit_res, "metrics", {}) or {}
    veh = metrics.get("veh", None)
    if veh:
        return str(veh)
    cid = getattr(client, "cid", None)
    if cid:
        return str(cid)

    return f"veh{idx}"


def _wait_for_pqc_npz(veh_id: str, round_num: int, timeout_s: float = 1.0) -> Optional[list]:
    """
    Avoid race...
    """
    deadline = time.time() + max(0.0, float(timeout_s))
    while True:
        w = get_decrypted_update(veh_id, round_num)
        if w is not None:
            return w
        if time.time() >= deadline:
            return None
        time.sleep(0.05)

class PqcFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crypto: Optional[PQCCrypto] = None
        self.oracle: Optional[OracleTransport] = None
        # Stable cid -> vehK mapping
        self._cid_to_veh: Dict[str, str] = {}
        self._next_veh = 0
        # Enable
        pqc_enabled = os.environ.get("FL_PQC_ENABLED", "0") == "1"
        oracle_enabled = os.environ.get("FL_ORACLE_ENABLED", "0") == "1"
        if pqc_enabled:
            self.crypto = PQCCrypto(
                {
                    "scheme": os.environ.get("FL_PQC_SCHEME", "ML-DSA-44"),
                    "key_dir": os.environ.get("FL_PQC_KEY_DIR", "pqc_keys"),
                }
            )
        if oracle_enabled:
            self.oracle = OracleTransport(
                {
                    "host": os.environ.get("FL_ORACLE_HOST", "127.0.0.1"),
                    "port": int(os.environ.get("FL_ORACLE_PORT", "12345")),
                    # server only SEND_GLOBAL; clients handle downlink receive
                    "downlink": {"enabled": False},
                }
            )
        if self.crypto is not None and self.oracle is not None:
            print("[SERVER] Downlink over ns-3 oracle ENABLED (SEND_GLOBAL)", flush=True)
        else:
            print("[SERVER] Downlink over ns-3 oracle disabled", flush=True)
    def configure_fit(self, server_round, parameters, client_manager):
        # start at configure_fit (true round start)
        _round_start_ts[server_round] = time.time()
        _round_e2e_written[server_round] = False
        fit_configs = super().configure_fit(server_round, parameters, client_manager)
        # Send actual downlink packets via ns-3 oracle
        if self.crypto is not None and self.oracle is not None:
            try:
                global_weights = fl.common.parameters_to_ndarrays(parameters)
            except Exception as e:
                print(f"[SERVER] ERROR converting parameters_to_ndarrays for downlink: {e}", flush=True)
                return fit_configs

            for idx, (client, fit_ins) in enumerate(fit_configs):
                cid = getattr(client, "cid", None)
                if cid is None:
                    cid = str(idx)

                veh_id = client_veh_id(client, self._cid_to_veh)

                try:
                    if isinstance(fit_ins.config, dict):
                        fit_ins.config["round"] = server_round
                        fit_ins.config["client_idx"] = idx
                        fit_ins.config["veh"] = veh_id
                except Exception:
                    pass

                try:
                    info = self.crypto._build_packet(
                        veh_id=veh_id,
                        round_id=server_round,
                        weights=global_weights,
                        direction="downlink",
                    )
                    pkt = info["packet"]
                    ack = self.oracle.send_global(veh_id, server_round, pkt)
                    print(f"[SERVER] SEND_GLOBAL veh={veh_id} r={server_round} bytes={len(pkt)} ack={ack}", flush=True)
                except Exception as e:
                    print(f"[SERVER] SEND_GLOBAL failed veh={veh_id} r={server_round}: {e}", flush=True)
        return fit_configs

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        print(f"\n[SERVER] aggregate_fit (PQC-aware), round {server_round}")
        if not results:
            t1 = time.time()
            t0 = _round_start_ts.get(server_round, t1)
            if not _round_e2e_written.get(server_round, False):
                with open(E2E_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([server_round, t0, t1, (t1 - t0) * 1000.0, 0])
                _round_e2e_written[server_round] = True
            return None, {}

        weights_results = []
        used_pqc = 0
        missing_pqc = 0

        for idx, (client, fit_res) in enumerate(results):
            cid = getattr(client, "cid", None)
            veh_id = None
            if cid is not None and hasattr(self, "_cid_to_veh"):
                veh_id = self._cid_to_veh.get(str(cid), None)
            if not veh_id:
                veh_id = _resolve_veh_id(client, fit_res, idx)

            # pqc_weights = _wait_for_pqc_npz(veh_id, server_round, timeout_s=1.0)
            pqc_weights = None
            if self.crypto is not None and self.oracle is not None:
                pqc_weights = _wait_for_pqc_npz(veh_id, server_round, timeout_s=1.0)

            if pqc_weights is not None:
                weights = pqc_weights
                used_pqc += 1
            else:
                missing_pqc += 1
                if self.crypto is not None and self.oracle is not None and ns3_drop_undelivered():
                    continue  # uplink not delivered over ns-3: excluded from this round
                # fallback to gRPC parameters if PQC not present
                weights = fl.common.parameters_to_ndarrays(fit_res.parameters)

            weights_results.append((weights, fit_res.num_examples))

        dropping = self.crypto is not None and self.oracle is not None and ns3_drop_undelivered()
        print(
            f"[SERVER] PQC-aware FedAvg: used_pqc={used_pqc}, missing={missing_pqc} "
            f"({'dropped' if dropping else 'gRPC fallback'})"
        )
        if not weights_results:
            print("[SERVER] No delivered updates this round; global model unchanged")
            return None, {"used_pqc": 0.0, "missing_pqc": float(missing_pqc)}

        num_examples_total = sum(num_examples for _, num_examples in weights_results)
        if num_examples_total == 0:
            print("[SERVER] WARNING: num_examples_total == 0 in aggregate_fit")
            return None, {}

        first_weights, _ = weights_results[0]
        agg_weights = [np.zeros_like(w, dtype=w.dtype) for w in first_weights]

        for weights, num_examples in weights_results:
            factor = num_examples / num_examples_total
            for i, w in enumerate(weights):
                agg_weights[i] += w * factor

        aggregated_parameters = fl.common.ndarrays_to_parameters(agg_weights)

        t1 = time.time()
        t0 = _round_start_ts.get(server_round, t1)
        if not _round_e2e_written.get(server_round, False):
            with open(E2E_LOG, "a", newline="") as f:
                csv.writer(f).writerow([server_round, t0, t1, (t1 - t0) * 1000.0, len(results)])
            _round_e2e_written[server_round] = True

        return aggregated_parameters, {
            "used_pqc": float(used_pqc),
            "missing_pqc": float(missing_pqc),
        }


def _read_num_clients(cfg_path='configs/base.yaml'):
    try:
        import yaml
        with open(cfg_path) as f:
            c = yaml.safe_load(f)
        return int(c.get('sumo', {}).get('num_clients', 10))
    except Exception:
        return 10

_NUM_CLIENTS = _read_num_clients()


strategy = PqcFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=_NUM_CLIENTS,
    min_evaluate_clients=0,
    min_available_clients=_NUM_CLIENTS,
    on_fit_config_fn=fit_config,
    evaluate_fn=get_eval_fn(),
)

if __name__ == "__main__":
    SERVER_ADDRESS = "0.0.0.0:8080"

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=50),
        grpc_max_message_length=256 * 1024 * 1024,
        strategy=strategy,
    )

# def fit_config(server_round: int):
#     """Return training configuration dict for each round.

#     passes the current round number to the client
#     """
#     config = {
#         "round": server_round,
#     }
#     return config

# def get_eval_fn(model):
#     # The `evaluate` function will be called after every round
#     def evaluate(
#         server_round: int,
#         parameters: fl.common.NDArrays,
#         config: Dict[str, fl.common.Scalar],
#     ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#         # model.set_weights(parameters)  # Update model with the latest parameters
#         # filename = f"round_{server_round}.h5"
#         # server_model_path = os.path.join(os.getcwd(),'server_model',filename)
#         # model.save_weights(server_model_path)
#         print("+++++++++++ Inside eval function server")
#         return 0.0, {"accuracy": 0.0}
#     return evaluate

# strategy=fl.server.strategy.FedAvg(
# fraction_fit=1.0,
# min_fit_clients=10,
# min_available_clients=10,
# on_fit_config_fn=fit_config,
# evaluate_fn=get_eval_fn(model=model)

# )

# fl.server.start_server(config=fl.server.ServerConfig(num_rounds=500),
#      strategy=strategy,
#      )
 