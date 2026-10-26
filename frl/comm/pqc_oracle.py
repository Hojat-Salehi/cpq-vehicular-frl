# pqc_oracle.py
import os
import csv
import time
import socket
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from frl.comm.pqc_crypto import verify_and_unpack_packet
from frl.utils import require_run_id

# LOGGING
RUN_ID = require_run_id()
LOG_DIR = os.path.join("pqc_logs", RUN_ID)

METRICS_CSV = os.path.join(LOG_DIR, "pqc_metrics.csv")

def _ensure_logging_setup() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "ts",
                    "veh",
                    "round",
                    "layers",
                    "wire_bytes",          # bytes received on socket
                    "model_bytes",         # sum of raw ndarray nbytes (client-side)
                    "payload_bytes",       # serialized weight payload length (client-side)
                    "wire_over_model",     # wire_bytes / model_bytes
                    "payload_over_model",  # payload_bytes / model_bytes
                    "verify_ok",
                    "uplink_latency_ms",
                    "sig_scheme",
                ]
            )


def _normalize_header(header: Dict[str, Any]) -> Dict[str, Any]:
    hdr = dict(header) if isinstance(header, dict) else {}
    if "veh" not in hdr:
        if "veh_id" in hdr:
            hdr["veh"] = hdr.get("veh_id")
        elif "vehicle" in hdr:
            hdr["veh"] = hdr.get("vehicle")
    return hdr


def _log_metrics(
    header: Dict[str, Any],
    weights: List[np.ndarray],
    wire_len: int,
    verify_ok: bool,
) -> None:
    _ensure_logging_setup()

    veh = str(header.get("veh", "") or "")
    rnd = int(header.get("round", -1))

    model_bytes = int(header.get("model_bytes", 0) or 0)
    payload_bytes = int(header.get("payload_bytes", 0) or 0)

    layers = len(weights)
    client_ts = float(header.get("ts", 0.0) or 0.0)
    now_ts = time.time()
    uplink_latency_ms = (now_ts - client_ts) * 1000.0 if client_ts > 0 else 0.0

    wire_over_model = (float(wire_len) / float(model_bytes)) if model_bytes > 0 else 0.0
    payload_over_model = (float(payload_bytes) / float(model_bytes)) if model_bytes > 0 else 0.0

    hdr_scheme = str(header.get("sig_scheme", "") or "")

    row = [
        time.time(),
        veh,
        rnd,
        layers,
        int(wire_len),
        int(model_bytes),
        int(payload_bytes),
        wire_over_model,
        payload_over_model,
        int(bool(verify_ok)),
        uplink_latency_ms,
        hdr_scheme,
    ]

    with open(METRICS_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


# WEIGHT STORAGE
def _weights_file_path(veh_id: str, round_num: int) -> Path:
    return Path(LOG_DIR) / f"weights_{veh_id}_round{round_num}.npz"

def _save_weights_atomic(header: Dict[str, Any], weights: List[np.ndarray]) -> None:
    """
    Save decrypted weights to NPZ (atomic rename to avoid partial reads).
    """
    veh = str(header.get("veh", "veh_unknown") or "veh_unknown")
    rnd = int(header.get("round", -1))

    out_path = _weights_file_path(veh, rnd)
    tmp_path = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)

    os.makedirs(out_path.parent, exist_ok=True)

    try:
        np.savez(
            str(tmp_path),
            **{f"layer_{i}": w for i, w in enumerate(weights)},
        )
        # Atomic replace on POSIX (and safe on Linux)
        os.replace(str(tmp_path), str(out_path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

def get_decrypted_update(veh_id: str, round_num: int) -> Optional[List[np.ndarray]]:
    """
    Load decrypted weights (already verified and saved) for the given
    vehicle and round. Returns list[np.ndarray] or None if not found.
    """
    path = _weights_file_path(veh_id, round_num)
    if not path.exists():
        return None

    data = np.load(path)

    def _layer_index(k: str) -> int:
        try:
            return int(k.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    keys = sorted(data.files, key=_layer_index)
    return [data[k] for k in keys]


# PACKET HANDLING
def handle_packet(data: bytes) -> Tuple[bool, Dict[str, Any], List[np.ndarray]]:
    """
    Verify + unpack PQC packet and persist metrics + weights.
    """
    wire_len = len(data)

    try:
        header, weights = verify_and_unpack_packet(data)
        header = _normalize_header(header)

        ok = True
        _log_metrics(header, weights, wire_len, verify_ok=ok)
        _save_weights_atomic(header, weights)

        return True, header, weights

    except Exception as e:
        print(f"[PQC Oracle] ERROR verifying packet: {e}", flush=True)
        _ensure_logging_setup()

        # Error row: keep column count aligned
        with open(METRICS_CSV, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    time.time(),  # ts
                    "",           # veh
                    -1,           # round
                    0,            # layers
                    wire_len,     # wire_bytes
                    0,            # model_bytes
                    0,            # payload_bytes
                    0.0,          # wire_over_model
                    0.0,          # payload_over_model
                    0,            # verify_ok
                    0.0,          # uplink_latency_ms
                    "",           # sig_scheme
                ]
            )

        return False, {}, []

# TCP LISTENER
def _read_all(conn: socket.socket, chunk_size: int = 4096, max_bytes: Optional[int] = None) -> bytes:
    """
    Read until peer closes
    """
    chunks: List[bytes] = []
    total = 0
    while True:
        buf = conn.recv(chunk_size)
        if not buf:
            break
        chunks.append(buf)
        total += len(buf)
        if max_bytes is not None and total >= max_bytes:
            break
    return b"".join(chunks)


def run_pqc_listener(host: str = "0.0.0.0", port: int = 23456) -> None:
    """
    Receives *raw PQC wire packets* forwarded by ns-3 and verifies them.
    Reply protocol:
      - "OK\n" on success
      - "BAD_SIG\n" on failure
    """
    _ensure_logging_setup()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(64)

    print(f"[PQC Oracle] Listening on {host}:{port} (RUN_ID={RUN_ID})", flush=True)

    while True:
        conn, addr = srv.accept()
        try:
            conn.settimeout(10.0)
            data = _read_all(conn)

            if not data:
                # empty connection; treat as failure but don't crash
                conn.sendall(b"BAD_SIG\n")
                continue

            ok, header, weights = handle_packet(data)
            if ok:
                print(
                    f"[PQC Oracle] OK veh={header.get('veh')} "
                    f"round={header.get('round')} layers={len(weights)} "
                    f"wire_bytes={len(data)}",
                    flush=True,
                )
                conn.sendall(b"OK\n")
            else:
                print(f"[PQC Oracle] BAD_SIG wire_bytes={len(data)}", flush=True)
                conn.sendall(b"BAD_SIG\n")

        except Exception as e:
            # Keep listener alive
            print(f"[PQC Oracle] Listener error from {addr}: {e}", flush=True)
            try:
                conn.sendall(b"BAD_SIG\n")
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


# CLIENT-SIDE API
def send_pqc_update(
    veh_id: str,
    round_num: int,
    weights: List[np.ndarray],
    crypto: Any = None,
    oracle: Any = None,
) -> None:
    if crypto is None:
        print(f"[CLIENT {veh_id}] Crypto handler is None: skipping PQC/Oracle path", flush=True)
        return

    try:
        pqc_payload = {
            "veh": str(veh_id),          # <-- matches update 2
            "round": int(round_num),
            "weights": weights,
        }

        pqc_out = crypto.encrypt(pqc_payload)
        # ADD THIS — do not modify the file structure, just a diagnostic print
        packet_candidate = pqc_out.get("packet", None)
        if packet_candidate:
            print(f"[PQC DEBUG] veh={veh_id} round={round_num} "
                f"num_weight_arrays={len(weights)} "
                f"weight_dtypes={[w.dtype for w in weights]} "
                f"weight_shapes={[w.shape for w in weights]} "
                f"total_array_bytes={sum(w.nbytes for w in weights)} "
                f"actual_packet_bytes={len(packet_candidate)}", flush=True)
        packet = pqc_out.get("packet", None)

        if packet is None:
            print(f"[CLIENT {veh_id}] PQC encrypt returned no packet", flush=True)
            return

        print(f"[CLIENT {veh_id}] Built PQC wire packet ({len(packet)} bytes)", flush=True)

        if oracle is None:
            print(f"[CLIENT {veh_id}] Oracle transport not configured; skipping SEND_UPDATE", flush=True)
            return

        ack = oracle.send_update(str(veh_id), int(round_num), packet)
        print(f"[CLIENT {veh_id}] Oracle SEND_UPDATE ACK: {ack}", flush=True)

    except Exception as e:
        print(f"[CLIENT {veh_id}] ERROR in PQC/Oracle path: {e}", flush=True)



if __name__ == "__main__":
    run_pqc_listener()
