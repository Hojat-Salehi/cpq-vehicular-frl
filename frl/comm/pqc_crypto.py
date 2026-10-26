from pathlib import Path
import csv
from typing import Any, Dict, Tuple, List, Optional
import time
import json
import os
import io
import numpy as np
import oqs
import struct

DEFAULT_SIG_SCHEME = "ML-DSA-44"

# LOGGING
RUN_ID = os.environ.get("FL_RUN_ID", time.strftime("run_%Y%m%d_%H%M%S"))
LOG_ROOT = Path("pqc_logs") / RUN_ID
LOG_ROOT.mkdir(parents=True, exist_ok=True)
PQC_CLIENT_LOG = LOG_ROOT / "pqc_client_metrics.csv"

def _append_client_metrics(row: dict) -> None:
    file_exists = PQC_CLIENT_LOG.exists()
    with PQC_CLIENT_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts",
                "veh",
                "round",
                "direction",
                "layers",
                "wire_bytes",
                "model_bytes",
                "payload_bytes",
                "wire_over_model",     # clearer than "compression_ratio"
                "payload_over_model",  # also useful
                "duration_ms",
                "scheme",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# KEY MANAGEMENT
def load_or_create_keypair(veh_id: str, scheme: str, key_dir: str) -> Tuple[bytes, bytes]:
    os.makedirs(key_dir, exist_ok=True)
    pk_path = os.path.join(key_dir, f"{veh_id}_{scheme}.pk")
    sk_path = os.path.join(key_dir, f"{veh_id}_{scheme}.sk")
    if os.path.exists(pk_path) and os.path.exists(sk_path):
        with open(pk_path, "rb") as f:
            pk = f.read()
        with open(sk_path, "rb") as f:
            sk = f.read()
        return pk, sk

    with oqs.Signature(scheme) as signer:
        pk = signer.generate_keypair()
        sk = signer.export_secret_key()

    with open(pk_path, "wb") as f:
        f.write(pk)
    with open(sk_path, "wb") as f:
        f.write(sk)

    return pk, sk


# SERIALIZATION
def compute_model_size_bytes(weights: List[np.ndarray]) -> int:
    return int(sum(int(w.nbytes) for w in weights))


def serialize_weights(weights: List[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, *weights)
    return buf.getvalue()

def deserialize_weights(payload: bytes) -> List[np.ndarray]:
    buf = io.BytesIO(payload)
    npz = np.load(buf, allow_pickle=False)

    def _idx(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 0

    keys = sorted(list(npz.files), key=_idx)
    return [npz[k] for k in keys]

# FORMAT WIRE: [u32 header_len][header_json][u32 sig_len][sig][payload]
def build_signed_packet(
    header_dict: Dict[str, Any],
    payload: bytes,
    sk: bytes,
    scheme: str,
) -> bytes:
    header_json = json.dumps(header_dict, separators=(",", ":"), sort_keys=True)
    header_bytes = header_json.encode("utf-8")

    with oqs.Signature(scheme, secret_key=sk) as signer:
        signature = signer.sign(header_bytes + payload)

    pkt = bytearray()
    pkt += struct.pack("!I", len(header_bytes))
    pkt += header_bytes
    pkt += struct.pack("!I", len(signature))
    pkt += signature
    pkt += payload
    return bytes(pkt)


def _split_packet(packet: bytes) -> Tuple[bytes, bytes, bytes]:
    """
    Safe split with bounds checks to avoid truncated / malformed packets.
    """
    if len(packet) < 4:
        raise ValueError("packet too short for header length")

    off = 0
    (hdr_len,) = struct.unpack_from("!I", packet, off)
    off += 4

    if hdr_len < 2 or hdr_len > 10_000_000:
        raise ValueError(f"unreasonable header length: {hdr_len}")
    if len(packet) < off + hdr_len:
        raise ValueError("packet truncated in header")

    header_bytes = packet[off : off + hdr_len]
    off += hdr_len

    if len(packet) < off + 4:
        raise ValueError("packet too short for signature length")

    (sig_len,) = struct.unpack_from("!I", packet, off)
    off += 4

    if sig_len < 1 or sig_len > 10_000_000:
        raise ValueError(f"unreasonable signature length: {sig_len}")
    if len(packet) < off + sig_len:
        raise ValueError("packet truncated in signature")

    sig = packet[off : off + sig_len]
    off += sig_len

    payload = packet[off:]
    if len(payload) == 0:
        raise ValueError("empty payload")

    return header_bytes, sig, payload


def verify_and_unpack_from_parts(
    header_bytes: bytes,
    sig: bytes,
    payload: bytes,
    public_key: bytes,
    scheme: str,
) -> Tuple[Dict[str, Any], bytes]:
    with oqs.Signature(scheme) as verifier:
        ok = verifier.verify(header_bytes + payload, sig, public_key)
    if not ok:
        raise ValueError("PQC signature verification failed")

    header_json = header_bytes.decode("utf-8")
    header_dict = json.loads(header_json)
    return header_dict, payload


def verify_and_unpack_packet(
    packet: bytes,
    key_dir: str = "pqc_keys",
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    header_bytes, sig, payload = _split_packet(packet)

    # Parse header once to find identities for key lookup
    header_json = header_bytes.decode("utf-8")
    header_pre = json.loads(header_json)

    veh = str(header_pre.get("veh", header_pre.get("veh_id", "veh_unknown")))
    sender = str(header_pre.get("sender", veh))
    scheme = str(header_pre.get("sig_scheme", DEFAULT_SIG_SCHEME))

    # Load sender public key
    pk_path = os.path.join(key_dir, f"{sender}_{scheme}.pk")
    if not os.path.exists(pk_path):
        raise ValueError(
            f"Public key not found for sender={sender}, scheme={scheme} at {pk_path} "
            f"(veh={veh})"
        )

    with open(pk_path, "rb") as f:
        pk = f.read()

    header, payload_out = verify_and_unpack_from_parts(
        header_bytes=header_bytes,
        sig=sig,
        payload=payload,
        public_key=pk,
        scheme=scheme,
    )

    weights = deserialize_weights(payload_out)
    return header, weights



# PQCCrypto: client encrypt/sign
class PQCCrypto(object):
    def __init__(self, config: Dict[str, Any]):
        self.scheme = config.get("scheme", DEFAULT_SIG_SCHEME)
        self.key_dir = config.get("key_dir", "pqc_keys")
        self.rsu_id = str(config.get("rsu_id", "rsu"))
        print(f"[PQC Enabled] scheme={self.scheme}, key_dir={self.key_dir}")
        self._key_cache: Dict[str, Tuple[bytes, bytes]] = {}

        try:
            self._get_keys_for_identity(self.rsu_id)
            print(f"[PQC] RSU keypair ready: {self.rsu_id}_{self.scheme}.pk/.sk", flush=True)
        except Exception as e:
            print(f"[PQC] WARNING: failed to init RSU keypair: {e}", flush=True)

    def _get_keys_for_identity(self, identity: str) -> Tuple[bytes, bytes]:
        identity = str(identity)
        if identity in self._key_cache:
            return self._key_cache[identity]
        pk, sk = load_or_create_keypair(identity, self.scheme, self.key_dir)
        self._key_cache[identity] = (pk, sk)
        return pk, sk
    
    def _get_keys_for_vehicle(self, veh_id: str) -> Tuple[bytes, bytes]:
        return self._get_keys_for_identity(veh_id)

    def _build_packet(
        self,
        veh_id: str,
        round_id: int,
        weights: List[np.ndarray],
        direction: str,
    ) -> Dict[str, Any]:
        sender = (self.rsu_id if direction == "downlink" else str(veh_id))
        pk, sk = self._get_keys_for_identity(sender)

        t0 = time.time()

        model_bytes = compute_model_size_bytes(weights)
        payload = serialize_weights(weights)
        payload_bytes = len(payload)

        header = {
            "sender": sender,      
            "veh": str(veh_id),        
            "round": int(round_id),
            "direction": direction,
            "model_bytes": int(model_bytes),
            "payload_bytes": int(payload_bytes),
            "ts": float(t0),
            "sig_scheme": self.scheme,
        }

        packet = build_signed_packet(
            header_dict=header,
            payload=payload,
            sk=sk,
            scheme=self.scheme,
        )

        t1 = time.time()
        wire_bytes = len(packet)

        wire_over_model = (wire_bytes / float(model_bytes)) if model_bytes > 0 else 0.0
        payload_over_model = (payload_bytes / float(model_bytes)) if model_bytes > 0 else 0.0

        return {
            "pk": pk,
            "header": header,
            "packet": packet,
            "model_bytes": int(model_bytes),
            "payload_bytes": int(payload_bytes),
            "wire_bytes": int(wire_bytes),
            "wire_over_model": float(wire_over_model),
            "payload_over_model": float(payload_over_model),
            "t0": float(t0),
            "t1": float(t1),
        }

    def encrypt(self, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("PQCCrypto.encrypt() expects dict with 'veh_id','round','weights'")

        veh_id = str(data.get("veh", data.get("veh_id", "veh_unknown")))
        round_id = int(data.get("round", 0))
        weights = data.get("weights", None)

        if weights is None:
            raise ValueError("PQCCrypto 'weights' missing in encrypt() data")
        for w in weights:
            if not isinstance(w, np.ndarray):
                raise TypeError("PQCCrypto: all weights must be numpy arrays")

        info = self._build_packet(
            veh_id=veh_id,
            round_id=round_id,
            weights=weights,
            direction="uplink",
        )

        print(
            f"Built packet for {veh_id} round {round_id}: "
            f"wire_bytes={info['wire_bytes']}, model_bytes={info['model_bytes']}, "
            f"wire_over_model={info['wire_over_model']:.4f}"
        )

        try:
            _append_client_metrics(
                {
                    "ts": info["t0"],
                    "veh": veh_id,
                    "round": round_id,
                    "direction": "uplink",
                    "layers": len(weights),
                    "wire_bytes": info["wire_bytes"],
                    "model_bytes": info["model_bytes"],
                    "payload_bytes": info["payload_bytes"],
                    "wire_over_model": info["wire_over_model"],
                    "payload_over_model": info["payload_over_model"],
                    "duration_ms": (info["t1"] - info["t0"]) * 1000.0,
                    "scheme": self.scheme,
                }
            )
        except Exception as e:
            print(f"WARNING: failed to log client metrics: {e}", flush=True)

        return {
            "packet": info["packet"],
            "header": info["header"],
        }

    def simulate_downlink(
        self,
        veh_id: str,
        round_id: int,
        weights: List[np.ndarray],
        rsu_id: str = "rsu",
    ) -> None:
        """
        Local test helper: build an RSU-signed downlink packet and verify it.
        """
        try:
            info = self._build_packet(
                veh_id=veh_id,
                round_id=round_id,
                weights=weights,
                direction="downlink",
            )

            # verify & unpack using RSU public key (info["pk"])
            header_bytes, sig, payload = _split_packet(info["packet"])
            header_out, payload_out = verify_and_unpack_from_parts(
                header_bytes=header_bytes,
                sig=sig,
                payload=payload,
                public_key=info["pk"],
                scheme=self.scheme,
            )
            _ = header_out
            _ = deserialize_weights(payload_out)

            _append_client_metrics(
                {
                    "ts": info["t0"],
                    "veh": veh_id,
                    "round": round_id,
                    "direction": "downlink",
                    "layers": len(weights),
                    "wire_bytes": info["wire_bytes"],
                    "model_bytes": info["model_bytes"],
                    "payload_bytes": info["payload_bytes"],
                    "wire_over_model": info["wire_over_model"],
                    "payload_over_model": info["payload_over_model"],
                    "duration_ms": (info["t1"] - info["t0"]) * 1000.0,
                    "scheme": self.scheme,
                }
            )
        except Exception as e:
            print(f"ERROR PQCCrypto.simulate_downlink for {veh_id} r{round_id}: {e}", flush=True)


    def decrypt(self, data: Any) -> Any:
        return data
