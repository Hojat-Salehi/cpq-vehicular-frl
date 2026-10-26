import socket
import re
import threading
import time
import os
import csv
import json
import struct
from typing import Any, Dict, Optional, Tuple, List
from frl.utils import require_run_id

# LOGGING
RUN_ID = require_run_id()
LOG_DIR = os.path.join("pqc_logs", RUN_ID)
ORACLE_TRANSPORT_LOG = os.path.join(LOG_DIR, "pqc_oracle_transport.csv")

def _ensure_log_header() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(ORACLE_TRANSPORT_LOG):
        with open(ORACLE_TRANSPORT_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts",          # real time at send
                    "veh",
                    "round",
                    "size_bytes",
                    "ack",         # OK / BAD_FWD / DROP / NO_ACK / ERROR / NO_DATA
                    "rtt_ms",      # end-to-end RTT (client <-> oracle)
                ]
            )

def _veh_index_from_id(veh_id: str) -> int:
    """
    Accepts:
      - "veh0", "veh12"
      - "0", "12" (if someone passes raw index)
    Returns integer vehicle index.
    """
    s = str(veh_id).strip()
    m = re.fullmatch(r"veh(\d+)", s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    raise ValueError(f"Invalid veh_id '{veh_id}' (expected 'vehN' or 'N')")

def _log_oracle_rtt(
    veh_id: str,
    round_num: int,
    size: int,
    ack: str,
    t_start: float,
    t_end: float,
) -> None:
    try:
        _ensure_log_header()
        rtt_ms = (t_end - t_start) * 1000.0
        with open(ORACLE_TRANSPORT_LOG, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    t_start,
                    veh_id,
                    round_num,
                    size,
                    ack,
                    rtt_ms,
                ]
            )
    except Exception as e:
        # Logging must never break the transport
        print(f"WARNING: failed to log RTT metrics: {e}", flush=True)

# SIMULATE DOWNLINK
class DownlinkListener:
    """
    Receives downlink packets forwarded from ns-3 via TCP.

    CURRENT ns-3 behavior:
      - One TCP connection per delivery
      - Payload = raw PQC wire bytes only
      - Sender closes after sending payload
    """

    def __init__(self, listen_host: str, listen_port: int, veh_id: str):
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.veh_id = str(veh_id)

        self._stop = False
        self._t: Optional[threading.Thread] = None

        # key: (veh_id, round) -> (payload_bytes, ts)
        self._buf: Dict[Tuple[str, int], Tuple[bytes, float]] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._t is not None:
            return
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop = True

    def pop(self, veh_id: str, round_num: int) -> Optional[bytes]:
        key = (str(veh_id), int(round_num))
        with self._lock:
            item = self._buf.pop(key, None)
        return item[0] if item else None

    def wait_for(self, veh_id: str, round_num: int, timeout: float = 10.0) -> Optional[bytes]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            pkt = self.pop(veh_id, round_num)
            if pkt is not None:
                return pkt
            time.sleep(0.01)
        return None

    def _extract_veh_round(self, packet: bytes) -> Optional[Tuple[str, int]]:
        """
        Parse ONLY the header (no signature verify) to read veh + round.
        Defensive bounds checks.
        """
        try:
            if len(packet) < 4:
                return None
            (hdr_len,) = struct.unpack_from("!I", packet, 0)
            off = 4
            if hdr_len < 2 or hdr_len > 10_000_000:
                return None
            if len(packet) < off + hdr_len:
                return None
            header_bytes = packet[off : off + hdr_len]
            header_json = header_bytes.decode("utf-8", errors="strict")
            hdr = json.loads(header_json)

            veh = str(hdr.get("veh", self.veh_id))
            rnd = int(hdr.get("round", -1))
            if rnd < 0:
                return None
            return veh, rnd
        except Exception:
            return None

    def _run(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.listen_host, self.listen_port))
        srv.listen(64)
        print(f"[DownlinkListener] Listening on {self.listen_host}:{self.listen_port} veh={self.veh_id}", flush=True)

        # Wake periodically so stop can work
        srv.settimeout(1.0)

        while not self._stop:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                continue

            try:
                conn.settimeout(5.0)

                chunks: List[bytes] = []
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)

                payload = b"".join(chunks)
                if not payload:
                    continue

                key = self._extract_veh_round(payload)
                if key is None:
                    # fallback: store as latest for veh with round=-1
                    key = (self.veh_id, -1)
                else:
                    if str(key[0]) != str(self.veh_id):
                        print(
                            f"[DownlinkListener] WARNING: veh mismatch on port {self.listen_port}: "
                            f"listener_veh={self.veh_id} header_veh={key[0]} (drop)",
                            flush=True,
                        )
                        continue

                with self._lock:
                    self._buf[(key[0], int(key[1]))] = (payload, time.time())

                print(
                    f"[DownlinkListener] Stored veh={key[0]} round={key[1]} bytes={len(payload)} from {addr}",
                    flush=True,
                )

            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        try:
            srv.close()
        except Exception:
            pass


# BRIDGE NS-3 ORACLE FOR SENDING PACKET
class OracleTransport:
    def __init__(self, cfg: Dict[str, Any]):
        self.host = cfg.get("host", cfg.get("ip", "127.0.0.1"))
        self.port = int(cfg.get("port", 12345))
        print(f"OracleTransport using Oracle at {self.host}:{self.port}", flush=True)

        dl = cfg.get("downlink", {}) or {}
        self._dl_enabled = bool(dl.get("enabled", False))
        self._dl_listen_host = str(dl.get("listen_host", "127.0.0.1"))
        self._dl_base_port = int(dl.get("base_port", 25000))

        self._dl_listener: Optional[DownlinkListener] = None
        self._dl_bound_port: Optional[int] = None
        self._dl_veh_id: Optional[str] = None


    def _ensure_downlink_listener(self, veh_id: str) -> None:
        if not self._dl_enabled:
            return

        veh_id = str(veh_id)
        veh_index = _veh_index_from_id(veh_id)
        port = int(self._dl_base_port) + int(veh_index)

        # If already started, enforce stable binding
        if self._dl_listener is not None:
            if str(self._dl_veh_id) != veh_id or int(self._dl_bound_port or -1) != port:
                raise RuntimeError(
                    f"DownlinkListener already bound for veh={self._dl_veh_id} "
                    f"port={self._dl_bound_port}, cannot rebind to veh={veh_id} port={port}"
                )
            return

        self._dl_listener = DownlinkListener(
            listen_host=self._dl_listen_host,
            listen_port=port,
            veh_id=veh_id,
        )
        self._dl_bound_port = port
        self._dl_veh_id = veh_id
        self._dl_listener.start()



    def recv_wave_downlink(self, veh_id: str, round_num: int, timeout: float = 10.0) -> Optional[bytes]:
        """
        Wait for a downlink packet forwarded from ns-3.
        """
        if not self._dl_enabled:
            return None
        self._ensure_downlink_listener(veh_id)
        if not self._dl_listener:
            return None

        pkt = self._dl_listener.wait_for(str(veh_id), int(round_num), timeout=timeout)
        if pkt is not None:
            return pkt

        # Fallback: if parsing failed upstream, it might be stored as (veh, -1)
        return self._dl_listener.wait_for(str(veh_id), -1, timeout=0.01)
    
    # SEND_UPDATE client to ns-3 oracle and logs metrics
    def send_update(self, veh_id: str, round_num: int, packet: bytes) -> str:
        header_line = f"SEND_UPDATE,{veh_id},{round_num},{len(packet)}\n"
        header = header_line.encode("ascii")  # Need to convert to ASCII

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(120.0)

        t_start = time.time()
        ack = "NO_ACK"

        try:
            s.connect((self.host, self.port))
            # Send header and PQC bytes
            s.sendall(header)
            total = 0
            size = len(packet)
            # Uses memoryview instead of creating a copy
            view = memoryview(packet)

            while total < size:
                sent = s.send(view[total:])
                if sent <= 0:
                    raise RuntimeError("socket.send returned 0 bytes")
                total += sent

            #Tell ns-3 oracle done sending payload
            try:
                s.shutdown(socket.SHUT_WR)
            except Exception:
                pass

            # read ACK
            ack_raw = b""
            while not ack_raw.endswith(b"\n"):
                try:
                    chunk = s.recv(32)
                except socket.timeout:
                    # Timed out waiting for ACK
                    break
                if not chunk:
                    # Remote closed
                    break
                ack_raw += chunk
            t_mid = time.time()
            rtt_ms = (t_mid - t_start) * 1000.0
            if ack_raw:
                ack = ack_raw.decode(errors="replace").strip()
                print(
                    f"OracleTransport ACK from Oracle: {ack} "
                    f"(veh={veh_id}, round={round_num}, rtt_ms={rtt_ms:.2f})",
                    flush=True,
                )
            else:
                print(
                    f"OracleTransport: No ACK from Oracle "
                    f"(veh={veh_id}, round={round_num}, rtt_ms={rtt_ms:.2f})",
                    flush=True,
                )
                ack = "NO_ACK"
        except Exception as e:
            t_mid = time.time()
            rtt_ms = (t_mid - t_start) * 1000.0
            ack = f"ERROR: {e}"
            print(
                f"ERROR during SEND_UPDATE: {e} "
                f"(veh={veh_id}, round={round_num}, rtt_ms={rtt_ms:.2f})",
                flush=True,
            )
        finally:
            t_end = time.time()
            try:
                s.close()
            except Exception:
                pass

            # Log RTT metrics (even on error / NO_ACK)
            _log_oracle_rtt(
                veh_id=veh_id,
                round_num=round_num,
                size=len(packet),
                ack=ack,
                t_start=t_start,
                t_end=t_end,
            )

        return ack  # OK / BAD_FWD / DROP / NO_ACK / ERROR
    
    def send_global(self, veh_id: str, round_num: int, packet: bytes) -> str:
        header_line = f"SEND_GLOBAL,{veh_id},{round_num},{len(packet)}\n"
        header = header_line.encode("ascii")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(120.0)

        t_start = time.time()
        ack = "NO_ACK"

        try:
            s.connect((self.host, self.port))
            s.sendall(header)
            s.sendall(packet)
            try:
                s.shutdown(socket.SHUT_WR)
            except Exception:
                pass

            ack_raw = b""
            while not ack_raw.endswith(b"\n"):
                chunk = s.recv(32)
                if not chunk:
                    break
                ack_raw += chunk

            ack = ack_raw.decode(errors="replace").strip() if ack_raw else "NO_ACK"
        except Exception as e:
            ack = f"ERROR: {e}"
        finally:
            try:
                s.close()
            except Exception:
                pass

            _log_oracle_rtt(
                veh_id=veh_id,
                round_num=round_num,
                size=len(packet),
                ack=f"SEND_GLOBAL:{ack}",
                t_start=t_start,
                t_end=time.time(),
            )

        return ack

    # Wait for global model from ns-3 Oracle, delivered as PQC packet
    def recv_global(
        self,
        veh_id: str,
        round_num: int,
        timeout: float = 10.0,
    ) -> Optional[bytes]:
        """
        Request a global model for (veh_id, round_num) from the ns-3 Oracle.

        Protocol (to be implemented on ns-3 side):

            Client  -> Oracle: "WAIT_GLOBAL,<veh>,<round>\\n"
            Oracle <- Client:  "OK,<size>\\n" + <size> bytes of PQC packet
                               OR "NO_DATA\\n" / "DROP\\n" / "ERROR,...\\n"

        Returns:
            - PQC packet bytes if OK,<size> was received and all bytes read.
            - None if NO_DATA/DROP/ERROR or any failure occurred.
        """
        cmd_line = f"WAIT_GLOBAL,{veh_id},{round_num}\n"
        cmd = cmd_line.encode("ascii")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)

        t_start = time.time()
        ack = "NO_DATA"
        packet: Optional[bytes] = None
        size_bytes = 0

        try:
            s.connect((self.host, self.port))
            s.sendall(cmd)

            # Read status line
            status_raw = b""
            while not status_raw.endswith(b"\n"):
                chunk = s.recv(64)
                if not chunk:
                    break
                status_raw += chunk

            if not status_raw:
                ack = "NO_ACK"
                print(
                    f"OracleTransport.recv_global: No status from Oracle "
                    f"(veh={veh_id}, round={round_num})",
                    flush=True,
                )
                return None

            status = status_raw.decode(errors="replace").strip()
            # Expect formats like "OK,<size>", "NO_DATA", "DROP", "ERROR,..."
            if status.startswith("OK,"):
                try:
                    _, size_str = status.split(",", 1)
                    size_bytes = int(size_str)
                except Exception as e:
                    ack = f"ERROR_PARSE_SIZE: {e}"
                    print(
                        f"OracleTransport.recv_global: bad OK header '{status}' "
                        f"(veh={veh_id}, round={round_num})",
                        flush=True,
                    )
                    return None

                # Read exactly size_bytes bytes as PQC packet
                buf = bytearray()
                remaining = size_bytes
                while remaining > 0:
                    chunk = s.recv(remaining)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    remaining -= len(chunk)

                if len(buf) != size_bytes:
                    ack = "ERROR_SHORT_READ"
                    print(
                        f"OracleTransport.recv_global: short read: "
                        f"expected={size_bytes}, got={len(buf)} "
                        f"(veh={veh_id}, round={round_num})",
                        flush=True,
                    )
                    return None

                packet = bytes(buf)
                ack = "OK"
                t_mid = time.time()
                rtt_ms = (t_mid - t_start) * 1000.0
                print(
                    f"OracleTransport.recv_global: OK, size={size_bytes} bytes "
                    f"(veh={veh_id}, round={round_num}, rtt_ms={rtt_ms:.2f})",
                    flush=True,
                )
                return packet

            else:
                # NO_DATA / DROP / ERROR / anything else
                ack = status
                t_mid = time.time()
                rtt_ms = (t_mid - t_start) * 1000.0
                print(
                    f"OracleTransport.recv_global: status='{status}' "
                    f"(veh={veh_id}, round={round_num}, rtt_ms={rtt_ms:.2f})",
                    flush=True,
                )
                return None

        except Exception as e:
            ack = f"ERROR: {e}"
            t_mid = time.time()
            rtt_ms = (t_mid - t_start) * 1000.0
            print(
                f"OracleTransport.recv_global ERROR: {e} "
                f"(veh={veh_id}, round={round_num}, rtt_ms={rtt_ms:.2f})",
                flush=True,
            )
            return None
        finally:
            t_end = time.time()
            try:
                s.close()
            except Exception:
                pass

            # Log RTT even if NO_DATA / ERROR / etc.
            _log_oracle_rtt(
                veh_id=veh_id,
                round_num=round_num,
                size=size_bytes,
                ack=ack,
                t_start=t_start,
                t_end=t_end,
            )

    # GET_STATUS (Doesn't affect FL)
    def get_status(self, veh_id: Optional[str] = None) -> str:
        cmd = "GET_STATUS\n" if veh_id is None else f"GET_STATUS,{veh_id}\n"
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        try:
            s.connect((self.host, self.port))
            s.sendall(cmd.encode("ascii"))
            resp_raw = b""
            while not resp_raw.endswith(b"\n"):
                chunk = s.recv(64)
                if not chunk:
                    break
                resp_raw += chunk
            resp = resp_raw.decode(errors="replace").strip()
            print(f"OracleTransport GET_STATUS reply: {resp}", flush=True)
            return resp or "NO_REPLY"
        except Exception as e:
            print(f"OracleTransport error in GET_STATUS: {e}", flush=True)
            return f"ERROR: {e}"
        finally:
            try:
                s.close()
            except Exception:
                pass