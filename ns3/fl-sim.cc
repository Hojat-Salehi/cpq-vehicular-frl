#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/random-waypoint-mobility-model.h"
#include "ns3/position-allocator.h"
#include "ns3/random-variable-stream.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/node-list.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/packet.h"
#include "ns3/inet-socket-address.h"
#include "ns3/socket.h"
#include "ns3/log.h"
#include "ns3/object-factory.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/neighbor-cache-helper.h"

#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
#include <cerrno>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <chrono>
#include <unordered_map>
#include <deque>
#include <zlib.h> 
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <set>

#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLSimulation");

// STATE CONFIGURATION

// Global simulation state
static NodeContainer g_vehicleNodes;
static Ptr<Node> g_rsuNode;

// Keep OracleState for distance logging/covariates (not used to generate latency/drops anymore)
struct OracleState
{
  bool drop = false;
  double latencyMs = 0.0;
  double distance = 0.0;
};

static std::map<std::string, OracleState> g_vehOracleState;
static std::mutex g_oracleMutex;

// POSIX control-plane server running flag
static std::atomic<bool> g_running(true);

// Forward PQC packets to Python pqc_listener
static const char*   kPqcHost = "127.0.0.1";
static const uint16_t kPqcPort = 23456;

// Uplink WAVE/UDP app port on RSU (veh -> RSU over ns-3)
static const uint16_t kWaveUdpPort = 40000;

// Downlink: vehicle UDP listen port base (RSU -> veh over ns-3)
// IMPORTANT: must NOT collide with kWaveUdpPort
static const uint16_t kVehUdpBasePort = 41000;

// python client TCP listen port = base + i
static const uint16_t kPyVehTcpBasePort = 25000;

// SEGMENTATION CONFIG
static const uint32_t kDefaultChunkPayload = 1200;

// Separate UDP ports for ACK channel (so we don't confuse data/chunks)
static const uint16_t kWaveAckPort = 40001; // RSU listens for ACKs here, vehicles too

// SR parameters (can later be hooked to command line)
static const uint16_t kSrInitWindow = 32;    // initial in-flight chunks
static const uint16_t kSrMaxWindow  = 128;   // cap
static const uint32_t kSrRtoMs      = 120;   // retransmission timeout
static const uint32_t kSrMaxRetries = 8;     // per chunk cap
static const uint32_t kSrPacingUs   = 200;   // micro pacing between chunk sends

// Channel / scenario knobs (set from the command line in main(); defaults = paper setup)
static std::string g_propModel       = "logdist_nakagami"; // logdist_nakagami | logdist | friis | none
static double      g_pathLossExp     = 2.7;
static double      g_refLossDb       = 46.6777;
static double      g_preambleMinRssi = -82.0;  // ns-3 ThresholdPreambleDetectionModel default
static double      g_rxSensitivity   = -101.0; // ns-3 WifiPhy default
static double      g_areaHalf        = 250.0;  // RWP area = [-a,a]^2 around the RSU
static bool        g_populateArp     = true;   // pre-fill ARP caches: 802.11p V2X has no ARP, and ns-3 ARP
                                                // marks an out-of-range RSU dead for 100 s, which blocks later uploads
static bool        g_logMobility     = true;   // per-second [MOB] position lines

// Lockstep mode (--lockstep=true): no real-time scheduler. A Python driver (ns3_link.py) sets
// vehicle positions/velocities from SUMO and advances ns-3 to SUMO's simulation time over one
// persistent TCP connection; transfers are started asynchronously and their completions are
// returned as events (with the received bytes) when the simulation is advanced.
static bool        g_lockstep        = false;
static double      g_rsuX            = 0.0;
static double      g_rsuY            = 0.0;
static const double kParkedPos       = 1.0e6;   // far away from everything: out of range

// RSU beacons (lockstep): the RSU broadcasts a small beacon every g_beaconInterval s; a vehicle
// that receives it knows it is in radio range (discovery through the simulated channel).
static double      g_beaconInterval  = 0.1;     // s; 0 disables beacons
static const uint16_t kBeaconPort     = 40002;
static const uint32_t kBeaconBytes    = 100;
static Ptr<Socket> g_rsuBeaconSock;
static std::vector<Ptr<Socket>> g_vehBeaconSock;
static std::vector<std::deque<double>> g_beaconRx;   // per vehicle: recent beacon reception times (s)

struct LockstepEvent
{
  bool downlink;
  uint64_t msgId;
  uint32_t vehIndex;
  bool ok;
  double t;
  std::vector<uint8_t> bytes;   // received message (empty on failure)
};
static std::vector<LockstepEvent> g_lsEvents;
static std::set<std::pair<bool, uint64_t>> g_lsReported;   // (downlink, msgId) already reported

static void LockstepEmit(bool downlink, uint64_t msgId, uint32_t vehIndex, bool ok,
                         const std::vector<uint8_t>* bytes)
{
  if (!g_lsReported.insert(std::make_pair(downlink, msgId)).second)
    return;   // first outcome wins (e.g. RSU complete before the sender saw all ACKs)
  LockstepEvent ev{downlink, msgId, vehIndex, ok, Simulator::Now().GetSeconds(), {}};
  if (ok && bytes)
    ev.bytes = *bytes;
  g_lsEvents.push_back(std::move(ev));
}

// Message types
static const uint8_t  kMsgTypeData  = 1;
static const uint8_t  kMsgTypeAck   = 2;
static const uint16_t kFlagDownlink = 0x0001;

// RSU IP (assigned after IPv4 setup)
static Ipv4Address g_rsuIp;

// Per-vehicle UDP sockets (uplink sockets: veh->RSU)
static std::vector<Ptr<Socket>> g_vehUdpSockets;
static Ptr<Socket> g_rsuUdpSocket; // RSU uplink receiver

static Ptr<Socket> g_rsuAckRxSock;
static Ptr<Socket> g_rsuAckTxSock;
static std::vector<Ptr<Socket>> g_vehAckRxSock;
static std::vector<Ptr<Socket>> g_vehAckTxSock;

// RSU UDP socket used to send downlink into WAVE channel
static Ptr<Socket> g_rsuDownlinkSock;
static std::vector<Ipv4Address> g_vehicleIps; // index i -> IP

static std::vector<Ptr<Socket>> g_vehDownlinkRxSock;

// Message id and completion tracking (uplink)
static std::atomic<uint64_t> g_msgIdCounter{1};
static std::mutex g_pendingMutex;
static std::map<uint64_t, std::shared_ptr<std::promise<bool>>> g_pendingPromises;

static std::mutex g_pendingDlMutex;
static std::map<uint64_t, std::shared_ptr<std::promise<bool>>> g_pendingDlPromises;

// Logging file for ns-3 measured results
static std::ofstream g_ns3Log;
static std::string g_runId = "";

// Selective Repeat sender state
struct SrChunkTx {
  bool acked = false;
  uint32_t retries = 0;
  uint64_t lastTxSimNs = 0;
};

struct SrSendState {
  uint64_t msgId = 0;
  uint32_t round = 0;
  uint16_t vehIndex = 0;
  bool downlink = false;

  uint32_t totalLen = 0;
  uint16_t totalChunks = 0;
  uint32_t chunkPayload = kDefaultChunkPayload;

  uint16_t window = kSrInitWindow;
  uint16_t base = 0;          // lowest unacked
  uint16_t nextToSend = 0;    // next chunk index to send

  uint64_t txSimNs0 = 0;

  std::shared_ptr<std::vector<uint8_t>> bytes;

  std::vector<SrChunkTx> st;
  bool complete = false;
  bool failed = false;
  std::string failReason = "";

  std::shared_ptr<std::promise<bool>> prom; // reuse your completion path
};

static std::mutex g_srMutex;
// key: (vehIndex,msgId,dir) -> state
static std::map<std::tuple<uint16_t,uint64_t,bool>, SrSendState> g_srSends;

// Receiver reassembly buffers
struct ReasmState {
  uint32_t round = 0;
  uint16_t vehIndex = 0;
  bool downlink = false;
  uint32_t totalLen = 0;
  uint16_t totalChunks = 0;
  uint64_t firstRxSimNs = 0;
  uint64_t txSimNs = 0;

  std::vector<bool> got;
  std::vector<std::vector<uint8_t>> parts;
};

static std::mutex g_rxMutex;
static std::map<std::tuple<uint16_t,uint64_t,bool>, ReasmState> g_reasm; // (vehIndex,msgId,true)

// ---- Forward declarations (needed because callbacks appear before helper defs) ----
static uint32_t Crc32Bytes(const uint8_t* data, size_t n);

static void BuildAck(std::vector<uint8_t>& out,
                     bool downlink,
                     uint64_t msgId,
                     uint32_t round,
                     uint16_t vehIndex,
                     uint16_t chunkIdx,
                     uint16_t totalChunks,
                     uint16_t windowHint,
                     uint64_t rxSimNs);

static bool ParseChunkHeader(const uint8_t* buf, size_t n,
                             uint8_t& msgType,
                             uint16_t& flags,
                             uint64_t& msgId,
                             uint32_t& round,
                             uint16_t& vehIndex,
                             uint16_t& chunkIdx,
                             uint16_t& totalChunks,
                             uint16_t& windowHint,
                             uint64_t& txSimNs,
                             uint32_t& totalLen,
                             uint32_t& chunkLen,
                             uint32_t& crc32v,
                             const uint8_t*& payloadPtr);

static void SrScheduleTick(uint16_t vehIndex, uint64_t msgId, bool downlink);

static void StartSrSendUplink(uint16_t vehIndex,
                              uint64_t msgId,
                              uint32_t round,
                              std::shared_ptr<std::vector<uint8_t>> bytes);

static void StartSrSendDownlink(uint16_t vehIndex,
                                uint64_t msgId,
                                uint32_t round,
                                std::shared_ptr<std::vector<uint8_t>> bytes,
                                std::shared_ptr<std::promise<bool>> prom);


// Helpers
static std::string GetRunId()
{
  const char* env = std::getenv("FL_RUN_ID");
  if (env && std::strlen(env) > 0)
    return std::string(env);

  // Fallback: run_YYYYMMDD_HHMMSS (localtime)
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "run_%Y%m%d_%H%M%S", &tm);
  return std::string(buf);
}

static void EnsureNs3LogOpen()
{
  if (g_ns3Log.is_open())
    return;

  g_runId = GetRunId();
  std::ostringstream dir;
  dir << "pqc_logs/" << g_runId;
  ::mkdir("pqc_logs", 0755); // ok if exists
  ::mkdir(dir.str().c_str(), 0755);

  std::ostringstream path;
  path << dir.str() << "/ns3_wave_metrics.csv";
  g_ns3Log.open(path.str(), std::ios::out | std::ios::app);

  // Write header if file is empty
  if (g_ns3Log.tellp() == 0)
    {
      g_ns3Log
        << "sim_ts_s,"
        << "msg_id,"
        << "veh,"
        << "round,"
        << "pqc_bytes,"
        << "tx_sim_ns,"
        << "rx_sim_ns,"
        << "oneway_sim_ms,"
        << "forward_ok,"
        << "dist_m\n";
      g_ns3Log.flush();
    }

  NS_LOG_INFO("ns-3 metrics logging to " << path.str());
}

static void
TrySetPhyThreshold(ns3::YansWifiPhyHelper& phy, const std::string& name, double v)
{
    try
    {
        phy.Set(name, ns3::DoubleValue(v));
        NS_LOG_UNCOND("[fl-sim] Set YansWifiPhy " << name << "=" << v);
    }
    catch (...)
    {
        NS_LOG_UNCOND("[fl-sim] WARNING: YansWifiPhy has no attribute '" << name
                       << "' (ignoring).");
    }
}

static inline uint32_t ParseVehIndex(const std::string& vehId)
{
  // Expect "veh<number>"
  if (vehId.rfind("veh", 0) != 0)
    return 0;
  std::string num = vehId.substr(3);
  return static_cast<uint32_t>(std::stoul(num));
}

// Serialize fixed app header into bytes (network order)
// Header:
//   u64 msg_id
//   u32 round
//   u16 veh_index
//   u64 tx_sim_ns
//   u32 pqc_len
static void AppendU64(std::vector<uint8_t>& out, uint64_t v)
{
  for (int i = 7; i >= 0; --i) out.push_back(static_cast<uint8_t>((v >> (8*i)) & 0xff));
}
static void AppendU32(std::vector<uint8_t>& out, uint32_t v)
{
  for (int i = 3; i >= 0; --i) out.push_back(static_cast<uint8_t>((v >> (8*i)) & 0xff));
}
static void AppendU16(std::vector<uint8_t>& out, uint16_t v)
{
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>(v & 0xff));
}

static uint64_t ReadU64(const uint8_t* p)
{
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
  return v;
}
static uint32_t ReadU32(const uint8_t* p)
{
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v = (v << 8) | p[i];
  return v;
}
static uint16_t ReadU16(const uint8_t* p)
{
  return static_cast<uint16_t>((p[0] << 8) | p[1]);
}

static void RsuAckRecvCallback(Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> pkt;
  while ((pkt = socket->RecvFrom(from))) {
    std::vector<uint8_t> buf(pkt->GetSize());
    pkt->CopyData(buf.data(), buf.size());

    uint8_t msgType; uint16_t flags; uint64_t msgId; uint32_t round;
    uint16_t vehIndex, chunkIdx, totalChunks, windowHint;
    uint64_t txSimNs; uint32_t totalLen, chunkLen, crc32v;
    const uint8_t* payloadPtr = nullptr;

    if (!ParseChunkHeader(buf.data(), buf.size(), msgType, flags, msgId, round, vehIndex,
                          chunkIdx, totalChunks, windowHint, txSimNs, totalLen, chunkLen, crc32v, payloadPtr))
      continue;
    
    if (msgType != kMsgTypeAck) continue;

      // Infer which SR send state to update based on the downlink flag in the ACK header
    bool downlink = (flags & kFlagDownlink) != 0;


    std::lock_guard<std::mutex> lk(g_srMutex);
    auto key = std::make_tuple(vehIndex, msgId, downlink);
    auto it = g_srSends.find(key);
    if (it == g_srSends.end()) continue;
    SrSendState& ss = it->second;
    if (chunkIdx < ss.totalChunks) {
      ss.st[chunkIdx].acked = true;
      // advance base
      while (ss.base < ss.totalChunks && ss.st[ss.base].acked) ss.base++;
      // AIMD: on good progress, increase window a bit
      if (ss.window < kSrMaxWindow) ss.window += 1;
    }
  }
}

static void VehAckRecvCallback(uint32_t vehIdx, Ptr<Socket> sock)
{
  Address from;
  Ptr<Packet> pkt;
  while ((pkt = sock->RecvFrom(from))) {
    std::vector<uint8_t> buf(pkt->GetSize());
    pkt->CopyData(buf.data(), buf.size());

    uint8_t msgType; uint16_t flags; uint64_t msgId; uint32_t round;
    uint16_t vehIndex, chunkIdx, totalChunks, windowHint;
    uint64_t txSimNs; uint32_t totalLen, chunkLen, crc32v;
    const uint8_t* payloadPtr = nullptr;

    if (!ParseChunkHeader(buf.data(), buf.size(), msgType, flags, msgId, round, vehIndex,
                          chunkIdx, totalChunks, windowHint, txSimNs, totalLen, chunkLen, crc32v, payloadPtr))
      continue;
    if (msgType != kMsgTypeAck) continue;

    bool downlink = false;

    std::lock_guard<std::mutex> lk(g_srMutex);
    auto key = std::make_tuple((uint16_t)vehIdx, msgId, downlink);
    auto it = g_srSends.find(key);
    if (it == g_srSends.end()) continue;
    SrSendState& ss = it->second;
    if (chunkIdx < ss.totalChunks) {
      ss.st[chunkIdx].acked = true;
      while (ss.base < ss.totalChunks && ss.st[ss.base].acked) ss.base++;
      if (ss.window < kSrMaxWindow) ss.window += 1;
    }
  }
}

static void ResolveUplinkPromise(uint64_t msgId, bool ok)
{
  std::shared_ptr<std::promise<bool>> prom;
  {
    std::lock_guard<std::mutex> lock(g_pendingMutex);
    auto it = g_pendingPromises.find(msgId);
    if (it != g_pendingPromises.end())
      {
        prom = it->second;
        g_pendingPromises.erase(it);
      }
  }
  if (prom)
    {
      try { prom->set_value(ok); }
      catch (...) {}
    }
}

// Forward to Python PQC oracle
static bool ForwardToPqcAggregator(const std::string& vehId,
                                  int round,
                                  const std::vector<uint8_t>& packet)
{
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0)
    {
      perror("ForwardToPqcAggregator socket");
      return false;
    }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kPqcPort);
  if (::inet_pton(AF_INET, kPqcHost, &addr.sin_addr) <= 0)
    {
      perror("ForwardToPqcAggregator inet_pton");
      close(sock);
      return false;
    }
  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
      perror("ForwardToPqcAggregator connect");
      close(sock);
      return false;
    }

  int one = 1;
  (void)setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));


  size_t total = 0;
  while (total < packet.size())
    {
      ssize_t n = ::write(sock, packet.data() + total, packet.size() - total);
      if (n <= 0)
        {
          perror("ForwardToPqcAggregator write");
          close(sock);
          return false;
        }
      total += static_cast<size_t>(n);
    }

  (void)::shutdown(sock, SHUT_WR);

  struct timeval tv {};
  tv.tv_sec = 5;
  tv.tv_usec = 0;
  (void)setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  std::string ack;
  ack.reserve(32);

  char ackBuf[64];
  while (ack.find('\n') == std::string::npos)
    {
      ssize_t n = ::read(sock, ackBuf, sizeof(ackBuf));
      if (n > 0)
        {
          ack.append(ackBuf, ackBuf + n);
          if (ack.size() > 256) break; // defensive cap
          continue;
        }
      if (n == 0)
        break; // peer closed
      // n < 0
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break; // timeout
      perror("ForwardToPqcAggregator read");
      break;
    }

  close(sock);

  // Normalize
  ack.erase(std::remove(ack.begin(), ack.end(), '\r'), ack.end());
  ack.erase(std::remove(ack.begin(), ack.end(), '\n'), ack.end());

  return (ack == "OK");
}

// Downlink forwarder: ns-3 -> python client process (TCP localhost:25000+vehIndex)
static bool ForwardToPythonClient(uint32_t vehIndex,
                                  const std::vector<uint8_t>& packet)
{
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) { perror("ForwardToPythonClient socket"); return false; }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kPyVehTcpBasePort + vehIndex);
  if (::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) <= 0) {
    perror("ForwardToPythonClient inet_pton");
    close(sock);
    return false;
  }

  // short timeouts so we don't block ns-3 realtime forever
  struct timeval tv{};
  tv.tv_sec = 2;
  tv.tv_usec = 0;
  (void)setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  (void)setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    perror("ForwardToPythonClient connect");
    close(sock);
    return false;
  }

  size_t total = 0;
  while (total < packet.size()) {
    ssize_t n = ::write(sock, packet.data() + total, packet.size() - total);
    if (n <= 0) { perror("ForwardToPythonClient write"); close(sock); return false; }
    total += (size_t)n;
  }

  (void)::shutdown(sock, SHUT_WR);
  close(sock);
  return true;
}

// Distance logging only
static void UpdateOracle()
{
  if (g_vehicleNodes.GetN() == 0 || g_rsuNode == nullptr)
    {
      Simulator::Schedule(MilliSeconds(100), &UpdateOracle);
      return;
    }

  Ptr<MobilityModel> rsuMob = g_rsuNode->GetObject<MobilityModel>();
  if (!rsuMob)
    {
      Simulator::Schedule(MilliSeconds(100), &UpdateOracle);
      return;
    }

  const double maxDistClamp = 500.0; // covariate only
  std::map<std::string, OracleState> localVehState;

  for (uint32_t i = 0; i < g_vehicleNodes.GetN(); ++i)
    {
      Ptr<Node> node = g_vehicleNodes.Get(i);
      Ptr<MobilityModel> vm = node->GetObject<MobilityModel>();
      if (!vm) continue;

      double raw_d = vm->GetDistanceFrom(rsuMob);
      double d = std::min(raw_d, maxDistClamp);

      std::string vehId = "veh" + std::to_string(i);
      OracleState st;
      st.distance = d;
      st.drop = false;
      st.latencyMs = 0.0;
      localVehState[vehId] = st;
    }

  {
    std::lock_guard<std::mutex> lock(g_oracleMutex);
    g_vehOracleState = std::move(localVehState);
  }

  Simulator::Schedule(MilliSeconds(50), &UpdateOracle);
}

// ns-3 WAVE send/receive (uplink)
static void SendViaNs3(uint32_t vehIndex,
                       uint64_t msgId,
                       int round,
                       std::shared_ptr<std::vector<uint8_t>> pqcBytes)
{
  if (vehIndex >= g_vehUdpSockets.size() || !g_vehUdpSockets[vehIndex])
    {
      NS_LOG_WARN("SendViaNs3: missing socket for vehIndex=" << vehIndex);
      return;
    }

  // Start SR chunking for uplink (veh -> RSU)
  StartSrSendUplink((uint16_t)vehIndex, msgId, (uint32_t)round, pqcBytes);
}


static void RsuRecvCallback(Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> pkt;

  while ((pkt = socket->RecvFrom(from)))
    {
      EnsureNs3LogOpen();

      std::vector<uint8_t> buf(pkt->GetSize());
      pkt->CopyData(buf.data(), buf.size());

      uint8_t msgType; uint16_t flags; uint64_t msgId; uint32_t round;
      uint16_t vehIndex, chunkIdx, totalChunks, windowHint;
      uint64_t txSimNs; uint32_t totalLen, chunkLen, crc32v;
      const uint8_t* payloadPtr = nullptr;

      if (!ParseChunkHeader(buf.data(), buf.size(), msgType, flags, msgId, round, vehIndex,
                            chunkIdx, totalChunks, windowHint, txSimNs, totalLen, chunkLen, crc32v, payloadPtr))
        {
          NS_LOG_WARN("RSU recv: bad chunk header");
          continue;
        }

      if (msgType != kMsgTypeData)
        {
          continue;
        }

      // CRC check
      uint32_t c = Crc32Bytes(payloadPtr, (size_t)chunkLen);
      if (c != crc32v)
        {
          NS_LOG_WARN("RSU recv: CRC mismatch msgId=" << msgId << " chunk=" << chunkIdx);
          // No ACK -> forces retransmit by timeout (or could send NACK later)
          continue;
        }

      uint64_t rxSimNs = Simulator::Now().GetNanoSeconds();

      // Store chunk
      bool completed = false;
      std::vector<uint8_t> assembled;

      {
        std::lock_guard<std::mutex> lk(g_rxMutex);
        auto key = std::make_tuple((uint16_t)vehIndex, msgId, false);
        auto& rs = g_reasm[key];
        if (rs.parts.empty())
          {
            rs.round = round;
            rs.vehIndex = vehIndex;
            rs.downlink = false;
            rs.totalLen = totalLen;
            rs.totalChunks = totalChunks;
            rs.firstRxSimNs = rxSimNs;
            rs.txSimNs = txSimNs;
            rs.got.assign(totalChunks, false);
            rs.parts.resize(totalChunks);
          }

        if (chunkIdx < rs.totalChunks && !rs.got[chunkIdx])
          {
            rs.parts[chunkIdx].assign(payloadPtr, payloadPtr + chunkLen);
            rs.got[chunkIdx] = true;
          }

        completed = std::all_of(rs.got.begin(), rs.got.end(), [](bool b){return b;});

        if (completed)
          {
            assembled.reserve(rs.totalLen);
            for (uint16_t i = 0; i < rs.totalChunks; ++i)
              {
                assembled.insert(assembled.end(), rs.parts[i].begin(), rs.parts[i].end());
              }
            g_reasm.erase(key);
          }
      }

      // Send ACK for this chunk back to the vehicle
      if (g_rsuAckTxSock)
        {
          std::vector<uint8_t> ackb;
          BuildAck(ackb, false, msgId, round, vehIndex, chunkIdx, totalChunks, kSrInitWindow, rxSimNs);
          Ptr<Packet> ap = Create<Packet>(ackb.data(), ackb.size());

          // reply to vehicle IP on ACK port
          if (vehIndex < g_vehicleIps.size())
            {
              InetSocketAddress dst = InetSocketAddress(g_vehicleIps[vehIndex], kWaveAckPort);
              g_rsuAckTxSock->SendTo(ap, 0, dst);
            }
        }

      if (!completed)
        continue;

      // Completed message: forward to pqc_oracle (lockstep: hand the bytes to the Python
      // driver, which verifies the PQC signature itself)
      std::string vehId = "veh" + std::to_string(vehIndex);
      bool forwardOk = true;
      if (g_lockstep)
        LockstepEmit(false, msgId, vehIndex, true, &assembled);
      else
        forwardOk = ForwardToPqcAggregator(vehId, (int)round, assembled);

      double oneway_ms = (double)(rxSimNs - txSimNs) / 1e6;

      double distM = -1.0;
      if (vehIndex < g_vehicleNodes.GetN())
        distM = g_vehicleNodes.Get(vehIndex)->GetObject<MobilityModel>()->GetDistanceFrom(
                  g_rsuNode->GetObject<MobilityModel>());

      g_ns3Log
        << Simulator::Now().GetSeconds() << ","
        << msgId << ","
        << vehId << ","
        << round << ","
        << totalLen << ","
        << txSimNs << ","
        << rxSimNs << ","
        << oneway_ms << ","
        << (forwardOk ? 1 : 0) << ","
        << distM << "\n";
      g_ns3Log.flush();

      ResolveUplinkPromise(msgId, forwardOk);
    }
}

// Header layout:
//   u8  msg_type   (1 = data chunk, 2 = ack)
//   u8  reserved
//   u16 flags      (bit0: downlink)
//   u64 msg_id
//   u32 round
//   u16 veh_index
//   u16 chunk_idx
//   u16 total_chunks
//   u16 window_hint   (sender's current window; optional telemetry)
//   u64 tx_sim_ns
//   u32 total_len
//   u32 chunk_len
//   u32 crc32         (crc of chunk payload bytes)
static void AppendU8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }

static bool ParseU8(const uint8_t*& p, const uint8_t* end, uint8_t& out)
{
  if (p + 1 > end) return false;
  out = *p; p += 1;
  return true;
}

static uint32_t Crc32Bytes(const uint8_t* data, size_t n)
{
  return (uint32_t)crc32(0L, data, (uInt)n);
}

static size_t ChunkHdrSize()
{
  return (size_t)(1+1+2+8+4+2+2+2+2+8+4+4+4); // 44 bytes
}

static void BuildDataChunk(std::vector<uint8_t>& out,
                           bool downlink,
                           uint64_t msgId,
                           uint32_t round,
                           uint16_t vehIndex,
                           uint16_t chunkIdx,
                           uint16_t totalChunks,
                           uint16_t windowHint,
                           uint64_t txSimNs,
                           uint32_t totalLen,
                           const uint8_t* payload,
                           uint32_t chunkLen)
{
  out.clear();
  out.reserve(ChunkHdrSize() + chunkLen);

  AppendU8(out, kMsgTypeData);
  AppendU8(out, 0);
  AppendU16(out, downlink ? kFlagDownlink : 0);
  AppendU64(out, msgId);
  AppendU32(out, round);
  AppendU16(out, vehIndex);
  AppendU16(out, chunkIdx);
  AppendU16(out, totalChunks);
  AppendU16(out, windowHint);
  AppendU64(out, txSimNs);
  AppendU32(out, totalLen);
  AppendU32(out, chunkLen);

  uint32_t c = Crc32Bytes(payload, chunkLen);
  AppendU32(out, c);

  out.insert(out.end(), payload, payload + chunkLen);
}

// ACK packet: no payload
static void BuildAck(std::vector<uint8_t>& out,
                     bool downlink,
                     uint64_t msgId,
                     uint32_t round,
                     uint16_t vehIndex,
                     uint16_t chunkIdx,
                     uint16_t totalChunks,
                     uint16_t windowHint,
                     uint64_t rxSimNs)
{
  out.clear();
  out.reserve(ChunkHdrSize());
  AppendU8(out, kMsgTypeAck);
  AppendU8(out, 0);
  AppendU16(out, downlink ? kFlagDownlink : 0);
  AppendU64(out, msgId);
  AppendU32(out, round);
  AppendU16(out, vehIndex);
  AppendU16(out, chunkIdx);
  AppendU16(out, totalChunks);
  AppendU16(out, windowHint);
  AppendU64(out, rxSimNs);
  AppendU32(out, 0);
  AppendU32(out, 0);
  AppendU32(out, 0);
}

// Parse common header
static bool ParseChunkHeader(const uint8_t* buf, size_t n,
                             uint8_t& msgType,
                             uint16_t& flags,
                             uint64_t& msgId,
                             uint32_t& round,
                             uint16_t& vehIndex,
                             uint16_t& chunkIdx,
                             uint16_t& totalChunks,
                             uint16_t& windowHint,
                             uint64_t& txSimNs,
                             uint32_t& totalLen,
                             uint32_t& chunkLen,
                             uint32_t& crc32v,
                             const uint8_t*& payloadPtr)
{
  if (n < ChunkHdrSize()) return false;
  const uint8_t* p = buf;
  const uint8_t* end = buf + n;

  if (!ParseU8(p, end, msgType)) return false;
  uint8_t reserved = 0;
  if (!ParseU8(p, end, reserved)) return false;

  flags = ReadU16(p); p += 2;
  msgId = ReadU64(p); p += 8;
  round = ReadU32(p); p += 4;
  vehIndex = ReadU16(p); p += 2;
  chunkIdx = ReadU16(p); p += 2;
  totalChunks = ReadU16(p); p += 2;
  windowHint = ReadU16(p); p += 2;
  txSimNs = ReadU64(p); p += 8;
  totalLen = ReadU32(p); p += 4;
  chunkLen = ReadU32(p); p += 4;
  crc32v = ReadU32(p); p += 4;

  payloadPtr = p;
  if (msgType == kMsgTypeData) {
    if ((size_t)(end - p) < chunkLen) return false;
  }
  return true;
}

// ns-3 downlink receive on vehicle nodes
// static void
// VehDownlinkRecvCallback(uint32_t vehIdx, ns3::Ptr<ns3::Socket> sock)
// {
//   Address from;
//   Ptr<Packet> p = sock->RecvFrom(from);
//   if (!p) return;

//   std::vector<uint8_t> buf(p->GetSize());
//   p->CopyData(buf.data(), buf.size());

//   NS_LOG_INFO("veh" << vehIdx << " RX downlink over WAVE bytes=" << buf.size());

//   bool ok = ForwardToPythonClient(vehIdx, buf);
//   if (!ok)
//   {
//     NS_LOG_WARN("veh" << vehIdx << " forward-to-python failed");
//   }
// }

// Chunk-aware vehicle downlink RX
static void
VehDownlinkRecvCallback(uint32_t vehIdx, ns3::Ptr<ns3::Socket> sock)
{
  Address from;
  Ptr<Packet> pkt;

  while ((pkt = sock->RecvFrom(from)))
    {
      std::vector<uint8_t> buf(pkt->GetSize());
      pkt->CopyData(buf.data(), buf.size());

      uint8_t msgType; uint16_t flags; uint64_t msgId; uint32_t round;
      uint16_t vehIndex, chunkIdx, totalChunks, windowHint;
      uint64_t txSimNs; uint32_t totalLen, chunkLen, crc32v;
      const uint8_t* payloadPtr = nullptr;

      if (!ParseChunkHeader(buf.data(), buf.size(), msgType, flags, msgId, round, vehIndex,
                            chunkIdx, totalChunks, windowHint, txSimNs, totalLen, chunkLen, crc32v, payloadPtr))
        {
          NS_LOG_WARN("veh" << vehIdx << " RX downlink: bad chunk header");
          continue;
        }

      if (msgType != kMsgTypeData)
        continue;

      // Must be downlink
      if ((flags & kFlagDownlink) == 0)
        {
          NS_LOG_WARN("veh" << vehIdx << " RX downlink: missing downlink flag msgId=" << msgId);
          continue;
        }

      // Veh index sanity
      if (vehIndex != (uint16_t)vehIdx)
        {
          NS_LOG_WARN("veh" << vehIdx << " RX downlink: veh mismatch hdrVeh=" << vehIndex
                            << " msgId=" << msgId);
          continue;
        }

      if (!g_lockstep)   // per-chunk trace; lockstep runs log transfers in comm.csv
      NS_LOG_UNCOND("veh" << vehIdx
                    << " DL RX msgId=" << msgId
                    << " round=" << round
                    << " chunk=" << chunkIdx << "/" << totalChunks
                    << " len=" << chunkLen
                    << " from=" << InetSocketAddress::ConvertFrom(from).GetIpv4()
                    << ":" << InetSocketAddress::ConvertFrom(from).GetPort());

      // CRC check
      uint32_t c = Crc32Bytes(payloadPtr, (size_t)chunkLen);
      if (c != crc32v)
        {
          NS_LOG_WARN("veh" << vehIdx << " RX downlink: CRC mismatch msgId=" << msgId
                            << " chunk=" << chunkIdx);
          // No ACK -> forces retransmit by timeout
          continue;
        }

      uint64_t rxSimNs = Simulator::Now().GetNanoSeconds();

      // Reassembly (reuse your g_reasm map; key=(veh,msgId,downlink=true))
      bool completed = false;
      std::vector<uint8_t> assembled;

      {
        std::lock_guard<std::mutex> lk(g_rxMutex);
        auto key = std::make_tuple((uint16_t)vehIdx, msgId, true);
        auto& rs = g_reasm[key];

        if (rs.parts.empty())
          {
            rs.round = round;
            rs.vehIndex = (uint16_t)vehIdx;
            rs.downlink = true;
            rs.totalLen = totalLen;
            rs.totalChunks = totalChunks;
            rs.firstRxSimNs = rxSimNs;
            rs.txSimNs = txSimNs;
            rs.got.assign(totalChunks, false);
            rs.parts.resize(totalChunks);
          }

        // Defensive: header mismatch against existing state
        if (rs.totalChunks != totalChunks || rs.totalLen != totalLen || rs.round != round)
          {
            NS_LOG_WARN("veh" << vehIdx << " RX downlink: reasm conflict msgId=" << msgId);
            g_reasm.erase(key);
            continue;
          }

        if (chunkIdx < rs.totalChunks && !rs.got[chunkIdx])
          {
            rs.parts[chunkIdx].assign(payloadPtr, payloadPtr + chunkLen);
            rs.got[chunkIdx] = true;
          }

        completed = std::all_of(rs.got.begin(), rs.got.end(), [](bool b){ return b; });

        if (completed)
          {
            assembled.reserve(rs.totalLen);
            for (uint16_t i = 0; i < rs.totalChunks; ++i)
              assembled.insert(assembled.end(), rs.parts[i].begin(), rs.parts[i].end());
            g_reasm.erase(key);
          }
      }

      // ACK this chunk back to RSU (vehicle -> RSU on ACK port)
      if (vehIdx < g_vehAckTxSock.size() && g_vehAckTxSock[vehIdx])
        {
          std::vector<uint8_t> ackb;
          BuildAck(ackb, true, msgId, round, (uint16_t)vehIdx, chunkIdx, totalChunks, kSrInitWindow, rxSimNs);
          Ptr<Packet> ap = Create<Packet>(ackb.data(), ackb.size());

          InetSocketAddress dst = InetSocketAddress(g_rsuIp, kWaveAckPort);
          int sent = g_vehAckTxSock[vehIdx]->SendTo(ap, 0, dst);

          if (!g_lockstep)
          NS_LOG_UNCOND("veh" << vehIdx
                        << " DL ACK tx msgId=" << msgId
                        << " chunk=" << chunkIdx
                        << " -> " << g_rsuIp << ":" << kWaveAckPort
                        << " sent=" << sent);
        }

      if (!completed)
        continue;

      // Complete message -> forward to python client TCP localhost:25000+vehIdx
      NS_LOG_INFO("veh" << vehIdx << " RX downlink COMPLETE msgId=" << msgId
                        << " round=" << round << " bytes=" << assembled.size());

      if (g_lockstep)
        {
          LockstepEmit(true, msgId, vehIdx, true, &assembled);
          continue;
        }
      bool ok = ForwardToPythonClient(vehIdx, assembled);
      if (!ok)
        NS_LOG_WARN("veh" << vehIdx << " forward-to-python failed (downlink) msgId=" << msgId);
    }
}

static void SendGlobalViaNs3(uint32_t vidx,
                             uint64_t msgId,
                             uint32_t round,
                             std::shared_ptr<std::vector<uint8_t>> packet)
{
  if (!g_rsuDownlinkSock)
    {
      NS_LOG_ERROR("RSU downlink UDP socket not initialized");
      return;
    }
  if (vidx >= g_vehicleIps.size())
    {
      NS_LOG_ERROR("veh index out of range vidx=" << vidx);
      return;
    }

  // Look up the promise created by SEND_GLOBAL (same msgId)
  std::shared_ptr<std::promise<bool>> prom;
  {
    std::lock_guard<std::mutex> lock(g_pendingDlMutex);
    auto it = g_pendingDlPromises.find(msgId);
    if (it != g_pendingDlPromises.end())
      prom = it->second;
  }

  if (!prom)
    {
      NS_LOG_ERROR("downlink: missing pending promise for msgId=" << msgId);
      return;
    }

  // Start SR downlink using the caller-provided msgId + promise
  StartSrSendDownlink((uint16_t)vidx, msgId, round, packet, prom);
}

// POSIX TCP Control-plane Oracle
static void OracleServerThread(uint16_t port)
{
  int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("Oracle socket"); return; }

  int opt = 1;
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
    {
      perror("Oracle setsockopt(SO_REUSEADDR)");
      close(server_fd);
      return;
    }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  if (bind(server_fd, (sockaddr *)&addr, sizeof(addr)) < 0)
    {
      perror("Oracle bind()");
      close(server_fd);
      return;
    }
  if (listen(server_fd, 5) < 0)
    {
      perror("Oracle listen()");
      close(server_fd);
      return;
    }

  NS_LOG_INFO("OracleServer: listening on 0.0.0.0:" << port);

  struct timeval tv{};
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  (void)setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  while (g_running.load())
    {
      sockaddr_in caddr{};
      socklen_t clen = sizeof(caddr);
      int cfd = accept(server_fd, (sockaddr*)&caddr, &clen);
      if (cfd < 0)
        {
          if (!g_running.load()) break;
          if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
          perror("Oracle accept");
          continue;
        }

      // Read header line
      std::string line;
      std::vector<uint8_t> initialPayload;
      bool headerOk = true;

      {
        char buf[256];
        bool headerDone = false;
        while (!headerDone)
          {
            ssize_t n = ::read(cfd, buf, sizeof(buf));
            if (n <= 0)
              {
                close(cfd);
                headerOk = false;
                break;
              }
            for (ssize_t i = 0; i < n; ++i)
              {
                char ch = buf[i];
                if (!headerDone)
                  {
                    line.push_back(ch);
                    if (ch == '\n')
                      {
                        headerDone = true;
                        if (i + 1 < n)
                          {
                            initialPayload.insert(
                              initialPayload.end(),
                              reinterpret_cast<uint8_t *>(buf + i + 1),
                              reinterpret_cast<uint8_t *>(buf + n));
                          }
                        break;
                      }
                  }
              }
            if (!headerOk || headerDone) break;
          }

        if (!headerOk) continue;

        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
      }

      // GET_STATUS: return distance covariate
      if (line.rfind("GET_STATUS", 0) == 0)
        {
          std::string vehId;
          size_t commaPos = line.find(',');
          if (commaPos != std::string::npos && commaPos + 1 < line.size())
            vehId = line.substr(commaPos + 1);

          double d = -1.0;
          {
            std::lock_guard<std::mutex> lock(g_oracleMutex);
            auto it = g_vehOracleState.find(vehId);
            if (it != g_vehOracleState.end()) d = it->second.distance;
          }

          std::ostringstream os;
          os << "OK," << d << "\n"; // d = vehicle-RSU distance in m (-1 if unknown)
          std::string resp = os.str();
          (void)::write(cfd, resp.c_str(), resp.size());
          close(cfd);
          continue;
        }

      // SEND_UPDATE: ingest bytes, then send via ns-3 UDP over 802.11p to RSU
      if (line.rfind("SEND_UPDATE", 0) == 0)
        {
          char vehIdBuf[64];
          int round = 0;
          long size = 0;

          int parsed = std::sscanf(line.c_str(), "SEND_UPDATE,%63[^,],%d,%ld",
                                   vehIdBuf, &round, &size);
          if (parsed != 3 || size <= 0)
            {
              const char* perr = "PARSE_ERR\n";
              (void)::write(cfd, perr, std::strlen(perr));
              close(cfd);
              continue;
            }

          std::string vehId(vehIdBuf);
          uint32_t vehIndex = ParseVehIndex(vehId);
          if (vehIndex >= g_vehicleNodes.GetN())
            {
              const char* perr = "PARSE_ERR\n";
              (void)::write(cfd, perr, std::strlen(perr));
              close(cfd);
              continue;
            }

          // Read exactly size bytes as PQC wire packet
          long remaining = size;
          long total_read = 0;
          std::vector<uint8_t> packet;
          packet.reserve(size);

          if (!initialPayload.empty())
            {
              long used = std::min<long>(remaining, (long)initialPayload.size());
              packet.insert(packet.end(), initialPayload.begin(), initialPayload.begin() + used);
              remaining -= used;
              total_read += used;
            }

          std::vector<uint8_t> buf(4096);
          while (remaining > 0)
            {
              ssize_t n = ::read(cfd, buf.data(), std::min<long>(remaining, (long)buf.size()));
              if (n <= 0) break;
              packet.insert(packet.end(), buf.begin(), buf.begin() + n);
              remaining -= n;
              total_read += n;
            }

          if (total_read != size)
            {
              const char* bad = "PARSE_ERR\n";
              (void)::write(cfd, bad, std::strlen(bad));
              close(cfd);
              continue;
            }

          // Create promise/future for completion (RSU reception + forward-to-python pqc_oracle)
          uint64_t msgId = g_msgIdCounter.fetch_add(1);
          auto prom = std::make_shared<std::promise<bool>>();
          std::future<bool> fut = prom->get_future();
          {
            std::lock_guard<std::mutex> lock(g_pendingMutex);
            g_pendingPromises[msgId] = prom;
          }

          auto sharedBytes = std::make_shared<std::vector<uint8_t>>(std::move(packet));

          Simulator::ScheduleWithContext(
            g_vehicleNodes.Get(vehIndex)->GetId(),
            Seconds(0.0),
            &SendViaNs3,
            vehIndex,
            msgId,
            round,
            sharedBytes
          );

          bool ok = false;
          if (fut.wait_for(std::chrono::seconds(10)) == std::future_status::ready)
            {
              ok = fut.get();
            }
          else
            {
              std::lock_guard<std::mutex> lock(g_pendingMutex);
              auto it = g_pendingPromises.find(msgId);
              if (it != g_pendingPromises.end())
                g_pendingPromises.erase(it);
              ok = false;
            }

          std::string ack = ok ? "OK\n" : "BAD_FWD\n";
          (void)::write(cfd, ack.c_str(), ack.size());
          close(cfd);
          continue;
        }

      // SEND_GLOBAL: server->vehicle downlink over ns-3 WAVE, payload is raw PQC packet bytes
      if (line.rfind("SEND_GLOBAL", 0) == 0)
      {
        char vehIdBuf[64];
        int round = 0;
        long size = 0;

        int parsed = std::sscanf(line.c_str(), "SEND_GLOBAL,%63[^,],%d,%ld",
                                vehIdBuf, &round, &size);
        if (parsed != 3 || size <= 0)
          {
            const char* perr = "PARSE_ERR\n";
            (void)::write(cfd, perr, std::strlen(perr));
            close(cfd);
            continue;
          }

        std::string vehId(vehIdBuf);

        uint32_t vidx = 0;
        if (vehId.rfind("veh", 0) == 0)
          {
            vidx = (uint32_t)std::stoul(vehId.substr(3));
          }
        else
          {
            const char* perr = "BAD_VEH\n";
            (void)::write(cfd, perr, std::strlen(perr));
            close(cfd);
            continue;
          }

        if (vidx >= g_vehicleNodes.GetN())
          {
            const char* perr = "BAD_VEH\n";
            (void)::write(cfd, perr, std::strlen(perr));
            close(cfd);
            continue;
          }

        long remaining = size;
        long total_read = 0;
        auto packet = std::make_shared<std::vector<uint8_t>>();
        packet->reserve((size_t)size);

        // consume any bytes already read after header newline
        if (!initialPayload.empty())
          {
            long used = std::min<long>(remaining, (long)initialPayload.size());
            packet->insert(packet->end(),
                          initialPayload.begin(),
                          initialPayload.begin() + used);
            remaining -= used;
            total_read += used;
          }

        std::vector<uint8_t> buf(4096);
        while (remaining > 0)
          {
            ssize_t n = ::read(cfd, buf.data(), std::min<long>(remaining, (long)buf.size()));
            if (n <= 0) break;
            packet->insert(packet->end(), buf.begin(), buf.begin() + n);
            remaining -= n;
            total_read += n;
          }

        if (total_read != size)
          {
            const char* bad = "SHORT_READ\n";
            (void)::write(cfd, bad, std::strlen(bad));
            close(cfd);
            continue;
          }

        // Create msgId + promise/future for *downlink* SR completion
        uint64_t msgId = g_msgIdCounter.fetch_add(1);
        auto prom = std::make_shared<std::promise<bool>>();
        std::future<bool> fut = prom->get_future();

        {
          std::lock_guard<std::mutex> lock(g_pendingDlMutex);
          g_pendingDlPromises[msgId] = prom;
        }

        // Kick SR downlink in ns-3 context
        Simulator::ScheduleNow(&SendGlobalViaNs3, vidx, msgId, (uint32_t)round, packet);

        // Wait for SR completion (ACKs) with timeout
        bool ok = false;
        if (fut.wait_for(std::chrono::seconds(10)) == std::future_status::ready)
          {
            ok = fut.get();
          }
        else
          {
            // timeout: cleanup pending (SR may still be running; this just prevents leaks)
            std::lock_guard<std::mutex> lock(g_pendingDlMutex);
            auto it = g_pendingDlPromises.find(msgId);
            if (it != g_pendingDlPromises.end())
              g_pendingDlPromises.erase(it);
            ok = false;
          }

        std::string resp = ok ? "OK\n" : "BAD_DL\n";
        (void)::write(cfd, resp.c_str(), resp.size());
        close(cfd);
        continue;
      }
        
      const char* uack = "UNKNOWN\n";
      (void)::write(cfd, uack, std::strlen(uack));
      close(cfd);
    }

  close(server_fd);
  NS_LOG_INFO("OracleServer: shut down");
}

static void SrScheduleTick(uint16_t vehIndex, uint64_t msgId, bool downlink);

// Start an SR send by creating SrSendState and kicking SrScheduleTick
static void StartSrSendUplink(uint16_t vehIndex,
                              uint64_t msgId,
                              uint32_t round,
                              std::shared_ptr<std::vector<uint8_t>> bytes)
{
  SrSendState ss;
  ss.msgId = msgId;
  ss.round = round;
  ss.vehIndex = vehIndex;
  ss.downlink = false;

  ss.totalLen = (uint32_t)bytes->size();
  ss.chunkPayload = kDefaultChunkPayload;
  ss.totalChunks = (uint16_t)((ss.totalLen + ss.chunkPayload - 1) / ss.chunkPayload);

  ss.window = kSrInitWindow;
  ss.base = 0;
  ss.nextToSend = 0;
  ss.txSimNs0 = Simulator::Now().GetNanoSeconds();

  ss.bytes = bytes;
  ss.st.assign(ss.totalChunks, SrChunkTx{});

  ss.prom = std::make_shared<std::promise<bool>>();

  {
    std::lock_guard<std::mutex> lk(g_srMutex);
    g_srSends[std::make_tuple(vehIndex, msgId, false)] = std::move(ss);
  }

  // Kick SR tick loop
  SrScheduleTick(vehIndex, msgId, false);
}

static void StartSrSendDownlink(uint16_t vehIndex,
                                uint64_t msgId,
                                uint32_t round,
                                std::shared_ptr<std::vector<uint8_t>> bytes,
                                std::shared_ptr<std::promise<bool>> prom)
{
  SrSendState ss;
  ss.msgId = msgId;
  ss.round = round;
  ss.vehIndex = vehIndex;
  ss.downlink = true;
  ss.prom = prom; 

  ss.totalLen = (uint32_t)bytes->size();
  ss.chunkPayload = kDefaultChunkPayload;
  ss.totalChunks = (uint16_t)((ss.totalLen + ss.chunkPayload - 1) / ss.chunkPayload);

  ss.window = kSrInitWindow;
  ss.base = 0;
  ss.nextToSend = 0;
  ss.txSimNs0 = Simulator::Now().GetNanoSeconds();

  ss.bytes = bytes;
  ss.st.assign(ss.totalChunks, SrChunkTx{});

  if (!ss.prom)
    {
      // Fallback (if someone calls StartSrSendDownlink without SEND_GLOBAL)
      ss.prom = std::make_shared<std::promise<bool>>();
    }

  {
    std::lock_guard<std::mutex> lk(g_srMutex);
    g_srSends[std::make_tuple(vehIndex, msgId, true)] = std::move(ss);
  }

  SrScheduleTick(vehIndex, msgId, true);
}

static void SrSendOne(uint16_t vehIndex, uint64_t msgId, bool downlink, uint16_t chunkIdx)
{
  std::lock_guard<std::mutex> lk(g_srMutex);
  auto key = std::make_tuple(vehIndex, msgId, downlink);
  auto it = g_srSends.find(key);
  if (it == g_srSends.end()) return;
  SrSendState& ss = it->second;
  if (ss.complete || ss.failed) return;
  if (chunkIdx >= ss.totalChunks) return;
  if (ss.st[chunkIdx].acked) return;

  uint32_t off = (uint32_t)chunkIdx * ss.chunkPayload;
  uint32_t len = std::min(ss.chunkPayload, ss.totalLen - off);

  std::vector<uint8_t> pktBytes;
  BuildDataChunk(pktBytes,
                 downlink,
                 ss.msgId,
                 ss.round,
                 ss.vehIndex,
                 chunkIdx,
                 ss.totalChunks,
                 ss.window,
                 Simulator::Now().GetNanoSeconds(),
                 ss.totalLen,
                 ss.bytes->data() + off,
                 len);

  Ptr<Packet> p = Create<Packet>(pktBytes.data(), pktBytes.size());

  if (!downlink) {
    // uplink: veh -> RSU using existing connected socket
    if (vehIndex < g_vehUdpSockets.size() && g_vehUdpSockets[vehIndex]) {
      g_vehUdpSockets[vehIndex]->Send(p);
    }
  } else {
    // downlink: RSU -> veh
    if (g_rsuDownlinkSock && vehIndex < g_vehicleIps.size()) {
      InetSocketAddress dst = InetSocketAddress(g_vehicleIps[vehIndex], kVehUdpBasePort + vehIndex);
      if (!g_lockstep)
      NS_LOG_UNCOND("RSU DL send msgId=" << msgId << " veh=" << vehIndex
                   << " chunk=" << chunkIdx << "/" << ss.totalChunks
                   << " to " << g_vehicleIps[vehIndex] << ":" << (kVehUdpBasePort + vehIndex));
      g_rsuDownlinkSock->SendTo(p, 0, dst);
    }
  }

  ss.st[chunkIdx].retries += 1;
  ss.st[chunkIdx].lastTxSimNs = Simulator::Now().GetNanoSeconds();

  // Congestion-aware pacing: schedule next tick soon
}

static void SrScheduleTick(uint16_t vehIndex, uint64_t msgId, bool downlink)
{
  // run every few ms to send new chunks and retransmit timeouts
  Simulator::Schedule(MilliSeconds(5), [vehIndex, msgId, downlink]() {
    std::shared_ptr<std::promise<bool>> promToResolve;
    bool resolveOk = false;
    std::string failReason;
    std::vector<uint16_t> toRetransmit;

    {
      std::lock_guard<std::mutex> lk(g_srMutex);
      auto key = std::make_tuple(vehIndex, msgId, downlink);
      auto it = g_srSends.find(key);
      if (it == g_srSends.end()) return;
      SrSendState& ss = it->second;
      if (ss.complete || ss.failed) {
        // finalize outside lock
      } else {
        // 1) retransmit timed-out chunks in window
        uint64_t nowNs = Simulator::Now().GetNanoSeconds();
        uint64_t rtoNs = (uint64_t)kSrRtoMs * 1000000ULL;

        for (uint16_t i = ss.base; i < ss.totalChunks && i < (uint16_t)(ss.base + ss.window); ++i) {
          if (ss.st[i].acked) continue;
          if (ss.st[i].lastTxSimNs == 0) continue; // not yet sent
          if (nowNs - ss.st[i].lastTxSimNs >= rtoNs) {
            if (ss.st[i].retries >= kSrMaxRetries) {
              ss.failed = true;
              ss.failReason = "CHUNK_TIMEOUT";
              break;
            }
            // AIMD: on timeout, cut window
            ss.window = std::max<uint16_t>(8, (uint16_t)(ss.window / 2));

            toRetransmit.push_back(i);
            // retransmit
            // (we’ll retransmit outside lock by marking list; simplest minimal: call SrSendOne after lock)
          }
        }

        // 2) send new chunks while window allows, spaced kSrPacingUs apart
        uint32_t paceSlot = 1;
        while (!ss.failed &&
               ss.nextToSend < ss.totalChunks &&
               ss.nextToSend < (uint16_t)(ss.base + ss.window))
        {
          // send next chunk
          uint16_t idx = ss.nextToSend;
          ss.nextToSend++;
          // send outside lock by scheduling a paced lambda
          Simulator::Schedule(MicroSeconds(kSrPacingUs * paceSlot++), [vehIndex, msgId, downlink, idx]() {
            SrSendOne(vehIndex, msgId, downlink, idx);
          });
        }

        // 3) check completion
        if (!ss.failed) {
          bool allAcked = true;
          for (uint16_t i = 0; i < ss.totalChunks; ++i) {
            if (!ss.st[i].acked) { allAcked = false; break; }
          }
          if (allAcked) {
            ss.complete = true;
          }
        }
      }

      // if complete/failed, pop state and resolve promise
      auto it2 = g_srSends.find(key);
      if (it2 != g_srSends.end()) {
        SrSendState& ss = it2->second;
        if (ss.complete || ss.failed) {
          promToResolve = ss.prom;
          resolveOk = ss.complete && !ss.failed;
          failReason = ss.failReason;
          g_srSends.erase(it2);
        }
      }
    }

    if (promToResolve) {
      promToResolve->set_value(resolveOk);
      if (g_lockstep && !resolveOk)
        LockstepEmit(downlink, msgId, vehIndex, false, nullptr);   // retry cap reached

      if (downlink) {
        std::lock_guard<std::mutex> lock(g_pendingDlMutex);
        g_pendingDlPromises.erase(msgId);
      } else if (!resolveOk) {
        // Uplink gave up (retry cap): report the drop now instead of at the 10 s timeout
        ResolveUplinkPromise(msgId, false);
      }
    }

    // Retransmit timed-out chunks (do NOT hold g_srMutex while calling SrSendOne)
    uint32_t rtxSlot = 1;
    for (uint16_t idx : toRetransmit)
      {
        Simulator::Schedule(MicroSeconds(kSrPacingUs * rtxSlot++), [vehIndex, msgId, downlink, idx]() {
          SrSendOne(vehIndex, msgId, downlink, idx);
        });
      }

    // If still active, continue ticking
    bool stillActive = false;
    {
      std::lock_guard<std::mutex> lk(g_srMutex);
      auto key = std::make_tuple(vehIndex, msgId, downlink);
      stillActive = (g_srSends.find(key) != g_srSends.end());
    }
    if (stillActive)
      {
        SrScheduleTick(vehIndex, msgId, downlink);
      }

  });
}

// This checks if the random waypoint mobility is actually working
static void LogPositions()
{
  Ptr<MobilityModel> rsuMob = g_rsuNode->GetObject<MobilityModel>();
  Vector rsuPos = rsuMob->GetPosition();

  for (uint32_t i = 0; i < g_vehicleNodes.GetN(); ++i)
  {
    Ptr<MobilityModel> m = g_vehicleNodes.Get(i)->GetObject<MobilityModel>();
    Vector p = m->GetPosition();
    double d = m->GetDistanceFrom(rsuMob);

    NS_LOG_UNCOND("[MOB] t=" << Simulator::Now().GetSeconds()
                  << " veh" << i
                  << " pos=(" << p.x << "," << p.y << ")"
                  << " d_to_rsu=" << d);
  }

  Simulator::Schedule(Seconds(1.0), &LogPositions);
}

// -------------------- LOCKSTEP CONTROL LOOP --------------------
// One persistent TCP connection from the Python driver (ns3_link.py). Commands (one line each):
//   POS i x y vx vy [i x y vx vy ...]   set position (m) and velocity (m/s) of vehicles
//   PARK i [i ...]                      move vehicles out of range (not in the network)
//   SEND_UPDATE veh round size\n<bytes> start an uplink now          -> "MSG <id>"
//   SEND_GLOBAL veh round size\n<bytes> start a downlink now         -> "MSG <id>"
//   ADVANCE t                           run the simulation until t (s) -> "EVENTS k" then per
//                                       event "UL|DL <id> <veh> <ok> <t> <nbytes>\n<bytes>"
//   STATUS i                            -> "<x> <y> <distance to RSU>"
//   NOW                                 -> "<t>"
//   HEARD w m                           vehicles that received at least m RSU beacons within
//                                       the last w s -> "HEARD k i1 ... ik"
//   RESET                               drop all transfers in flight, park all vehicles -> "OK"
//   QUIT                                -> "BYE"
class LineConn
{
public:
  explicit LineConn(int fd) : m_fd(fd) {}
  bool ReadLine(std::string& line)
  {
    line.clear();
    while (true)
      {
        auto nl = std::find(m_buf.begin(), m_buf.end(), '\n');
        if (nl != m_buf.end())
          {
            line.assign(m_buf.begin(), nl);
            m_buf.erase(m_buf.begin(), nl + 1);
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            return true;
          }
        if (!Fill()) return false;
      }
  }
  bool ReadExact(size_t n, std::vector<uint8_t>& out)
  {
    while (m_buf.size() < n)
      if (!Fill()) return false;
    out.assign(m_buf.begin(), m_buf.begin() + n);
    m_buf.erase(m_buf.begin(), m_buf.begin() + n);
    return true;
  }
  bool Write(const void* data, size_t n)
  {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (n > 0)
      {
        ssize_t w = ::write(m_fd, p, n);
        if (w <= 0) return false;
        p += w;
        n -= (size_t)w;
      }
    return true;
  }
  bool WriteLine(const std::string& s) { std::string l = s + "\n"; return Write(l.data(), l.size()); }

private:
  bool Fill()
  {
    char tmp[65536];
    ssize_t n = ::read(m_fd, tmp, sizeof(tmp));
    if (n <= 0) return false;
    m_buf.insert(m_buf.end(), tmp, tmp + n);
    return true;
  }
  int m_fd;
  std::deque<char> m_buf;
};

static void RsuSendBeacon()
{
  Ptr<Packet> p = Create<Packet>(kBeaconBytes);
  g_rsuBeaconSock->SendTo(p, 0, InetSocketAddress(Ipv4Address("255.255.255.255"), kBeaconPort));
  Simulator::Schedule(Seconds(g_beaconInterval), &RsuSendBeacon);
}

static void VehBeaconRecv(uint32_t vehIdx, Ptr<Socket> sock)
{
  while (sock->Recv())
    {
      auto& q = g_beaconRx[vehIdx];
      q.push_back(Simulator::Now().GetSeconds());
      while (q.size() > 64) q.pop_front();
    }
}

static void SetupBeacons()
{
  g_rsuBeaconSock = Socket::CreateSocket(g_rsuNode, UdpSocketFactory::GetTypeId());
  g_rsuBeaconSock->SetAllowBroadcast(true);
  g_rsuBeaconSock->Bind();
  g_beaconRx.assign(g_vehicleNodes.GetN(), std::deque<double>());
  g_vehBeaconSock.resize(g_vehicleNodes.GetN());
  for (uint32_t i = 0; i < g_vehicleNodes.GetN(); ++i)
    {
      Ptr<Socket> rx = Socket::CreateSocket(g_vehicleNodes.Get(i), UdpSocketFactory::GetTypeId());
      rx->Bind(InetSocketAddress(Ipv4Address::GetAny(), kBeaconPort));
      rx->SetRecvCallback(MakeBoundCallback(&VehBeaconRecv, i));
      g_vehBeaconSock[i] = rx;
    }
  Simulator::Schedule(Seconds(g_beaconInterval), &RsuSendBeacon);
}

static void SetVehicleMotion(uint32_t i, double x, double y, double vx, double vy)
{
  Ptr<ConstantVelocityMobilityModel> m =
      g_vehicleNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
  m->SetPosition(Vector(x, y, 0.0));
  m->SetVelocity(Vector(vx, vy, 0.0));
}

static void ParkVehicle(uint32_t i)
{
  // spread parked vehicles apart as well, so they never hear each other
  SetVehicleMotion(i, kParkedPos + 1000.0 * i, kParkedPos, 0.0, 0.0);
}

static void LockstepControlLoop(uint16_t port)
{
  int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0 || listen(server_fd, 1) < 0)
    {
      perror("lockstep bind/listen");
      return;
    }
  NS_LOG_UNCOND("[fl-sim] lockstep: listening on 127.0.0.1:" << port);
  int cfd = accept(server_fd, nullptr, nullptr);
  close(server_fd);
  if (cfd < 0) { perror("lockstep accept"); return; }
  int one = 1;
  setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  LineConn conn(cfd);

  std::string line;
  while (conn.ReadLine(line))
    {
      std::istringstream in(line);
      std::string cmd;
      in >> cmd;

      if (cmd == "POS")
        {
          uint32_t i; double x, y, vx, vy;
          while (in >> i >> x >> y >> vx >> vy)
            if (i < g_vehicleNodes.GetN()) SetVehicleMotion(i, x, y, vx, vy);
          conn.WriteLine("OK");
        }
      else if (cmd == "PARK")
        {
          uint32_t i;
          while (in >> i)
            if (i < g_vehicleNodes.GetN()) ParkVehicle(i);
          conn.WriteLine("OK");
        }
      else if (cmd == "SEND_UPDATE" || cmd == "SEND_GLOBAL")
        {
          std::string vehId; int round = 0; long size = 0;
          in >> vehId >> round >> size;
          std::vector<uint8_t> payload;
          if (size <= 0 || !conn.ReadExact((size_t)size, payload)) break;
          uint32_t vidx = ParseVehIndex(vehId);
          if (vidx >= g_vehicleNodes.GetN()) { conn.WriteLine("BAD_VEH"); continue; }
          uint64_t msgId = g_msgIdCounter.fetch_add(1);
          auto bytes = std::make_shared<std::vector<uint8_t>>(std::move(payload));
          if (cmd == "SEND_UPDATE")
            {
              Simulator::ScheduleWithContext(g_vehicleNodes.Get(vidx)->GetId(), Seconds(0.0),
                                             &SendViaNs3, vidx, msgId, round, bytes);
            }
          else
            {
              {
                std::lock_guard<std::mutex> lock(g_pendingDlMutex);
                g_pendingDlPromises[msgId] = std::make_shared<std::promise<bool>>();
              }
              Simulator::ScheduleWithContext(g_rsuNode->GetId(), Seconds(0.0),
                                             &SendGlobalViaNs3, vidx, msgId, (uint32_t)round, bytes);
            }
          conn.WriteLine("MSG " + std::to_string(msgId));
        }
      else if (cmd == "ADVANCE")
        {
          double t = 0.0;
          in >> t;
          Time target = Seconds(t);
          if (target > Simulator::Now())
            {
              Simulator::Stop(target - Simulator::Now());
              Simulator::Run();
            }
          std::vector<LockstepEvent> evs;
          evs.swap(g_lsEvents);
          bool ok = conn.WriteLine("EVENTS " + std::to_string(evs.size()));
          for (const auto& ev : evs)
            {
              std::ostringstream os;
              os << std::setprecision(9) << (ev.downlink ? "DL " : "UL ") << ev.msgId << " "
                 << ev.vehIndex << " " << (ev.ok ? 1 : 0) << " " << ev.t << " " << ev.bytes.size();
              ok = ok && conn.WriteLine(os.str());
              if (!ev.bytes.empty())
                ok = ok && conn.Write(ev.bytes.data(), ev.bytes.size());
            }
          if (!ok) break;
        }
      else if (cmd == "STATUS")
        {
          uint32_t i = 0;
          in >> i;
          if (i >= g_vehicleNodes.GetN()) { conn.WriteLine("BAD_VEH"); continue; }
          Ptr<MobilityModel> m = g_vehicleNodes.Get(i)->GetObject<MobilityModel>();
          Vector p = m->GetPosition();
          std::ostringstream os;
          os << std::setprecision(9) << p.x << " " << p.y << " "
             << m->GetDistanceFrom(g_rsuNode->GetObject<MobilityModel>());
          conn.WriteLine(os.str());
        }
      else if (cmd == "HEARD")
        {
          double w = 0.5;
          uint32_t minCount = 1;
          in >> w >> minCount;
          double now = Simulator::Now().GetSeconds();
          std::ostringstream os;
          std::vector<uint32_t> heard;
          for (uint32_t i = 0; i < g_beaconRx.size(); ++i)
            {
              uint32_t n = (uint32_t)std::count_if(g_beaconRx[i].begin(), g_beaconRx[i].end(),
                                                   [&](double t) { return now - t <= w + 1e-9; });
              if (n >= minCount) heard.push_back(i);
            }
          os << "HEARD " << heard.size();
          for (uint32_t i : heard) os << " " << i;
          conn.WriteLine(os.str());
        }
      else if (cmd == "NOW")
        {
          std::ostringstream os;
          os << std::setprecision(12) << Simulator::Now().GetSeconds();
          conn.WriteLine(os.str());
        }
      else if (cmd == "RESET")
        {
          {
            std::lock_guard<std::mutex> lk(g_srMutex);
            g_srSends.clear();      // pending SR ticks find no state and stop
          }
          {
            std::lock_guard<std::mutex> lk(g_rxMutex);
            g_reasm.clear();
          }
          {
            std::lock_guard<std::mutex> lk(g_pendingDlMutex);
            g_pendingDlPromises.clear();
          }
          g_lsEvents.clear();
          for (auto& q : g_beaconRx) q.clear();
          for (uint32_t i = 0; i < g_vehicleNodes.GetN(); ++i) ParkVehicle(i);
          conn.WriteLine("OK");
        }
      else if (cmd == "QUIT")
        {
          conn.WriteLine("BYE");
          break;
        }
      else
        {
          conn.WriteLine("UNKNOWN");
        }
    }
  close(cfd);
}

// -------------------- MAIN --------------------

int main(int argc, char *argv[])
{
  LogComponentEnable("FLSimulation", LOG_LEVEL_INFO);

  uint32_t numVehicles = 5;
  CommandLine cmd(__FILE__);
  cmd.AddValue("numVehicles", "Number of vehicle nodes", numVehicles);
  cmd.AddValue("propModel", "logdist_nakagami | logdist | friis | none", g_propModel);
  cmd.AddValue("pathLossExp", "Log-distance path-loss exponent", g_pathLossExp);
  cmd.AddValue("refLoss", "Log-distance reference loss at 1 m (dB)", g_refLossDb);
  cmd.AddValue("preambleMinRssi", "Preamble-detection minimum RSSI (dBm)", g_preambleMinRssi);
  cmd.AddValue("rxSensitivity", "PHY RX sensitivity (dBm)", g_rxSensitivity);
  cmd.AddValue("areaHalf", "Half side of the RWP area around the RSU (m)", g_areaHalf);
  cmd.AddValue("populateArp", "Pre-populate ARP caches", g_populateArp);
  cmd.AddValue("logMobility", "Print per-second [MOB] position lines", g_logMobility);
  uint16_t controlPort = 12345;
  cmd.AddValue("lockstep", "Driven by SUMO time and positions (ns3_link.py) instead of real time", g_lockstep);
  cmd.AddValue("rsuX", "RSU x position (m)", g_rsuX);
  cmd.AddValue("rsuY", "RSU y position (m)", g_rsuY);
  cmd.AddValue("controlPort", "TCP port of the control plane (oracle / lockstep driver)", controlPort);
  cmd.AddValue("beaconInterval", "Lockstep: RSU beacon period in s (0 = no beacons)", g_beaconInterval);
  cmd.Parse(argc, argv);

  NS_LOG_UNCOND("[fl-sim] propModel=" << g_propModel << " n=" << g_pathLossExp
                << " L0=" << g_refLossDb << " preambleMinRssi=" << g_preambleMinRssi
                << " rxSensitivity=" << g_rxSensitivity << " areaHalf=" << g_areaHalf
                << " populateArp=" << g_populateArp << " lockstep=" << g_lockstep
                << " rsu=(" << g_rsuX << "," << g_rsuY << ")");

  if (!g_lockstep)
    GlobalValue::Bind("SimulatorImplementationType",
                      StringValue("ns3::RealtimeSimulatorImpl"));

  // 1) Create nodes
  NodeContainer vehicleNodes;
  vehicleNodes.Create(numVehicles);

  NodeContainer rsuNodeContainer;
  rsuNodeContainer.Create(1);

  g_vehicleNodes = vehicleNodes;
  g_rsuNode = rsuNodeContainer.Get(0);

  // 2) Internet stack
  InternetStackHelper internet;
  internet.Install(vehicleNodes);
  internet.Install(rsuNodeContainer);

  // 3) Mobility
  MobilityHelper rsuMob;
  rsuMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  rsuMob.Install(rsuNodeContainer);

  Ptr<ConstantPositionMobilityModel> rsuMobility =
      rsuNodeContainer.Get(0)->GetObject<ConstantPositionMobilityModel>();
  rsuMobility->SetPosition(Vector(g_rsuX, g_rsuY, 0.0));

  MobilityHelper vehMob;
  if (g_lockstep)
    {
      // Vehicles: positions and velocities come from SUMO; parked out of range until set
      vehMob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
      vehMob.Install(vehicleNodes);
      for (uint32_t i = 0; i < numVehicles; ++i) ParkVehicle(i);
    }
  else
    {
  // Vehicles: RandomWaypoint

  // Initial positions (required by RandomWaypoint)
  Ptr<PositionAllocator> posAlloc = CreateObject<RandomRectanglePositionAllocator>();
  std::ostringstream areaRv;
  areaRv << "ns3::UniformRandomVariable[Min=" << -g_areaHalf << "|Max=" << g_areaHalf << "]";
  posAlloc->SetAttribute("X", StringValue(areaRv.str()));
  posAlloc->SetAttribute("Y", StringValue(areaRv.str()));

  vehMob.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                          "Speed", StringValue("ns3::UniformRandomVariable[Min=5.0|Max=15.0]"),
                          "Pause", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
                          "PositionAllocator", PointerValue(posAlloc));

  vehMob.SetPositionAllocator(posAlloc); // important: initial positions
  vehMob.Install(vehicleNodes);
    }

  if (g_logMobility)
    Simulator::Schedule(Seconds(1.0), &LogPositions);

  // 4) Wireless: 802.11p ad-hoc with realistic channel models
  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

  if (g_propModel == "logdist_nakagami" || g_propModel == "logdist")
    {
      wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                                     "Exponent", DoubleValue(g_pathLossExp),
                                     "ReferenceLoss", DoubleValue(g_refLossDb));
      if (g_propModel == "logdist_nakagami")
        wifiChannel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");
    }
  else if (g_propModel == "friis")
    {
      wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel",
                                     "Frequency", DoubleValue(5.9e9));
    }
  else if (g_propModel == "none")
    {
      // Yans needs a loss model; this one passes every frame at full TX power
      wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                                     "MaxRange", DoubleValue(1e6));
    }
  else
    {
      NS_FATAL_ERROR("Unknown propModel " << g_propModel);
    }

  YansWifiPhyHelper wifiPhy;
  wifiPhy.SetChannel(wifiChannel.Create());

  wifiPhy.Set("TxPowerStart", DoubleValue(20.0));
  wifiPhy.Set("TxPowerEnd", DoubleValue(20.0));
  wifiPhy.Set("RxNoiseFigure", DoubleValue(7.0));
  wifiPhy.Set("RxSensitivity", DoubleValue(g_rxSensitivity));
  wifiPhy.SetPreambleDetectionModel("ns3::ThresholdPreambleDetectionModel",
                                    "MinimumRssi", DoubleValue(g_preambleMinRssi));
  TrySetPhyThreshold(wifiPhy, "CcaEdThreshold", -85.0);

  WifiMacHelper wifiMac;
  wifiMac.SetType("ns3::AdhocWifiMac", "QosSupported", BooleanValue(false));

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211p);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue("OfdmRate6MbpsBW10MHz"),
                               "ControlMode", StringValue("OfdmRate6MbpsBW10MHz"));

  NetDeviceContainer vehicleDevices = wifi.Install(wifiPhy, wifiMac, vehicleNodes);
  NetDeviceContainer rsuDevices     = wifi.Install(wifiPhy, wifiMac, rsuNodeContainer);

  // 5) IPv4 addressing
  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer vehIfs = ipv4.Assign(vehicleDevices);
  Ipv4InterfaceContainer rsuIfs = ipv4.Assign(rsuDevices);

  g_rsuIp = rsuIfs.GetAddress(0);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  if (g_populateArp)
    {
      NeighborCacheHelper neighborCache;
      neighborCache.PopulateNeighborCache();
    }

  // Cache vehicle IPs for RSU->vehicle sends
  g_vehicleIps.clear();
  for (uint32_t i = 0; i < vehicleNodes.GetN(); ++i) {
    Ptr<Ipv4> ipv4v = vehicleNodes.Get(i)->GetObject<Ipv4>();
    Ipv4Address ip = ipv4v->GetAddress(1, 0).GetLocal();
    g_vehicleIps.push_back(ip);
    NS_LOG_INFO("veh" << i << " IP=" << ip);
  }

  // RSU UDP socket used to send downlink into WAVE channel
  g_rsuDownlinkSock = Socket::CreateSocket(g_rsuNode, UdpSocketFactory::GetTypeId());
  g_rsuDownlinkSock->Bind();

  // 6) Create ns-3 UDP sockets for uplink
  g_vehUdpSockets.resize(numVehicles);

  for (uint32_t i = 0; i < numVehicles; ++i)
    {
      Ptr<Node> veh = vehicleNodes.Get(i);
      Ptr<Socket> s = Socket::CreateSocket(veh, UdpSocketFactory::GetTypeId());
      InetSocketAddress dst = InetSocketAddress(g_rsuIp, kWaveUdpPort);
      s->Connect(dst);
      g_vehUdpSockets[i] = s;
    }

  g_rsuUdpSocket = Socket::CreateSocket(g_rsuNode, UdpSocketFactory::GetTypeId());
  InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), kWaveUdpPort);
  g_rsuUdpSocket->Bind(local);
  g_rsuUdpSocket->SetRecvCallback(MakeCallback(&RsuRecvCallback));

  NS_LOG_INFO("RSU bound UDP recv on " << g_rsuIp << ":" << kWaveUdpPort);

  g_rsuAckRxSock = Socket::CreateSocket(g_rsuNode, UdpSocketFactory::GetTypeId());
  InetSocketAddress rsuAckLocal = InetSocketAddress(g_rsuIp, kWaveAckPort);
  g_rsuAckRxSock->Bind(rsuAckLocal);
  g_rsuAckRxSock->SetRecvCallback(MakeCallback(&RsuAckRecvCallback));

  g_rsuAckTxSock = Socket::CreateSocket(g_rsuNode, UdpSocketFactory::GetTypeId());
  g_rsuAckTxSock->Bind();

  // Vehicle ACK sockets
  g_vehAckRxSock.resize(numVehicles);
  g_vehAckTxSock.resize(numVehicles);

  for (uint32_t i = 0; i < numVehicles; ++i) {
    Ptr<Node> v = vehicleNodes.Get(i);

    Ptr<Socket> arx = Socket::CreateSocket(v, UdpSocketFactory::GetTypeId());
    InetSocketAddress vAckLocal = InetSocketAddress(g_vehicleIps[i], kWaveAckPort);
    if (arx->Bind(vAckLocal) != 0) {
      NS_LOG_ERROR("veh" << i << " bind ACK port failed");
    } else {
      arx->SetRecvCallback(MakeBoundCallback(&VehAckRecvCallback, i));
    }
    g_vehAckRxSock[i] = arx;

    Ptr<Socket> atx = Socket::CreateSocket(v, UdpSocketFactory::GetTypeId());
    atx->Bind();
    g_vehAckTxSock[i] = atx;
  }

  g_vehDownlinkRxSock.resize(vehicleNodes.GetN());

  for (uint32_t i = 0; i < vehicleNodes.GetN(); ++i) {
    Ptr<Node> v = vehicleNodes.Get(i);
    Ptr<Socket> rx = Socket::CreateSocket(v, UdpSocketFactory::GetTypeId());
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), kVehUdpBasePort + i);

    if (rx->Bind(local) != 0) {
      NS_LOG_ERROR("Bind failed for veh" << i << " udp port " << (kVehUdpBasePort + i));
      continue;
    }

    rx->SetRecvCallback(MakeBoundCallback(&VehDownlinkRecvCallback, (uint32_t)i));

    g_vehDownlinkRxSock[i] = rx;
  }

  // 7) Start periodic distance updates (covariate only)
  Simulator::Schedule(Seconds(0.0), &UpdateOracle);

  if (g_lockstep)
    {
      if (g_beaconInterval > 0)
        SetupBeacons();
      LockstepControlLoop(controlPort);
      if (g_ns3Log.is_open())
        g_ns3Log.close();
      Simulator::Destroy();
      return 0;
    }

  // 8) Start POSIX control-plane Oracle server thread
  std::thread oracleThread(OracleServerThread, controlPort);

  // 9) Run
  Simulator::Stop(Seconds(24 * 3600));
  Simulator::Run();

  // 10) Shutdown
  g_running.store(false);
  if (oracleThread.joinable())
    oracleThread.join();

  if (g_ns3Log.is_open())
    g_ns3Log.close();

  Simulator::Destroy();
  return 0;
}
