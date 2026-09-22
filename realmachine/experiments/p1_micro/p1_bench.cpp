// p1_bench.cpp — P1 microbenchmark: single-operation latency/throughput of
// the Ludo view (DataPlaneLudo) and maintenance unit (ControlPlaneLudo).
//
// No model inference; pure data-structure timing on one testbed CPU.
//
// Benches (CSV rows to --out):
//   view_lookup       1-thread per-op latency (p50/p99, chrono) + ops/s,
//                     then --threads concurrent read-only ops/s.
//   request_lookup    end-to-end "request-level lookup": L× (SHA-256 level
//                     digest over 4·cut bytes + DP lookup) + full-pool
//                     scoring over W workers.  Supports the "local
//                     microsecond-scale" claim.
//   cp_insert_nomove  maintenance-unit insert, path ≤ 1 (no cuckoo move)
//   cp_insert_move    maintenance-unit insert, path > 1 (with cuckoo moves)
//   cp_update         changeValue on existing key
//   cp_remove         remove existing key
//   dp_apply_insert   replay of an insert path at an edge view
//   dp_apply_update   single-slot (bs,value) update at an edge view
//   cp_event_stream   mixed maintenance stream ops/s (70% upd/20% ins/10% rm)
//
// Load-factor sensitivity: pass --lf-list "0.85,0.90,0.95"; the table is
// built once per LF at fixed capacity (capacity is what the ctor rounds
// E/LF to; the CSV reports the ACHIEVED load factor E_used/capacity).
//
// Build: see build.sh (g++ -O3 -march=native against deps/smash).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <openssl/sha.h>

#include "Ludo/ludo_cp_dp.h"

using K = uint64_t;
using V = uint32_t;
constexpr uint VL = 32;
using CP = ControlPlaneLudo<K, V, VL>;
using DP = DataPlaneLudo<K, V, VL>;

static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
static inline K key_of(uint64_t i) {
  return splitmix64(i) & 0x7FFFFFFFFFFFFFFFULL;
}

using Clock = std::chrono::steady_clock;
static inline double ns_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

struct Stats {
  double p50, p99, mean;
  size_t n;
};
static Stats stats_of(std::vector<double>& v) {
  if (v.empty()) return {0, 0, 0, 0};
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  auto pct = [&](double q) {
    double k = (v.size() - 1) * q;
    size_t lo = (size_t)k, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (k - lo);
  };
  return {pct(0.50), pct(0.99), sum / v.size(), v.size()};
}

static FILE* g_out = nullptr;
static int g_threads = 8;
static void emit_lat(const char* bench, const char* op, uint64_t E,
                     double lf, std::vector<double>& v, const char* notes) {
  if (v.empty()) {
    fprintf(g_out, "%s,%s,%" PRIu64 ",%.4f,1,ns,,,,0,,%s;n=0\n",
            bench, op, E, lf, notes);
    fflush(g_out);
    fprintf(stderr, "  %-18s %-16s lf=%.3f  (no samples)\n", bench, op, lf);
    return;
  }
  Stats s = stats_of(v);
  fprintf(g_out, "%s,%s,%" PRIu64 ",%.4f,1,ns,%.1f,%.1f,%.1f,%zu,%.0f,%s\n",
          bench, op, E, lf, s.p50, s.p99, s.mean, s.n, 1e9 / s.mean, notes);
  fflush(g_out);
  fprintf(stderr, "  %-18s %-16s lf=%.3f  p50=%.0f p99=%.0f ns n=%zu\n",
          bench, op, lf, s.p50, s.p99, s.n);
}

static void emit(const char* bench, const char* op, uint64_t E, double lf,
                 int threads, const char* unit, const Stats& s,
                 double ops_per_s, const char* notes) {
  fprintf(g_out, "%s,%s,%" PRIu64 ",%.4f,%d,%s,%.1f,%.1f,%.1f,%zu,%.0f,%s\n",
          bench, op, E, lf, threads, unit, s.p50, s.p99, s.mean, s.n,
          ops_per_s, notes);
  fflush(g_out);
  fprintf(stderr, "  %-18s %-16s lf=%.3f thr=%d  p50=%.0f p99=%.0f %s  "
                  "(%.2fM ops/s)\n",
          bench, op, lf, threads, s.p50, s.p99, unit, ops_per_s / 1e6);
}

// ── view lookup ─────────────────────────────────────────────────────────────
static void bench_view_lookup(const DP& dp, const std::vector<K>& keys,
                              uint64_t E, double lf, int max_threads) {
  std::mt19937_64 rng(7);
  const size_t N_TIMED = std::min<size_t>(2'000'000, keys.size());
  std::vector<uint32_t> idx(N_TIMED);
  for (auto& i : idx) i = rng() % keys.size();

  volatile uint64_t sink = 0;
  // per-op timed (chrono overhead ~20-40ns; measured and reported)
  {
    Clock::time_point t0 = Clock::now();
    for (int i = 0; i < 1000; i++) sink += ns_since(t0) > 0;
    std::vector<double> ovh(10001);
    for (auto& o : ovh) {
      auto a = Clock::now();
      o = ns_since(a);
    }
    Stats os = stats_of(ovh);
    emit("clock_overhead", "steady_clock", 0, 0, 1, "ns", os, 0, "");
  }
  std::vector<double> lat(N_TIMED);
  for (size_t i = 0; i < N_TIMED; i++) {
    auto t0 = Clock::now();
    sink ^= dp.lookUp(keys[idx[i]]);
    lat[i] = ns_since(t0);
  }
  Stats s = stats_of(lat);

  // untimed loop → clean single-thread throughput
  auto th_run = [&](int nthr) {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> total{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < nthr; t++) {
      ts.emplace_back([&, t] {
        std::mt19937_64 r(100 + t);
        uint64_t local = 0, acc = 0;
        while (!stop.load(std::memory_order_relaxed)) {
          for (int k = 0; k < 1024; k++)
            acc ^= dp.lookUp(keys[r() % keys.size()]);
          local += 1024;
        }
        total += local + (acc & 1);
      });
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));
    stop = true;
    for (auto& t : ts) t.join();
    return total.load() / 3.0;
  };
  double ops1 = th_run(1);
  emit("view_lookup", "lookup", E, lf, 1, "ns", s, ops1, "read-only");
  double opsN = th_run(max_threads);
  Stats none{0, 0, 0, 0};
  char note[64];
  snprintf(note, sizeof note, "scaling=%.2fx", opsN / ops1);
  emit("view_lookup", "lookup", E, lf, max_threads, "ns", none, opsN, note);
}

// ── request-level lookup: L digests + L lookups + W-worker scoring ──────────
static void bench_request_lookup(const DP& dp, const std::vector<K>& keys,
                                 uint64_t E, double lf,
                                 const std::vector<int>& cuts, int W) {
  std::mt19937_64 rng(11);
  const int PROMPT = cuts.back();
  std::vector<int32_t> tokens(PROMPT);
  for (auto& t : tokens) t = (int32_t)(rng() % 128000);
  const size_t N = 200'000;
  std::vector<double> lat(N);
  volatile uint64_t sink = 0;
  std::vector<double> q(W), belief(W), prate(W), wan(W);
  for (int w = 0; w < W; w++) {
    q[w] = (double)(rng() % 1000);
    belief[w] = 0.9;
    prate[w] = 0.177;
    wan[w] = (w % 3) ? 80.0 : 0.0;
  }
  for (size_t i = 0; i < N; i++) {
    tokens[0] = (int32_t)i;  // vary the prefix → no trivial caching
    auto t0 = Clock::now();
    uint64_t match_key = 0;
    double best = 1e30;
    // L level digests + lookups
    std::vector<V> hits(cuts.size());
    for (size_t l = 0; l < cuts.size(); l++) {
      unsigned char d[SHA256_DIGEST_LENGTH];
      SHA256(reinterpret_cast<const unsigned char*>(tokens.data()),
             (size_t)cuts[l] * sizeof(int32_t), d);
      uint64_t k;
      memcpy(&k, d, 8);
      k &= 0x7FFFFFFFFFFFFFFFULL;
      // half the probes hit real keys (realistic mixed hit/miss cost)
      if (l % 2 == 0) k = keys[k % keys.size()];
      hits[l] = dp.lookUp(k);
      match_key ^= hits[l];
    }
    // full-pool joint scoring
    for (int w = 0; w < W; w++) {
      double em = (double)(hits[w % cuts.size()] ? cuts[w % cuts.size()] : 0)
                  * belief[w];
      double t_p = (PROMPT - em) * prate[w];
      double sc = q[w] + 1.5 * t_p + wan[w];
      if (sc < best) best = sc;
    }
    sink ^= match_key ^ (uint64_t)best;
    lat[i] = ns_since(t0);
  }
  Stats s = stats_of(lat);
  char note[96];
  snprintf(note, sizeof note, "L=%zu;W=%d;digest=sha256(4*cut B);prompt=%d",
           cuts.size(), W, PROMPT);
  emit("request_lookup", "e2e", E, lf, 1, "ns", s, 1e9 / s.mean, note);
}

// ── maintenance unit (CP) + edge view (DP) event benches ────────────────────
static void bench_events(uint64_t capacity_hint, double lf, uint64_t E_used) {
  CP cp((uint32_t)capacity_hint);
  cp.setSeed(0x5EED);

  const size_t TIMED = 200'000;
  const uint64_t warm_target = E_used > TIMED ? E_used - TIMED : 0;

  // ── phase 1: fast warm fill to just below target LF (untimed,
  // offline inserts — path recording is an online-mode feature).
  uint64_t i = 0, inserted = 0, consec = 0, fallback = 0;
  std::vector<K> present;
  present.reserve(E_used + TIMED);
  while (inserted < warm_target) {
    K k = key_of(0xABCD000000000000ULL + i++);
    auto r = cp.insert(k, (V)(inserted + 1), false);
    if (r.status < 0) {
      if (++consec > 1000) {
        fprintf(stderr, "  warm saturated %" PRIu64 "/%" PRIu64 "\n",
                inserted, warm_target);
        break;
      }
      continue;
    }
    consec = 0;
    inserted++;
    present.push_back(k);
  }
  // Edge-view mirror snapshots the warm state (authority → DP snapshot).
  DP dpm(cp);

  // ── phase 2: timed inserts at target LF, PRODUCTION protocol —
  // online=true (records the cuckoo-move path, replayed at the DP
  // mirror); a rebuild-needed status triggers the authority's rebuild
  // (emulated inline as a forced same-capacity rebuild, timed
  // separately — production runs it async off the critical path) and a
  // fresh DP snapshot.
  std::vector<double> ins_nomove, ins_move, rebuild_ms, dpi;
  uint64_t rebuilds = 0;
  for (size_t j = 0; j < TIMED; j++) {
    K k = key_of(0xABCD000000000000ULL + i++);
    V v = (V)(inserted + 1);
    auto t0 = Clock::now();
    auto r = cp.insert(k, v, true);
    double ns = ns_since(t0);
    if (r.status < 0) {
      auto t0r = Clock::now();
      cp.resizeCapacity(cp.capacity, true);
      rebuild_ms.push_back(ns_since(t0r) / 1e6);
      rebuilds++;
      dpm = DP(cp);
      t0 = Clock::now();
      r = cp.insert(k, v, true);
      ns = ns_since(t0);
      if (r.status < 0) {
        fallback++;
        continue;
      }
    }
    inserted++;
    present.push_back(k);
    // classify: path ≤1 = plain slot write; >1 = cuckoo move chain
    (r.path.size() <= 1 ? ins_nomove : ins_move).push_back(ns);
    auto t0d = Clock::now();
    dpm.applyInsert(r.path, v);
    dpi.push_back(ns_since(t0d));
  }
  double lf_real = (double)cp.size() / cp.capacity;
  char note[120];
  snprintf(note, sizeof note, "rebuilds=%" PRIu64 ";fallbacks=%" PRIu64
           ";timed=%zu", rebuilds, fallback, TIMED);
  emit_lat("cp_insert_nomove", "insert_new", cp.size(), lf_real,
           ins_nomove, note);
  emit_lat("cp_insert_move", "insert_new_cuckoo", cp.size(), lf_real,
           ins_move, note);
  if (!rebuild_ms.empty()) {
    Stats srb = stats_of(rebuild_ms);
    emit("cp_rebuild", "forced_same_capacity", cp.size(), lf_real, 1, "ms",
         srb, 0, "async_off_critical_path_in_prod");
  }
  emit_lat("dp_apply_insert", "apply_path", cp.size(), lf_real, dpi, note);

  // Load-factor sensitivity of LOOKUP too (plan item 3 covers both the
  // read and write paths): the DP mirror is exactly at lf_real here.
  bench_view_lookup(dpm, present, cp.size(), lf_real, g_threads);

  // update existing (changeValue) + dp apply_update
  std::mt19937_64 rng(23);
  if (present.empty()) return;
  const size_t NU = std::min<size_t>(300'000, present.size() * 4);
  std::vector<double> upd(NU);
  std::vector<std::pair<uint32_t, V>> bss;
  bss.reserve(NU);
  for (size_t j = 0; j < NU; j++) {
    K k = present[rng() % present.size()];
    V nv = (V)rng();
    auto t0 = Clock::now();
    auto r = cp.changeValue(k, nv);
    upd[j] = ns_since(t0);
    if (!r.path.empty()) {
      const auto& e = r.path.back();
      bss.emplace_back(((uint32_t)e.bid << 2) | (e.sid & 3u), nv);
    }
  }
  Stats su = stats_of(upd);
  emit("cp_update", "change_value", inserted, lf_real, 1, "ns", su,
       1e9 / su.mean, "existing-key");
  std::vector<double> dpu(bss.size());
  for (size_t j = 0; j < bss.size(); j++) {
    auto t0 = Clock::now();
    dpm.applyUpdate(bss[j].first, bss[j].second);
    dpu[j] = ns_since(t0);
  }
  Stats sdu = stats_of(dpu);
  emit("dp_apply_update", "apply", inserted, lf_real, 1, "ns", sdu,
       1e9 / sdu.mean, "single-slot");

  // remove
  const size_t NR = std::min<size_t>(200'000, present.size() / 2);
  std::vector<double> rem(NR);
  for (size_t j = 0; j < NR; j++) {
    K k = present.back();
    present.pop_back();
    auto t0 = Clock::now();
    cp.remove(k);
    rem[j] = ns_since(t0);
  }
  Stats sr = stats_of(rem);
  emit("cp_remove", "remove", cp.size(), (double)cp.size() / cp.capacity, 1,
       "ns", sr, 1e9 / sr.mean, "existing-key");

  // mixed event stream throughput (70% upd / 20% ins / 10% rm), 3 s
  std::atomic<uint64_t> nev{0};
  auto t_end = Clock::now() + std::chrono::seconds(3);
  uint64_t fresh = 0xC0DE000000000000ULL;
  while (Clock::now() < t_end) {
    for (int k = 0; k < 256; k++) {
      uint64_t r = rng();
      int pick = (int)(r % 10);
      if (pick < 7 && !present.empty()) {
        cp.changeValue(present[r % present.size()], (V)r);
      } else if (pick < 9 && cp.size() < cp.capacity * 0.94) {
        auto res = cp.insert(key_of(fresh++), (V)r, true);
        if (res.status >= 0) present.push_back(key_of(fresh - 1));
        // rebuild-needed events are skipped here (async in prod); their
        // rate is bounded by the cp_rebuild row above.
      } else if (!present.empty()) {
        cp.remove(present.back());
        present.pop_back();
      }
    }
    nev += 256;
  }
  Stats none{0, 0, 0, 0};
  emit("cp_event_stream", "mixed_70u20i10r", cp.size(),
       (double)cp.size() / cp.capacity, 1, "ops", none, nev.load() / 3.0, "");
}

int main(int argc, char** argv) {
  uint64_t E = 10'000'000;
  int max_threads = 8;
  std::string lf_list = "0.85,0.90,0.95";
  std::string cuts_s = "256,512,1024,2048,4096";
  std::string out = "p1_results.csv";
  int W = 14;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--E") E = strtoull(next().c_str(), nullptr, 10);
    else if (a == "--threads") { max_threads = atoi(next().c_str());
                                 g_threads = max_threads; }
    else if (a == "--lf-list") lf_list = next();
    else if (a == "--cuts") cuts_s = next();
    else if (a == "--workers") W = atoi(next().c_str());
    else if (a == "--out") out = next();
  }
  g_out = fopen(out.c_str(), "w");
  fprintf(g_out, "bench,op,E,load_factor,threads,unit,p50,p99,mean,n,"
                 "ops_per_s,notes\n");
  fflush(g_out);

  std::vector<int> cuts;
  for (size_t p = 0; p < cuts_s.size();) {
    size_t q = cuts_s.find(',', p);
    if (q == std::string::npos) q = cuts_s.size();
    cuts.push_back(atoi(cuts_s.substr(p, q - p).c_str()));
    p = q + 1;
  }

  // ── E=1e7 view: build once, lookup + request-level benches ──
  fprintf(stderr, "building CP with %" PRIu64 " entries...\n", E);
  CP cp((uint32_t)E);
  cp.setSeed(0x5EED);
  fprintf(stderr, "  capacity=%u num_buckets=%u\n", cp.capacity,
          cp.num_buckets_);
  // Never target beyond 95% of what the table can actually hold — a
  // saturated Ludo rejects (or crawls on) every further insert and the
  // naive retry loop spins forever (observed: 87 min at 99.9% CPU).
  uint64_t target = std::min<uint64_t>(E, (uint64_t)(cp.capacity * 0.95));
  std::vector<K> keys;
  keys.reserve(target);
  uint64_t i = 0, fb = 0, consec = 0;
  while (keys.size() < target) {
    K k = key_of(i++);
    // online=false: bulk build does rebuilds inline (status 1/2 = ok
    // after rebuild).  online=true NEVER rebuilds — after the first
    // rebuild-needed point every write fails (production defers those
    // to the authority's fallback + async rebuild).
    auto r = cp.insert(k, (V)(keys.size() % 0xFFFFFFFF) + 1, false);
    if (r.status < 0) {
      fb++;
      if (++consec > 1000) {
        fprintf(stderr, "  saturated at %zu entries (capacity %u) — "
                        "continuing with achieved size\n",
                keys.size(), cp.capacity);
        break;
      }
      continue;
    }
    consec = 0;
    keys.push_back(k);
    if (keys.size() % 1'000'000 == 0)
      fprintf(stderr, "  built %zuM\n", keys.size() / 1'000'000);
  }
  double lf_view = (double)keys.size() / cp.capacity;
  fprintf(stderr, "view built: E=%zu capacity=%u lf=%.3f fallbacks=%" PRIu64
                  "\n", keys.size(), cp.capacity, lf_view, fb);
  DP dp(cp);
  bench_view_lookup(dp, keys, E, lf_view, max_threads);
  bench_request_lookup(dp, keys, E, lf_view, cuts, W);

  // ── load-factor sensitivity on the maintenance unit ──
  // capacity fixed by ctor hint; E_used = lf × capacity.
  for (size_t p = 0; p < lf_list.size();) {
    size_t q = lf_list.find(',', p);
    if (q == std::string::npos) q = lf_list.size();
    double lf = atof(lf_list.substr(p, q - p).c_str());
    p = q + 1;
    CP probe(1'000'000);  // fixed 1M-hint unit → capacity rounds up
    uint64_t cap = probe.capacity;
    uint64_t E_used = (uint64_t)(lf * cap);
    fprintf(stderr, "event benches at lf=%.2f (capacity=%" PRIu64
                    ", E=%" PRIu64 ")...\n", lf, cap, E_used);
    bench_events(1'000'000, lf, E_used);
  }
  fclose(g_out);
  fprintf(stderr, "done -> %s\n", out.c_str());
  return 0;
}
