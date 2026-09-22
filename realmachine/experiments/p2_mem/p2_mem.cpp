// p2_mem.cpp — P2 memory-footprint benchmark. One STRUCTURE per process
// (clean RSS deltas): --structure ludo_view | ludo_cp | radix |
// planetserve | hashdir.
//
// All structures are fed the SAME workload: real token sequences from
// the mixed trace (results_tmp/p2_tokens.bin, see export_tokens.py),
// tenant-amplified exactly like the mt50 trace (per-tenant random
// 30–80-token prefix prepended), until the Ludo-view entry count
// (Σ_seq #grid-cuts ≤ len) reaches --E.  Per entry, holders are drawn
// deterministically (1 worker, +1 more with p=0.2, of 14).
//
// Two accountings per run:
//   rss_delta_bytes  — VmRSS after build − before build (page-level truth)
//   self_bytes       — structure's own bookkeeping of what it allocated
//
// Entry semantics per structure (reported in the CSV):
//   ludo_view / ludo_cp : one entry per (sequence, cut) chunk key
//   hashdir             : same keys as the view
//   radix               : full token sequences (all prefixes implicit)
//   planetserve         : one gate-leaf worker registration per sequence
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <openssl/sha.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "Ludo/ludo_cp_dp.h"

using K = uint64_t;
using V = uint32_t;
constexpr uint VL = 32;
using CP = ControlPlaneLudo<K, V, VL>;
using DP = DataPlaneLudo<K, V, VL>;

static const int CUTS[] = {256, 512, 1024, 2048, 4096};
static const int N_CUTS = 5;
static const int N_WORKERS = 14;

static size_t rss_bytes() {
#ifdef __APPLE__
  // phys_footprint, NOT resident_size: macOS transparently compresses
  // cold low-entropy pages (token arrays, pointer-dense trie nodes), and
  // resident_size excludes them — observed 5-10× under-reporting on the
  // radix / planetserve_eq cells.  phys_footprint accounts compressed
  // pages at their uncompressed size (Apple's "memory footprint"),
  // matching what Linux VmRSS reports on an uncompressed system.
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                (task_info_t)&info, &count) != KERN_SUCCESS)
    return 0;
  return (size_t)info.phys_footprint;
#else
  std::ifstream f("/proc/self/statm");
  size_t total, resident;
  f >> total >> resident;
  return resident * (size_t)sysconf(_SC_PAGESIZE);
#endif
}

// --fast-hash: 8-byte-stride xor-multiply mixer (xxhash-style).  Key
// VALUES don't affect any structure's footprint (cuckoo/othello/dict
// behave identically on any uniform 64-bit keys); this only removes the
// SHA-256 wall-time bottleneck at E ≥ 1e8.  Equivalence vs SHA at the
// same E is validated in the runner (bits/entry match to <1%).
static bool g_fast_hash = false;

static K fast_key(const int32_t* toks, size_t n) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(toks);
  size_t nb = n * 4;
  uint64_t h = 0x9E3779B97F4A7C15ULL ^ (nb * 0xC2B2AE3D27D4EB4FULL);
  size_t i = 0;
  for (; i + 8 <= nb; i += 8) {
    uint64_t w;
    memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 29;
  }
  for (; i < nb; i++) h = (h ^ p[i]) * 0x100000001B3ULL;
  h ^= h >> 32;
  h *= 0xD6E8FEB86659FD93ULL;
  h ^= h >> 32;
  return h & 0x7FFFFFFFFFFFFFFFULL;
}

static K chunk_key(const int32_t* toks, size_t n) {
  if (g_fast_hash) return fast_key(toks, n);
  unsigned char d[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(toks), n * 4, d);
  K k;
  memcpy(&k, d, 8);
  return k & 0x7FFFFFFFFFFFFFFFULL;
}

// ── corpus: base sequences + on-the-fly tenant amplification ────────────────
struct Corpus {
  std::vector<std::vector<int32_t>> base;
  void load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    uint32_t n;
    f.read((char*)&n, 4);
    base.resize(n);
    for (auto& s : base) {
      uint32_t len;
      f.read((char*)&len, 4);
      s.resize(len);
      f.read((char*)s.data(), (size_t)len * 4);
    }
  }
  // sequence for (tenant, base_idx): tenant prefix + base tokens
  std::vector<int32_t> seq(uint32_t tenant, uint32_t bi) const {
    std::mt19937_64 r(0x7EA0000ULL + tenant);
    size_t plen = 30 + r() % 51;                 // 30–80, mt50-style
    const auto& b = base[bi];
    std::vector<int32_t> s;
    s.reserve(plen + b.size());
    for (size_t i = 0; i < plen; i++) s.push_back((int32_t)(r() % 128000));
    s.insert(s.end(), b.begin(), b.end());
    return s;
  }
};

static int holders_of(uint64_t eid, uint32_t* out) {  // 1–2 workers
  out[0] = (uint32_t)(eid % N_WORKERS);
  if (eid % 5 == 0) { out[1] = (uint32_t)((eid / 7) % N_WORKERS); return 2; }
  return 1;
}

// ── structures ──────────────────────────────────────────────────────────────
struct SelfSized {
  virtual void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
                       uint64_t& entries) = 0;
  virtual size_t self_bytes() const = 0;
  virtual const char* entry_semantics() const = 0;
  virtual ~SelfSized() = default;
};

// 1) Ludo maintenance unit (CP), and 2) view (DP built from CP at end).
struct LudoStruct : SelfSized {
  std::unique_ptr<CP> cp;
  bool view_mode;
  std::unique_ptr<DP> dp;
  size_t rss_pre_dp = 0;   // view mode: RSS right before DP construction
  explicit LudoStruct(uint64_t E, bool view) : view_mode(view) {
    // Hint exactly E: capacities move in ~2× steps (0.95 × 4·2^k), a
    // padded hint can jump a whole tier (observed: E=1e7 ×1.06 → LF .39).
    cp = std::make_unique<CP>((uint32_t)E);
    cp->setSeed(0x5EED);
  }
  void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
               uint64_t& entries) override {
    uint32_t w[2];
    for (int c = 0; c < N_CUTS; c++) {
      if ((size_t)CUTS[c] > s.size()) break;
      K k = chunk_key(s.data(), CUTS[c]);
      int nw = holders_of(eid++, w);
      V v = 0;
      for (int j = 0; j < nw; j++) v |= (V)0b01 << (2 * w[j]);
      auto r = cp->insert(k, v, false);
      if (r.status >= 0) entries++;
    }
  }
  void finalize() {
    if (view_mode) {
      rss_pre_dp = rss_bytes();
      dp = std::make_unique<DP>(*cp);
    }
    // Sizing ground truth for the ludo_view_alloc formula (validated at
    // small E, then trusted at E where the CP no longer fits in RAM).
    fprintf(stderr,
            "  [sizing] nb=%u oth_ma=%u oth_mb=%u oth_mem_words=%zu\n",
            cp->num_buckets_, cp->locator.ma, cp->locator.mb,
            (size_t)cp->locator.mem.size());
  }
  size_t self_bytes() const override {
    // DP: per bucket 1B seed + 4×4B values (+ padding) + othello arrays.
    // CP: adds 4×8B keys + occupancy per bucket.
    size_t oth = cp->locator.mem.size() * 8;
    if (view_mode)
      return cp->num_buckets_ * (sizeof(typename DP::Bucket)) + oth;
    return cp->num_buckets_ * (sizeof(typename CP::Bucket)) + oth;
  }
  const char* entry_semantics() const override { return "seq_x_cut_keys"; }
};

// 1b) ludo_view_alloc — the edge VIEW's exact memory layout for E entries,
// allocated directly (DP bucket array + Othello bit arrays) and page-
// touched, WITHOUT building the 2.5×-larger maintenance CP first.  The
// view's layout is a deterministic function of E alone (contents don't
// change its size), so this measures the real 1e9-scale view RSS on a
// machine where the CP build would not fit.  Formulas mirror
// LudoCommon/ControlPlaneOthello::resizeCapacity and are validated
// against the [sizing] line of real builds at small E.
struct LudoAllocStruct : SelfSized {
  std::vector<typename DP::Bucket> buckets;
  std::vector<uint64_t> oth_mem;
  uint64_t nb = 64, ma = 0, mb = 0;
  // `keys` = expected DISTINCT key count (the Othello arrays track distinct
  // keys, ~2x-sized on the grow path: resizeCapacity(nKeysInOthello*2)); the
  // caller passes E x measured distinct-ratio from the largest real build.
  explicit LudoAllocStruct(uint64_t E, uint64_t keys) {
    while (nb * CP::kLoadFactor * CP::kSlotsPerBucket < (double)E) nb <<= 1;
    mb = 256;                 // pow2 >= 2x distinct keys (grow-path sizing)
    while (mb < 2 * keys) mb <<= 1;
    ma = (uint64_t)(1.33334 * mb);
    buckets.assign(nb, typename DP::Bucket{});
    oth_mem.assign(((ma + mb) * 1 /*VDL*/ + 63) / 64, 0);
    // WRITE-touch every page so the footprint reflects the full working
    // set (a read of an untouched zero-fill page maps the shared zero
    // page and never becomes dirty/resident).
    const size_t pg = 4096;
    auto* bp = reinterpret_cast<uint8_t*>(buckets.data());
    for (size_t i = 0; i < nb * sizeof(typename DP::Bucket); i += pg)
      bp[i] = (uint8_t)(i >> 12);
    auto* op = reinterpret_cast<uint8_t*>(oth_mem.data());
    for (size_t i = 0; i < oth_mem.size() * 8; i += pg)
      op[i] = (uint8_t)(i >> 12);
  }
  void add_seq(const std::vector<int32_t>&, uint64_t&, uint64_t&) override {}
  size_t self_bytes() const override {
    return nb * sizeof(typename DP::Bucket) + oth_mem.size() * 8;
  }
  const char* entry_semantics() const override { return "view_layout_alloc"; }
};

// 3) general token-level radix tree (SGLang RadixAttention organization:
// nodes hold a token segment + child map + (worker,tier) holder list).
struct RadixStruct : SelfSized {
  struct Node {
    std::vector<int32_t> seg;
    std::unordered_map<int32_t, Node*> ch;
    std::vector<std::pair<uint32_t, uint8_t>> holders;
  };
  Node root;
  size_t n_nodes = 1, seg_tokens = 0, n_holders = 0;
  void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
               uint64_t& entries) override {
    Node* cur = &root;
    size_t i = 0;
    while (i < s.size()) {
      auto it = cur->ch.find(s[i]);
      if (it == cur->ch.end()) {
        Node* n = new Node();
        n->seg.assign(s.begin() + i, s.end());
        seg_tokens += n->seg.size();
        cur->ch.emplace(s[i], n);
        n_nodes++;
        cur = n;
        i = s.size();
      } else {
        Node* n = it->second;
        size_t m = 0;
        while (m < n->seg.size() && i + m < s.size() && n->seg[m] == s[i + m])
          m++;
        if (m < n->seg.size()) {          // split
          Node* tail = new Node();
          tail->seg.assign(n->seg.begin() + m, n->seg.end());
          tail->ch.swap(n->ch);
          tail->holders.swap(n->holders);
          n->seg.resize(m);
          n->ch.emplace(tail->seg[0], tail);
          n_nodes++;
        }
        i += m;
        cur = n;
      }
    }
    uint32_t w[2];
    int nw = holders_of(eid++, w);
    for (int j = 0; j < nw; j++) {
      cur->holders.emplace_back(w[j], 1);
      n_holders++;
    }
    entries++;                            // one full sequence
  }
  size_t self_bytes() const override {
    // node struct + seg tokens + hashmap node overhead (~56 B/slot
    // libstdc++) + holder vec payload
    return n_nodes * (sizeof(Node) + 56) + seg_tokens * 4 + n_holders * 8;
  }
  const char* entry_semantics() const override { return "full_sequences"; }
};

// 3b) radix_eq — the SGLang-style token radix tree made information-
// equivalent to the Ludo view, in BOTH directions: indexed depth capped
// at the deepest grid cut (the view stores nothing past 4096 tokens; the
// plain 'radix' arm stores full 20k-token sequences = surplus info), and
// holders registered at every cut-boundary node — one registration per
// (sequence, cut) — instead of once per sequence tail (deficit info).
// Node organization (path-compressed segment + child map + holder list)
// is unchanged from RadixStruct.
struct RadixEqStruct : SelfSized {
  struct Node {
    std::vector<int32_t> seg;
    std::unordered_map<int32_t, Node*> ch;
    std::vector<std::pair<uint32_t, uint8_t>> holders;
  };
  Node root;
  size_t n_nodes = 1, seg_tokens = 0, n_holders = 0;
  // Insert s[0..n) and return the node ending exactly at depth n,
  // splitting path-compressed segments where the boundary falls inside.
  Node* insert_prefix(const std::vector<int32_t>& s, size_t n) {
    Node* cur = &root;
    size_t i = 0;
    while (i < n) {
      auto it = cur->ch.find(s[i]);
      if (it == cur->ch.end()) {
        Node* nn = new Node();
        nn->seg.assign(s.begin() + i, s.begin() + n);
        seg_tokens += nn->seg.size();
        cur->ch.emplace(s[i], nn);
        n_nodes++;
        return nn;
      }
      Node* nd = it->second;
      size_t m = 0;
      while (m < nd->seg.size() && i + m < n && nd->seg[m] == s[i + m]) m++;
      if (m < nd->seg.size()) {          // boundary or divergence inside seg
        Node* tail = new Node();
        tail->seg.assign(nd->seg.begin() + m, nd->seg.end());
        tail->ch.swap(nd->ch);
        tail->holders.swap(nd->holders);
        nd->seg.resize(m);
        nd->ch.emplace(tail->seg[0], tail);
        n_nodes++;
        if (i + m == n) return nd;       // cut boundary → nd ends at n
        i += m;
        cur = nd;                        // divergence → branch on next loop
        continue;
      }
      i += m;
      cur = nd;
      if (i == n) return cur;
    }
    return cur;
  }
  void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
               uint64_t& entries) override {
    uint32_t w[2];
    for (int c = 0; c < N_CUTS; c++) {
      if ((size_t)CUTS[c] > s.size()) break;
      Node* nd = insert_prefix(s, (size_t)CUTS[c]);
      int nw = holders_of(eid++, w);
      for (int j = 0; j < nw; j++) {
        nd->holders.emplace_back(w[j], 1);
        n_holders++;
      }
      entries++;
    }
  }
  size_t self_bytes() const override {
    return n_nodes * (sizeof(Node) + 56) + seg_tokens * 4 + n_holders * 8;
  }
  const char* entry_semantics() const override { return "seq_x_cut_regs"; }
};

// 4) PlanetServe HashRadixTree — REIMPLEMENTED from their design
// (HashRadixTree.py:13-24 node = {children: Dict[hash,Node], model_list,
// is_end, parent}; chunk_size=10 tokens, gate=10 chunks, hash mod 15).
// MIT-licensed original; design only, no code linked.
struct PlanetStruct : SelfSized {
  struct Node {
    std::unordered_map<int, Node*> ch;
    std::vector<uint32_t> models;
    Node* parent = nullptr;
    bool is_end = false;
  };
  Node root;
  size_t n_nodes = 1, n_models = 0;
  static int hash_chunk(const int32_t* t, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (uint64_t)t[i]; h *= 1099511628211ULL; }
    return (int)(h % 15);
  }
  void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
               uint64_t& entries) override {
    Node* cur = &root;
    size_t nch = std::min<size_t>(10, s.size() / 10);   // gate=10
    for (size_t c = 0; c < nch; c++) {
      int h = hash_chunk(s.data() + c * 10, 10);
      auto it = cur->ch.find(h);
      if (it == cur->ch.end()) {
        Node* n = new Node();
        n->parent = cur;
        cur->ch.emplace(h, n);
        n_nodes++;
        cur = n;
      } else cur = it->second;
    }
    cur->is_end = true;
    uint32_t w[2];
    int nw = holders_of(eid++, w);
    for (int j = 0; j < nw; j++) {
      cur->models.push_back(w[j]);
      n_models++;
    }
    entries++;                            // one gate-leaf registration
  }
  size_t self_bytes() const override {
    return n_nodes * (sizeof(Node) + 56) + n_models * 4;
  }
  const char* entry_semantics() const override { return "gate_leaf_regs"; }
};

// 4b) planetserve_eq — PlanetServe's node organization (HashRadixTree.py:
// children dict + model_list + parent, chunk_size=10, NO path compression),
// configured to carry the SAME information as the Ludo view: full 64-bit
// chunk hashes (no mod-15 folding), path depth to the deepest grid cut
// (4096 tok = 410 chunks) instead of the 10-chunk gate, and holders
// registered at every cut-boundary node — one registration per
// (sequence, cut), exactly the view's entry set.  The original
// 'planetserve' arm (gate=10, mod 15) is kept for reference; this arm
// answers "what does the HRT cost when it must be as informative as the
// MPH view".
struct PlanetEqStruct : SelfSized {
  struct Node {
    std::unordered_map<uint64_t, Node*> ch;
    std::vector<uint32_t> models;
    Node* parent = nullptr;
    bool is_end = false;
  };
  Node root;
  size_t n_nodes = 1, n_models = 0;
  static uint64_t hash_chunk64(const int32_t* t, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
      h ^= (uint64_t)(uint32_t)t[i];
      h *= 1099511628211ULL;
    }
    return h;
  }
  void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
               uint64_t& entries) override {
    int max_ci = -1;
    for (int c = 0; c < N_CUTS; c++)
      if ((size_t)CUTS[c] <= s.size()) max_ci = c;
    if (max_ci < 0) return;
    // ceil-chunks to the deepest cut (cuts are not multiples of 10; the
    // trailing partial chunk carries the last <10 tokens of the cut).
    size_t nch = ((size_t)CUTS[max_ci] + 9) / 10;
    Node* cur = &root;
    int ci = 0;
    uint32_t w[2];
    for (size_t c = 0; c < nch; c++) {
      size_t rem = s.size() - c * 10;
      uint64_t h = hash_chunk64(s.data() + c * 10, rem < 10 ? rem : 10);
      auto it = cur->ch.find(h);
      if (it == cur->ch.end()) {
        Node* n = new Node();
        n->parent = cur;
        cur->ch.emplace(h, n);
        n_nodes++;
        cur = n;
      } else cur = it->second;
      // Holder registration at each cut boundary ((c+1)*10 tokens covers
      // CUTS[ci]) — mirrors the view's per-(seq,cut) entries incl. the
      // shared eid-driven holder draw.
      if (ci <= max_ci && (c + 1) * 10 >= (size_t)CUTS[ci]) {
        cur->is_end = true;
        int nw = holders_of(eid++, w);
        for (int j = 0; j < nw; j++) {
          cur->models.push_back(w[j]);
          n_models++;
        }
        entries++;
        ci++;
      }
    }
  }
  size_t self_bytes() const override {
    return n_nodes * (sizeof(Node) + 56) + n_models * 4;
  }
  const char* entry_semantics() const override { return "seq_x_cut_regs"; }
};

// 5) naive hash directory
struct HashDirStruct : SelfSized {
  std::unordered_map<K, std::vector<std::pair<uint32_t, uint8_t>>> dir;
  size_t n_holders = 0;
  void add_seq(const std::vector<int32_t>& s, uint64_t& eid,
               uint64_t& entries) override {
    uint32_t w[2];
    for (int c = 0; c < N_CUTS; c++) {
      if ((size_t)CUTS[c] > s.size()) break;
      K k = chunk_key(s.data(), CUTS[c]);
      auto& v = dir[k];
      int nw = holders_of(eid++, w);
      for (int j = 0; j < nw; j++) { v.emplace_back(w[j], 1); n_holders++; }
      entries++;
    }
  }
  size_t self_bytes() const override {
    // bucket array + per-node (key + vector header + libstdc++ node ovh)
    return dir.bucket_count() * 8 +
           dir.size() * (8 + sizeof(std::vector<std::pair<uint32_t, uint8_t>>)
                         + 32) + n_holders * 8;
  }
  const char* entry_semantics() const override { return "seq_x_cut_keys"; }
};

int main(int argc, char** argv) {
  std::string structure, tokens = "results_tmp/p2_tokens.bin";
  uint64_t E = 1'000'000;
  uint64_t alloc_keys = 0;                 // ludo_view_alloc: distinct keys
  double mem_cap_gb = 0.0;                 // 0 = no cap
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--structure") structure = next();
    else if (a == "--E") E = strtoull(next().c_str(), nullptr, 10);
    else if (a == "--tokens") tokens = next();
    else if (a == "--fast-hash") g_fast_hash = true;
    else if (a == "--mem-cap-gb") mem_cap_gb = atof(next().c_str());
    else if (a == "--alloc-keys") alloc_keys = strtoull(next().c_str(), nullptr, 10);
  }
  Corpus corpus;
  corpus.load(tokens.c_str());
  size_t rss0 = rss_bytes();
  size_t cap_bytes = (size_t)(mem_cap_gb * (1ULL << 30));

  std::unique_ptr<SelfSized> st;
  LudoStruct* ludo = nullptr;
  bool alloc_mode = false;
  if (structure == "ludo_view" || structure == "ludo_cp") {
    auto p = std::make_unique<LudoStruct>(E, structure == "ludo_view");
    ludo = p.get();
    st = std::move(p);
  } else if (structure == "ludo_view_alloc") {
    st = std::make_unique<LudoAllocStruct>(E, alloc_keys ? alloc_keys : E);
    alloc_mode = true;
  } else if (structure == "radix") st = std::make_unique<RadixStruct>();
  else if (structure == "radix_eq") st = std::make_unique<RadixEqStruct>();
  else if (structure == "planetserve") st = std::make_unique<PlanetStruct>();
  else if (structure == "planetserve_eq") st = std::make_unique<PlanetEqStruct>();
  else if (structure == "hashdir") st = std::make_unique<HashDirStruct>();
  else { fprintf(stderr, "unknown --structure %s\n", structure.c_str()); return 1; }

  // Feed identical workload order to every structure: round-robin
  // (tenant, base) until the VIEW-entry counter (seq × cuts) reaches E.
  // (alloc mode skips the feed — the view's layout is content-independent.)
  uint64_t eid = 0, entries = 0, view_entries = 0, seqs = 0;
  uint32_t tenant = 0, bi = 0;
  while (!alloc_mode && view_entries < E) {
    auto s = corpus.seq(tenant, bi);
    int cuts = 0;
    for (int c = 0; c < N_CUTS; c++) if ((size_t)CUTS[c] <= s.size()) cuts++;
    if (cuts == 0) { bi = (bi + 1) % corpus.base.size(); if (!bi) tenant++; continue; }
    st->add_seq(s, eid, entries);
    view_entries += cuts;
    seqs++;
    bi = (bi + 1) % corpus.base.size();
    if (bi == 0) tenant++;
    if (seqs % 200000 == 0) {
      size_t r = rss_bytes();
      fprintf(stderr, "  %s: %" PRIu64 " seqs, %" PRIu64 " view-entries, rss=%zuMB\n",
              structure.c_str(), seqs, view_entries, r >> 20);
      if (cap_bytes && r > cap_bytes) {
        // Graceful OOM guard: emit a parsable abort row and exit 2 —
        // the cell is reported as "does not fit" instead of swap-death.
        printf("%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%zu,%zu,%.1f,%.1f,%.3f,%s\n",
               structure.c_str(), view_entries, entries, seqs, tenant + 1,
               (size_t)0, (size_t)0, 0.0, 0.0, 0.0, "OOM_ABORT");
        fflush(stdout);
        return 2;
      }
    }
  }
  if (alloc_mode) { view_entries = E; entries = E; }
  if (ludo) ludo->finalize();
  size_t rss1 = rss_bytes();

  double lf = 0;
  if (ludo) lf = (double)ludo->cp->size() / ludo->cp->capacity;
  // CSV row: structure,E_view,entries_native,seqs,tenants,rss_delta,self,
  //          bits_per_view_entry(rss),bits_per_view_entry(self),lf,semantics
  size_t rssd = rss1 > rss0 ? rss1 - rss0 : 0;
  if (ludo && ludo->view_mode)
    // The view is the DP alone — the CP in this process exists only to
    // derive it (production edge views deserialize a snapshot).
    rssd = rss1 > ludo->rss_pre_dp ? rss1 - ludo->rss_pre_dp : 0;
  size_t self = st->self_bytes();
  printf("%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%zu,%zu,%.1f,%.1f,%.3f,%s\n",
         structure.c_str(), view_entries, entries, seqs, tenant + 1,
         rssd, self,
         8.0 * rssd / view_entries, 8.0 * self / view_entries, lf,
         st->entry_semantics());
  return 0;
}
