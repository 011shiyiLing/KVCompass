/**
 * ludo_binding.cpp — pybind11 wrapper around Smash's ControlPlaneLudo + DataPlaneLudo.
 *
 * This binding underpins the L1+L2 maintenance plane (authority / lookup
 * separation, path-entry incremental protocol, fallback + async rebuild).
 * It replaces the older single-class wrapper around ControlPlaneMinimalPerfectCuckoo.
 *
 * Concrete instantiation (selected by -DLUDO_VALUE_BITS, default 32)
 *   K  = uint64_t   (chunk_to_key — SHA-256 truncated to 63 bits)
 *   V  = uint32_t / uint64_t (packed: 16/32 worker × 2-bit tier;
 *                             bit 2*i..2*i+1 = worker i)
 *   VL = 32 / 64    (all value bits stored)
 *
 * LUDO_VALUE_BITS=32 builds module `ludo_binding`  (16 workers);
 * LUDO_VALUE_BITS=64 builds module `ludo_binding64` (32 workers) — same
 * source, same API, wider packed value.
 *
 * Two Python classes are exported:
 *
 *   LudoControlPlane  — single writer.  `insert` returns the cuckoo-move
 *                       path needed to apply the insert at the data plane;
 *                       `locate` (a.k.a. lookUpBucketSlot) gives (bid, slot)
 *                       for a known key so the authority can emit single-slot
 *                       Update messages; `serialize_dp` produces the bytes
 *                       a fresh DP can deserialize from after a rebuild.
 *
 *   LudoDataPlane     — read-only mirror.  `apply_insert(path, value)`,
 *                       `apply_update(bs, value)`, `deserialize(bytes)`.
 *
 * The legacy `ludo_table.LudoTable` (gossip mode) keeps using
 * `LudoControlPlane.insert/lookup/update/remove/size`; the dict it now
 * returns from insert/remove/update is ignored on its old call sites.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Ludo/ludo_cp_dp.h"

namespace py = pybind11;

#ifndef LUDO_VALUE_BITS
#define LUDO_VALUE_BITS 32
#endif

using K = uint64_t;
#if LUDO_VALUE_BITS == 64
using V           = uint64_t;
constexpr uint VL = 64;
#define LUDO_MODULE_NAME ludo_binding64
#define LUDO_MODULE_DOC \
    "Smash-flavour Ludo CP+DP binding — VL=64 (32 workers × 2-bit tier)."
#elif LUDO_VALUE_BITS == 32
using V           = uint32_t;
constexpr uint VL = 32;
#define LUDO_MODULE_NAME ludo_binding
#define LUDO_MODULE_DOC \
    "Smash-flavour Ludo CP+DP binding — VL=32 (16 workers × 2-bit tier)."
#else
#error "LUDO_VALUE_BITS must be 32 or 64"
#endif

using SmashCP    = ControlPlaneLudo<K, V, VL>;
using SmashDP    = DataPlaneLudo<K, V, VL>;
using PathEntry  = Ludo_PathEntry<K>;
using UpdateRes  = LudoUpdateResult<K>;


// ── Helpers ──────────────────────────────────────────────────────────────────
static py::dict path_entry_to_dict(const PathEntry &e) {
    py::dict d;
    d["status"]    = e.status;
    d["bid"]       = static_cast<uint32_t>(e.bid);
    d["sid"]       = static_cast<uint8_t>(e.sid);
    d["newSeed"]   = e.newSeed;
    d["s0"]        = static_cast<uint8_t>(e.s0);
    d["s1"]        = static_cast<uint8_t>(e.s1);
    d["s2"]        = static_cast<uint8_t>(e.s2);
    d["s3"]        = static_cast<uint8_t>(e.s3);
    d["locatorCC"] = py::cast(e.locatorCC);
    return d;
}

static PathEntry dict_to_path_entry(const py::dict &d) {
    PathEntry e;
    e.status    = d["status"].cast<int>();
    e.bid       = d["bid"].cast<uint32_t>();
    e.sid       = d["sid"].cast<uint8_t>();
    e.newSeed   = d["newSeed"].cast<uint8_t>();
    e.s0        = d["s0"].cast<uint8_t>();
    e.s1        = d["s1"].cast<uint8_t>();
    e.s2        = d["s2"].cast<uint8_t>();
    e.s3        = d["s3"].cast<uint8_t>();
    e.locatorCC = d["locatorCC"].cast<std::vector<uint32_t>>();
    return e;
}

static py::dict result_to_dict(const UpdateRes &r) {
    py::list path;
    for (const auto &e : r.path) {
        path.append(path_entry_to_dict(e));
    }
    py::dict out;
    out["status"] = r.status;
    out["path"]   = path;
    return out;
}


// ── ControlPlane wrapper ─────────────────────────────────────────────────────
class LudoControlPlane {
public:
    explicit LudoControlPlane(uint32_t expected_entries = 1024,
                              uint32_t seed = 0x12345678U)
        : cp_(std::max<uint32_t>(expected_entries, 64U)) {
        cp_.setSeed(seed);
        // The Smash ctor sizes the cuckoo buckets for `capacity` but the
        // Othello locator only for the initial key set — which is EMPTY
        // here, so online inserts start failing with status -1 after a few
        // hundred keys.  Rebuild the locator at full capacity up front
        // (nKeys == 0, so this is cheap).
        cp_.resizeCapacity(std::max<uint32_t>(expected_entries, 64U),
                           /*forceBuild=*/true);
    }

    /// Insert key→value. Returns dict {status, path[]}.
    /// status==0  ok, path is the cuckoo-move sequence to replay at the DP
    /// status==-1 othello rebuild needed, fallback the write
    /// status==-2 cuckoo enlarge needed, fallback the write
    /// status==1/2 ok after offline rebuild (used only with online=false).
    py::dict insert(K key, V value) {
        UpdateRes r = cp_.insert(key, value, /*online=*/true);
        return result_to_dict(r);
    }

    /// Lookup. Returns int value, or None.
    py::object lookup(K key) const {
        V out = 0;
        if (cp_.lookUp(key, out)) {
            return py::cast(out);
        }
        return py::none();
    }

    /// (bucket_id, slot) for a known key, or None on miss.
    /// This is what the authority needs to emit single-slot Update messages.
    py::object locate(K key) const {
        auto p = cp_.locate(key);
        if (p.first == uint32_t(-1)) return py::none();
        return py::make_tuple(p.first, p.second);
    }

    /// Convenience: (bid<<2)|sid.
    py::object locate_bs(K key) const {
        auto p = cp_.locate(key);
        if (p.first == uint32_t(-1)) return py::none();
        return py::cast((p.first << 2) | (p.second & 0x3U));
    }

    /// Remove a key. Returns dict {status, path[]} (Smash's CP::remove
    /// returns an empty result on success; callers should rely on a prior
    /// `lookup` to know whether the key was there).
    py::dict remove(K key) {
        UpdateRes r = cp_.remove(key);
        return result_to_dict(r);
    }

    /// Backwards-compat wrapper for the legacy LudoTable.  Returns True
    /// iff the key was present prior to removal.
    bool remove_bool(K key) {
        V tmp = 0;
        bool present = cp_.lookUp(key, tmp);
        cp_.remove(key);
        return present;
    }

    /// Update an existing key's value. Raises if key not present (matches
    /// the old binding's contract used by ludo_table.LudoTable).
    py::dict update(K key, V value) {
        V tmp = 0;
        if (!cp_.lookUp(key, tmp)) {
            throw std::runtime_error("LudoControlPlane::update: key not found");
        }
        UpdateRes r = cp_.changeValue(key, value);
        return result_to_dict(r);
    }

    /// Same as `update` but returns the (bid, sid) of the updated slot
    /// directly. Convenience: avoids a second `locate` call from Python.
    py::object change_value(K key, V value) {
        UpdateRes r = cp_.changeValue(key, value);
        if (r.path.empty()) return py::none();
        const auto &e = r.path.back();
        py::dict out;
        out["status"] = r.status;
        out["bid"]    = static_cast<uint32_t>(e.bid);
        out["sid"]    = static_cast<uint8_t>(e.sid);
        out["bs"]     = (static_cast<uint32_t>(e.bid) << 2) | (e.sid & 0x3U);
        return out;
    }

    void resize_capacity(uint32_t target_capacity, bool force_build = false) {
        cp_.resizeCapacity(target_capacity, force_build);
    }

    /// Merge a Python dict[key→value] into the CP. Entries with value==0
    /// are interpreted as tombstones and removed (mirrors the
    /// `is_valid(packed) = packed != 0` predicate the authority uses on its
    /// fallback table after a rebuild).
    void merge_dict(py::dict d) {
        std::unordered_map<K, V> m;
        m.reserve(py::len(d));
        for (auto item : d) {
            m.emplace(item.first.cast<K>(), item.second.cast<V>());
        }
        cp_.Merge(m, [](const V &v) -> bool { return v != 0; });
    }

    /// Serialize the CP into bytes a fresh LudoDataPlane.deserialize() can
    /// consume.  Layout (little-endian, packed):
    ///   u64 hab.s | u32 hd.s | u32 ma | u32 mb | u64 mem_n | u64[] mem
    ///   u64 h.s | u64 digestH.s | u32 num_buckets
    ///   per bucket: u8 seed | V[4] values (already permuted to DP slot order)
    /// (hab is FastHasher64/Hasher64 with uint64_t seed; hd is Hasher32
    /// with uint32_t seed — see deps/smash/Othello/othello_cp_dp.h).
    py::bytes serialize_dp() const {
        std::ostringstream oss(std::ios::binary);
        const auto &loc = cp_.locator;

        write_u64(oss, loc.hab.s);
        write_u32(oss, loc.hd.s);
        write_u32(oss, loc.ma);
        write_u32(oss, loc.mb);

        const uint64_t mem_n = loc.mem.size();
        write_u64(oss, mem_n);
        if (mem_n) {
            oss.write(reinterpret_cast<const char *>(loc.mem.data()),
                      static_cast<std::streamsize>(mem_n * sizeof(uint64_t)));
        }

        write_u64(oss, cp_.h.s);
        write_u64(oss, cp_.digestH.s);
        write_u32(oss, cp_.num_buckets_);

        for (uint32_t bid = 0; bid < cp_.num_buckets_; ++bid) {
            const auto &cb = cp_.buckets_[bid];
            uint8_t seed = cb.seed;
            oss.write(reinterpret_cast<const char *>(&seed), 1);

            V vals[4] = {0, 0, 0, 0};
            FastHasher64<K> locate(seed);
            for (uint8_t s = 0; s < 4; ++s) {
                if (cb.occupiedMask & (1u << s)) {
                    const uint8_t dp_slot =
                        static_cast<uint8_t>(locate(cb.keys[s]) >> 62);
                    vals[dp_slot] = cb.values[s];
                }
            }
            oss.write(reinterpret_cast<const char *>(vals), sizeof(vals));
        }
        return py::bytes(oss.str());
    }

    uint32_t size() const         { return cp_.size(); }
    uint32_t num_buckets() const  { return cp_.num_buckets_; }
    uint32_t capacity() const     { return cp_.capacity; }

private:
    SmashCP cp_;

    static void write_u32(std::ostream &os, uint32_t v) {
        os.write(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    static void write_u64(std::ostream &os, uint64_t v) {
        os.write(reinterpret_cast<const char *>(&v), sizeof(v));
    }
};


// ── DataPlane wrapper ────────────────────────────────────────────────────────
class LudoDataPlane {
public:
    LudoDataPlane() = default;

    /// Lookup. Returns int (the packed value); None when value is 0
    /// (treated as "no holder", matches the bitmap-zero semantics).
    py::object lookup(K key) const {
        if (dp_.num_buckets_ == 0) return py::none();
        V out = dp_.lookUp(key);
        if (out == 0) return py::none();
        return py::cast(out);
    }

    /// Apply a path-entry sequence + value (mode-0 Insert from authority).
    /// path is a list of dicts as produced by LudoControlPlane.insert().
    void apply_insert(py::list path, V value) {
        std::vector<PathEntry> entries;
        entries.reserve(py::len(path));
        for (auto h : path) {
            entries.emplace_back(dict_to_path_entry(h.cast<py::dict>()));
        }
        dp_.applyInsert(entries, value);
    }

    /// Apply a single (bs, value) update. bs = (bid<<2)|sid.
    /// Used for: existing-key value change, and Remove-as-Update(value=0).
    void apply_update(uint32_t bs, V value) {
        dp_.applyUpdate(bs, value);
    }

    /// Restore from a CP-serialized snapshot (authority's UpdateLudo broadcast).
    void deserialize(py::bytes data) {
        std::string s(data);
        const char *p = s.data();
        const size_t n = s.size();
        size_t off = 0;

        auto need = [&](size_t k) {
            if (off + k > n) {
                throw std::runtime_error(
                    "LudoDataPlane::deserialize: truncated input");
            }
        };
        auto read_u32 = [&]() {
            need(4);
            uint32_t v;
            std::memcpy(&v, p + off, 4);
            off += 4;
            return v;
        };
        auto read_u64 = [&]() {
            need(8);
            uint64_t v;
            std::memcpy(&v, p + off, 8);
            off += 8;
            return v;
        };
        auto read_u8 = [&]() {
            need(1);
            uint8_t v = static_cast<uint8_t>(p[off]);
            off += 1;
            return v;
        };

        auto &loc = dp_.locator;
        loc.hab.s = read_u64();
        loc.hd.s  = read_u32();    // Hasher32: uint32_t seed
        loc.ma    = read_u32();
        loc.mb    = read_u32();

        const uint64_t mem_n = read_u64();
        need(mem_n * sizeof(uint64_t));
        loc.mem.resize(mem_n);
        if (mem_n) {
            std::memcpy(loc.mem.data(), p + off,
                        mem_n * sizeof(uint64_t));
            off += mem_n * sizeof(uint64_t);
        }

        dp_.h.s        = read_u64();
        dp_.digestH.s  = read_u64();
        const uint32_t nb = read_u32();
        dp_.num_buckets_ = nb;
        dp_.buckets.assign(nb, typename SmashDP::Bucket{});
        for (uint32_t bid = 0; bid < nb; ++bid) {
            uint8_t seed = read_u8();
            need(sizeof(V) * 4);
            V vals[4];
            std::memcpy(vals, p + off, sizeof(vals));
            off += sizeof(vals);
            dp_.buckets[bid].seed = seed;
            for (int i = 0; i < 4; ++i) {
                dp_.buckets[bid].values[i] = vals[i];
            }
        }
    }

    uint32_t num_buckets() const { return dp_.num_buckets_; }

private:
    SmashDP dp_;
};


PYBIND11_MODULE(LUDO_MODULE_NAME, m) {
    m.doc() = LUDO_MODULE_DOC;

    // module_local: ludo_binding and ludo_binding64 define C++ classes with
    // identical mangled names; without this tag pybind11's cross-module type
    // registry rejects whichever module is imported second.
    py::class_<LudoControlPlane>(m, "LudoControlPlane", py::module_local())
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("expected_entries") = 1024,
             py::arg("seed") = 0x12345678U,
             "Create a CP table sized for ~expected_entries keys with the "
             "given Smash-style locator seed (per-shard seed isolates "
             "cuckoo placement across shards; mirrors Smash's "
             "`ludo.setSeed(thisId * 0xe2211)` in master_fn.h).")
        .def("insert",          &LudoControlPlane::insert,
             py::arg("key"), py::arg("value"),
             "Insert key→value. Returns {'status', 'path'}; status<0 means "
             "the caller must fallback the write and trigger rebuild.")
        .def("lookup",          &LudoControlPlane::lookup,
             py::arg("key"))
        .def("locate",          &LudoControlPlane::locate,
             py::arg("key"),
             "Return (bid, slot) for a known key. Used to emit single-slot "
             "Update wire messages without re-sending a full path.")
        .def("locate_bs",       &LudoControlPlane::locate_bs,
             py::arg("key"))
        .def("remove",          &LudoControlPlane::remove,
             py::arg("key"))
        .def("remove_bool",     &LudoControlPlane::remove_bool,
             py::arg("key"),
             "Legacy contract: True iff key was present.")
        .def("update",          &LudoControlPlane::update,
             py::arg("key"), py::arg("value"),
             "Legacy contract: raises if key absent. Wraps changeValue.")
        .def("change_value",    &LudoControlPlane::change_value,
             py::arg("key"), py::arg("value"),
             "Returns {'status','bid','sid','bs'} for the (single) updated slot.")
        .def("resize_capacity", &LudoControlPlane::resize_capacity,
             py::arg("target_capacity"),
             py::arg("force_build") = false)
        .def("merge_dict",      &LudoControlPlane::merge_dict,
             py::arg("d"),
             "Merge dict[key→value] into the CP; value==0 entries are deleted.")
        .def("serialize_dp",    &LudoControlPlane::serialize_dp,
             "Snapshot bytes that LudoDataPlane.deserialize() will accept.")
        .def("size",            &LudoControlPlane::size)
        .def("num_buckets",     &LudoControlPlane::num_buckets)
        .def("capacity",        &LudoControlPlane::capacity);

    py::class_<LudoDataPlane>(m, "LudoDataPlane", py::module_local())
        .def(py::init<>())
        .def("lookup",        &LudoDataPlane::lookup,
             py::arg("key"))
        .def("apply_insert",  &LudoDataPlane::apply_insert,
             py::arg("path"), py::arg("value"))
        .def("apply_update",  &LudoDataPlane::apply_update,
             py::arg("bs"), py::arg("value"),
             "bs = (bid<<2)|sid. Use value=0 to remove (slot value clears).")
        .def("deserialize",   &LudoDataPlane::deserialize,
             py::arg("data"))
        .def("num_buckets",   &LudoDataPlane::num_buckets);
}
