"""
Build the ludo_binding pybind11 extension.

Usage (from the src/ludo/ directory):
    pip install pybind11
    python setup.py build_ext --inplace

The .so file will be placed in src/ludo/ so that `from src.ludo import ludo_binding`
works when the project root is on sys.path.
"""

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# ── Paths ──────────────────────────────────────────────────────────────────────
# We compile against the Smash flavour of Ludo (deps/smash/Ludo/ludo_cp_dp.h),
# which provides ControlPlaneLudo + DataPlaneLudo + path-entry insert results
# — the substrate for the L1+L2 maintenance plane (see master_fn.h /
# lookup_fn.h for the protocol shape we're mirroring on the Python side).
HERE = Path(__file__).parent.resolve()                     # src/ludo/
LUDO_ROOT = (HERE / "../../deps/smash").resolve()          # deps/smash/

# ── Source files required to compile the Ludo hash table ──────────────────────
# Header chain (Ludo/, Othello/, CuckooPresized/, utils/{hash,lfsr64,
# disjointset}.h) only references InputBase / TeeOstream / Clocker symbols
# inside `if(full_debug)` blocks. With -O3 -DNDEBUG those are dead-stripped,
# so we don't need common.cpp or input_types.cpp — just the hash sources.
LUDO_SOURCES = [
    str(LUDO_ROOT / "utils" / "farmhash" / "farmhash.cc"),
    str(LUDO_ROOT / "utils" / "hashutil.cc"),
    str(HERE / "smash_runtime_stubs.cpp"),
]

# ── Compiler flags ─────────────────────────────────────────────────────────────
# -DNDEBUG    : disables debug assertions and Counter profiling inside Ludo
# -O3 -ffast-math : maximum optimisation (-Ofast is deprecated in newer clang)
# -march=native : use native CPU instructions
# -mavx / -maes : x86-only; omitted on Apple Silicon (arm64)
import platform as _platform
_is_arm64 = _platform.machine() == "arm64"
EXTRA_COMPILE_ARGS = [
    "-std=c++17",
    "-DNDEBUG",
    "-O3",
    "-ffast-math",
    "-Wno-unused-result",
] + ([] if _is_arm64 else ["-mavx", "-maes"])

# ── Libraries ──────────────────────────────────────────────────────────────────
# hashutil.cc uses OpenSSL for some hash variants.
# On macOS, Homebrew installs OpenSSL to a non-default path; find it explicitly.
LIBRARIES = ["ssl", "crypto", "pthread"]

import subprocess as _sp
LIBRARY_DIRS = []
INCLUDE_DIRS = [str(LUDO_ROOT)]
try:
    _openssl_prefix = _sp.check_output(
        ["brew", "--prefix", "openssl"], text=True
    ).strip()
    LIBRARY_DIRS.append(f"{_openssl_prefix}/lib")
    INCLUDE_DIRS.append(f"{_openssl_prefix}/include")
except Exception:
    pass  # not macOS or brew not available; rely on system paths

ext_modules = [
    Pybind11Extension(
        "ludo_binding",
        sources=[str(HERE / "ludo_binding.cpp")] + LUDO_SOURCES,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        libraries=LIBRARIES,
    ),
    # Same source, V=uint64_t / VL=64 → 32 workers × 2-bit tier.  Fleets
    # (or per-region shards) with 17-32 workers import this module instead;
    # ludo_table.LudoTable picks the backend by worker count.
    Pybind11Extension(
        "ludo_binding64",
        sources=[str(HERE / "ludo_binding.cpp")] + LUDO_SOURCES,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        libraries=LIBRARIES,
        define_macros=[("LUDO_VALUE_BITS", "64")],
    ),
]

setup(
    name="ludo_binding",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
)
