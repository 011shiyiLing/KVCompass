/**
 * smash_runtime_stubs.cpp — minimum link surface for deps/smash headers when
 * compiled into the pybind11 module without deps/smash/common.cpp.
 *
 * deps/smash/common.cpp pulls in boost::stacktrace, gperftools, signal
 * handlers and a `dist/` mkdir on init — none of which are appropriate for
 * a Python extension. We instead provide just the symbols that the header
 * chain (Ludo / Othello / CuckooPresized / utils) actually references at
 * link time under -O3 -DNDEBUG (with PROFILE undefined).
 *
 * Currently this is just `human(uint64_t)`, called unconditionally from
 * Othello's `tryBuild()` log line. If future header upgrades reference more
 * symbols (Clocker statics, TeeOstream globals, InputBase fields), add the
 * minimum stub here rather than linking the boost-heavy common.cpp.
 */
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdint>

std::string human(uint64_t word) {
    std::ostringstream ss;
    if (word < 1024)               { ss << word; }
    else if (word < 10240)         { ss << std::setprecision(2) << word * 1.0 / 1024 << "K"; }
    else if (word < 1048576)       { ss << word / 1024 << "K"; }
    else if (word < 10485760)      { ss << word * 1.0 / 1048576 << "M"; }
    else if (word < (1048576 << 10)) { ss << word / 1048576 << "M"; }
    else                            { ss << word * 1.0 / (1U << 30) << "G"; }
    return ss.str();
}
