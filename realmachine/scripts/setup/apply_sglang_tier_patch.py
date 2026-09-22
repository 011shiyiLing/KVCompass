#!/usr/bin/env python3
"""apply_sglang_tier_patch.py — wire the four notify_* hooks into SGLang.

Patches SGLang 0.5.10.post1's mem_cache layer so the discovery plane
sees real KV state-machine transitions:

    cache_finished_req (radix_cache.py)  → notify_advertise   (VRAM admission)
    write_backup       (hiradix_cache.py) → notify_demote     (VRAM → DRAM)
    load_back          (hiradix_cache.py) → notify_promote    (DRAM → VRAM)
    evict_host         (hiradix_cache.py) → notify_evict      (DRAM → gone)

All insertions are bracketed by sentinel comments
``# >>>LUDO_TIER_HOOK_START id=<id>`` ... ``# <<<LUDO_TIER_HOOK_END id=<id>``
so the patch is idempotent: re-running detects each block and skips it;
``--revert`` strips every block by sentinel.

Usage
    python3 scripts/setup/apply_sglang_tier_patch.py            # apply
    python3 scripts/setup/apply_sglang_tier_patch.py --revert   # undo
    python3 scripts/setup/apply_sglang_tier_patch.py --check    # report status only
"""
from __future__ import annotations
import argparse, importlib, importlib.util, re, sys
from pathlib import Path


def find_sglang_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        sys.exit("ERROR: 'import sglang' failed. Install sglang first.")
    return Path(spec.origin).parent

START = "# >>>LUDO_TIER_HOOK_START"
END   = "# <<<LUDO_TIER_HOOK_END"

IMPORT_BLOCK = """\
# >>>LUDO_TIER_HOOK_START id=imports
try:
    from ludo_sglang.worker.tier_notifier import (
        notify_advertise as _ludo_notify_advertise,
        notify_demote    as _ludo_notify_demote,
        notify_promote   as _ludo_notify_promote,
        notify_evict     as _ludo_notify_evict,
    )
except Exception:  # ludo_sglang not on PYTHONPATH → silently disable hooks
    def _ludo_notify_advertise(*a, **kw): pass
    def _ludo_notify_demote(*a, **kw):    pass
    def _ludo_notify_promote(*a, **kw):   pass
    def _ludo_notify_evict(*a, **kw):     pass
# <<<LUDO_TIER_HOOK_END id=imports
"""

HELPER_BLOCK = """\
# >>>LUDO_TIER_HOOK_START id=helper
def _ludo_full_token_ids(node):
    \"\"\"Walk from root to ``node`` and concatenate every ancestor's
    ``key.token_ids`` so the tier_notifier can derive the per-level
    Ludo keys via ``chunk_to_key(token_ids[:cs])`` for every
    ``cs in LUDO_CHUNK_SIZES``.\"\"\"
    parts = []
    cur = node
    while cur is not None and getattr(cur, "parent", None) is not None:
        k = getattr(cur, "key", None)
        if k is not None and getattr(k, "token_ids", None):
            parts.append(list(k.token_ids))
        cur = cur.parent
    out = []
    for p in reversed(parts):
        out.extend(p)
    return out
# <<<LUDO_TIER_HOOK_END id=helper
"""


def has_sentinel(text: str, sid: str) -> bool:
    return f"{START} id={sid}" in text and f"{END} id={sid}" in text


def insert_after(text: str, anchor_re: str, insertion: str) -> str:
    """Insert *insertion* immediately after the line that matches anchor_re.
    The matched line's indentation is detected from the line's leading
    whitespace and applied to every non-blank line of *insertion*."""
    m = re.search(anchor_re, text, flags=re.MULTILINE)
    if not m:
        raise SystemExit(f"ERROR: anchor not found: {anchor_re!r}")
    eol = text.find("\n", m.end())
    if eol < 0:
        eol = len(text)
    line_start = text.rfind("\n", 0, m.start()) + 1
    # Indent = the leading whitespace of the matched line itself, NOT the
    # span [line_start, m.start()] (which is empty when the regex starts
    # with ^\s*).
    line = text[line_start:eol]
    indent = re.match(r"[ \t]*", line).group(0)
    indented = "\n".join((indent + ln if ln.strip() else ln)
                         for ln in insertion.splitlines())
    return text[:eol + 1] + indented + "\n" + text[eol + 1:]


def strip_sentinel(text: str, sid: str) -> str:
    pattern = re.compile(
        rf"^[ \t]*{re.escape(START)} id={re.escape(sid)}\n.*?^[ \t]*{re.escape(END)} id={re.escape(sid)}\n",
        flags=re.DOTALL | re.MULTILINE,
    )
    return pattern.sub("", text)


# ── Concrete edits ─────────────────────────────────────────────────────────
def patch_radix_cache(text: str) -> str:
    # 1) imports
    if not has_sentinel(text, "imports"):
        # Insert after the last `import` near the top of the file.
        last_import = max(
            (m.start() for m in re.finditer(r"(?m)^(?:from |import )\S+", text[:4000])),
            default=-1,
        )
        if last_import < 0:
            raise SystemExit("ERROR: could not find import region in radix_cache.py")
        eol = text.find("\n", last_import)
        text = text[:eol + 1] + "\n" + IMPORT_BLOCK + "\n" + text[eol + 1:]

    # 2) notify_advertise in cache_finished_req
    if not has_sentinel(text, "advertise"):
        # Insert just before `# free the unaligned tail` so we already have
        # `token_ids` in scope (computed at line 477) and the request is
        # logically "admitted into the radix tree".
        anchor = r"^[ \t]*# free the unaligned tail"
        text = insert_after(
            text, anchor,
            "# >>>LUDO_TIER_HOOK_START id=advertise\n"
            "if is_insert and token_ids:\n"
            "    _ludo_notify_advertise(token_ids=list(token_ids))\n"
            "# <<<LUDO_TIER_HOOK_END id=advertise"
        )
    return text


def patch_hiradix_cache(text: str) -> str:
    # 1) imports + helper
    if not has_sentinel(text, "imports"):
        last_import = max(
            (m.start() for m in re.finditer(r"(?m)^(?:from |import )\S+", text[:4000])),
            default=-1,
        )
        if last_import < 0:
            raise SystemExit("ERROR: could not find import region in hiradix_cache.py")
        eol = text.find("\n", last_import)
        text = text[:eol + 1] + "\n" + IMPORT_BLOCK + "\n" + text[eol + 1:]

    if not has_sentinel(text, "helper"):
        # Place the helper right after the imports block so all class methods
        # below see it at module scope.
        idx = text.find("# <<<LUDO_TIER_HOOK_END id=imports")
        eol = text.find("\n", idx) + 1
        text = text[:eol] + "\n" + HELPER_BLOCK + "\n" + text[eol:]

    # 2) write_backup → notify_demote
    if not has_sentinel(text, "demote"):
        # Anchor: inside `def write_backup(...)`, after `node.host_value = host_indices.clone()`
        anchor = r"^[ \t]*node\.host_value = host_indices\.clone\(\)\s*$"
        text = insert_after(
            text, anchor,
            "# >>>LUDO_TIER_HOOK_START id=demote\n"
            "_t = _ludo_full_token_ids(node)\n"
            "if _t:\n"
            "    _ludo_notify_demote(token_ids=_t)\n"
            "# <<<LUDO_TIER_HOOK_END id=demote"
        )

    # 3) load_back → notify_promote
    if not has_sentinel(text, "promote"):
        # Anchor: the loop body where node.value gets assigned the loaded device indices.
        anchor = r"^[ \t]*node\.value = device_indices\[offset : offset \+ len\(node\.host_value\)\]\.clone\(\)\s*$"
        text = insert_after(
            text, anchor,
            "# >>>LUDO_TIER_HOOK_START id=promote\n"
            "_t = _ludo_full_token_ids(node)\n"
            "if _t:\n"
            "    _ludo_notify_promote(token_ids=_t)\n"
            "# <<<LUDO_TIER_HOOK_END id=promote"
        )

    # 4) evict_host → notify_evict
    if not has_sentinel(text, "evict"):
        # Anchor: right after `self._record_remove_event(x)` inside evict_host.
        anchor = r"^[ \t]*self\._record_remove_event\(x\)\s*$"
        text = insert_after(
            text, anchor,
            "# >>>LUDO_TIER_HOOK_START id=evict\n"
            "_t = _ludo_full_token_ids(x)\n"
            "if _t:\n"
            "    _ludo_notify_evict(token_ids=_t)\n"
            "# <<<LUDO_TIER_HOOK_END id=evict"
        )
    return text


def revert(text: str, ids) -> str:
    for sid in ids:
        text = strip_sentinel(text, sid)
    return text


# ── Driver ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--revert", action="store_true",
                   help="strip every LUDO_TIER_HOOK_* block and exit")
    g.add_argument("--check", action="store_true",
                   help="report patch status without modifying anything")
    args = ap.parse_args()

    sg_root  = find_sglang_root()
    rc_path  = sg_root / "srt/mem_cache/radix_cache.py"
    hrc_path = sg_root / "srt/mem_cache/hiradix_cache.py"
    if not rc_path.exists():  sys.exit(f"missing {rc_path}")
    if not hrc_path.exists(): sys.exit(f"missing {hrc_path}")

    rc_text  = rc_path.read_text()
    hrc_text = hrc_path.read_text()

    if args.check:
        print(f"sglang at      : {sg_root}")
        print(f"radix_cache.py : "
              f"imports={'yes' if has_sentinel(rc_text,'imports') else 'no '}  "
              f"advertise={'yes' if has_sentinel(rc_text,'advertise') else 'no '}")
        print(f"hiradix_cache.py: "
              f"imports={'yes' if has_sentinel(hrc_text,'imports') else 'no '}  "
              f"helper={'yes' if has_sentinel(hrc_text,'helper') else 'no '}  "
              f"demote={'yes' if has_sentinel(hrc_text,'demote') else 'no '}  "
              f"promote={'yes' if has_sentinel(hrc_text,'promote') else 'no '}  "
              f"evict={'yes' if has_sentinel(hrc_text,'evict') else 'no '}")
        return

    if args.revert:
        new_rc  = revert(rc_text,  ["advertise", "imports"])
        new_hrc = revert(hrc_text, ["evict", "promote", "demote", "helper", "imports"])
        if new_rc  != rc_text:  rc_path.write_text(new_rc);  print(f"  reverted {rc_path}")
        if new_hrc != hrc_text: hrc_path.write_text(new_hrc); print(f"  reverted {hrc_path}")
        print("done.")
        return

    new_rc  = patch_radix_cache(rc_text)
    new_hrc = patch_hiradix_cache(hrc_text)

    changed = False
    if new_rc != rc_text:
        rc_path.write_text(new_rc);  print(f"  patched {rc_path}");  changed = True
    if new_hrc != hrc_text:
        hrc_path.write_text(new_hrc); print(f"  patched {hrc_path}"); changed = True
    if not changed:
        print("  (already patched — nothing to do)")
    print("done.")


if __name__ == "__main__":
    main()
