# tools/trace_imports.py
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def is_aicf_mod(name: str) -> bool:
    return name == "aicf_fw" or name.startswith("aicf_fw.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True, help="run target script path, e.g. examples/python/python_framework_test/v2_kid_trace_by_id_test.py")
    ap.add_argument("--repo-root", default=".", help="repo root path")
    ap.add_argument("--out", default="artifacts/import_trace.json")
    args, unknown = ap.parse_known_args()

    repo_root = Path(args.repo_root).resolve()
    entry = Path(args.entry).resolve()

    # Ensure repo root is importable
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    imported: dict[str, str] = {}
    orig_import = __import__

    def traced_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = orig_import(name, globals, locals, fromlist, level)

        # Record module + submodules that became available
        for mname, m in list(sys.modules.items()):
            if not mname or not is_aicf_mod(mname):
                continue
            f = getattr(m, "__file__", None)
            if not f:
                continue
            try:
                fpath = str(Path(f).resolve())
            except Exception:
                continue
            # keep only paths under repo
            if fpath.startswith(str(repo_root)):
                imported[mname] = fpath

        return mod

    builtins = __import__.__module__
    # monkeypatch via builtins module
    import builtins as bi
    bi.__import__ = traced_import

    # Forward any extra args to the entry script (sys.argv)
    sys.argv = [str(entry)] + unknown

    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        bi.__import__ = orig_import

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to relative paths for readability
    rel = {k: str(Path(v).resolve().relative_to(repo_root)) for k, v in sorted(imported.items())}

    out_path.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote: {out_path}")
    print(f"[OK] aicf modules imported: {len(rel)}")


if __name__ == "__main__":
    main()
