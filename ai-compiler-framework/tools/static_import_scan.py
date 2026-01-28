from __future__ import annotations
import ast
import argparse
from pathlib import Path

def scan_file(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return set()
    try:
        tree = ast.parse(src)
    except Exception:
        return set()

    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                mods.add(n.module)
    return mods

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="python/aicf_fw")
    ap.add_argument("--out", default="artifacts/static_imports.txt")
    args = ap.parse_args()

    root = Path(args.root)
    all_mods = set()
    for py in root.rglob("*.py"):
        all_mods |= scan_file(py)

    aicf = sorted([m for m in all_mods if m == "aicf_fw" or m.startswith("aicf_fw.")])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(aicf) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {args.out} (n={len(aicf)})")

if __name__ == "__main__":
    main()
