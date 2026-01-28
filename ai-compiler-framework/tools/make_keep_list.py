# tools/make_keep_list.py
from __future__ import annotations
import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/keep_files.txt")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    files = set()

    for p in args.inputs:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for _, relpath in data.items():
            files.add(relpath.replace("\\", "/"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # keep only .py / .pyd / .so / .dll 등 “실제 파일”
    keep = sorted([f for f in files if Path(f).suffix in (".py", ".pyd", ".so", ".dll")])

    out.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"[OK] wrote keep list: {out}  (n={len(keep)})")

if __name__ == "__main__":
    main()
