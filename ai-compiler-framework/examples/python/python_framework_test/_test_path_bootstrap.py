from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start
    while p.parent != p:
        # repo root 폴더명이 다르면 여기 조건만 바꿔.
        if p.name == "ai-compiler-framework":
            return p
        p = p.parent
    return start.parents[0]


def ensure_test_paths() -> dict[str, Path]:
    """
    Ensures:
      - examples/python is importable (aicf_fw.*)
      - build/python is importable (aicf_cuda.*)
    Returns a dict with resolved paths.
    """
    this = Path(__file__).resolve()
    repo = _find_repo_root(this)

    examples_py = repo / "examples" / "python"
    build_py = repo / "build" / "python"

    for p in (examples_py, build_py):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    return {"repo": repo, "examples_py": examples_py, "build_py": build_py}


def import_cuda_ext():
    """
    Returns aicf_cuda._C (pybind extension).
    """
    ensure_test_paths()
    from aicf_cuda import _C  # type: ignore
    return _C
