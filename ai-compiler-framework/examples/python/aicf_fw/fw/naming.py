from __future__ import annotations

def param_name(module_prefix: str, local: str) -> str:
    # e.g. module_prefix="0", local="W" -> "0.W"
    if module_prefix == "":
        return local
    return f"{module_prefix}.{local}"

def opt_m_name(pname: str) -> str:
    return f"opt.m.{pname}"

def opt_v_name(pname: str) -> str:
    return f"opt.v.{pname}"

BC1_NAME = "opt.bc1_inv"
BC2_NAME = "opt.bc2_inv"
