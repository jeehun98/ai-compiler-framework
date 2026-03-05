#!/usr/bin/env bash
mkdir -p out/ncu
ncu --set speedOfLight --target-processes all -o out/ncu/report_fast "$@"
