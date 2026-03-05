#!/usr/bin/env bash
mkdir -p out/ncu
ncu --set full --target-processes all -o out/ncu/report_full "$@"
