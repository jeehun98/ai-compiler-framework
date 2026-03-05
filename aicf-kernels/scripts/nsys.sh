#!/usr/bin/env bash
mkdir -p out/nsys
nsys profile -o out/nsys/report "$@"
