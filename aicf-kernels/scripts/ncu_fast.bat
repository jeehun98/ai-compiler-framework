@echo off
setlocal
if not exist out\ncu mkdir out\ncu
REM Fast iteration profile: smaller set, faster collection.
ncu --set speedOfLight --target-processes all -o out\ncu\report_fast %*
endlocal
