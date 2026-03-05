@echo off
setlocal
if not exist out\ncu mkdir out\ncu
REM Heavy profile: full set. Use for final reporting only.
ncu --set full --target-processes all -o out\ncu\report_full %*
endlocal
