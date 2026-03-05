@echo off
setlocal
if not exist out\nsys mkdir out\nsys
nsys profile -o out\nsys\report %*
endlocal
