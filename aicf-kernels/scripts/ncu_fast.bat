@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM ncu_fast.bat
REM - 초고속 반복용 Nsight Compute 실행 스크립트
REM - 기본값: launchStats + kernel 1회(launch-count 1)만 수집
REM
REM 사용 예)
REM   scripts\ncu_fast.bat build\bin\vector_add.exe --n 16777216 --iters 10
REM
REM 옵션(환경변수로 오버라이드)
REM   set NCU_SET=basic          (default: launchStats)
REM   set NCU_KERNEL=vector_add  (default: 비움 = 전체)
REM   set NCU_LAUNCH_COUNT=1     (default: 1)
REM   set NCU_LAUNCH_SKIP=0      (default: 0)
REM   set NCU_OUT=out\ncu\report_fast
REM ============================================================

if not exist out\ncu mkdir out\ncu

if "%NCU_SET%"=="" set "NCU_SET=launchStats"
if "%NCU_LAUNCH_COUNT%"=="" set "NCU_LAUNCH_COUNT=1"
if "%NCU_LAUNCH_SKIP%"=="" set "NCU_LAUNCH_SKIP=0"
if "%NCU_OUT%"=="" set "NCU_OUT=out\ncu\report_fast"

REM 첫 인자는 프로파일링할 실행파일이어야 함
if "%~1"=="" (
  echo Usage: %~nx0 ^<exe^> [exe args...]
  exit /b 2
)

set "EXE=%~1"
shift

set "KERNEL_ARGS="
if not "%NCU_KERNEL%"=="" set "KERNEL_ARGS=--kernel-name %NCU_KERNEL%"

echo [ncu_fast] set=%NCU_SET% kernel=%NCU_KERNEL% launch_skip=%NCU_LAUNCH_SKIP% launch_count=%NCU_LAUNCH_COUNT%
echo [ncu_fast] out=%NCU_OUT%.ncu-rep
echo [ncu_fast] exe=%EXE% %*

REM --target-processes all : 자식 프로세스/런타임 런치 케이스에서도 안정적
REM --launch-skip / --launch-count : 워밍업 이후 1회만 찍기
ncu --set %NCU_SET% --target-processes all ^
    %KERNEL_ARGS% ^
    --launch-skip %NCU_LAUNCH_SKIP% --launch-count %NCU_LAUNCH_COUNT% ^
    -o "%NCU_OUT%" ^
    "%EXE%" %*

endlocal