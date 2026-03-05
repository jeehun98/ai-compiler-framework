@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: 사용법 체크
if "%~1"=="" goto usage
if "%~2"=="" goto usage

set "KERNEL=%~1"
set "EXE=%~2"
shift
shift

:: --- 환경 설정 (필요 시 외부에서 set으로 변경 가능) ---
if "%NCU_SET%"=="" set "NCU_SET=basic"
if "%NCU_LAUNCH_COUNT%"=="" set "NCU_LAUNCH_COUNT=1"
if "%NCU_LAUNCH_SKIP%"=="" set "NCU_LAUNCH_SKIP=0"
if "%NCU_FORCE%"=="" set "NCU_FORCE=1"
if "%NCU_OUTDIR%"=="" set "NCU_OUTDIR=out\ncu"
if "%MET_OUTDIR%"=="" set "MET_OUTDIR=out\metrics"
if "%NCU_TAG%"=="" set "NCU_TAG=%KERNEL%"

:: 핵심 분석 메트릭 (Roofline 및 Resource 점유율 위주)
if "%NCU_METRICS%"=="" (
  set "NCU_METRICS=gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,l2tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"
)

:: 폴더 생성
if not exist "%NCU_OUTDIR%" mkdir "%NCU_OUTDIR%"
if not exist "%MET_OUTDIR%" mkdir "%MET_OUTDIR%"

:: 타임스탬프 생성 (yyyyMMdd_HHmmss_fff)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "TS=%%i"

set "REP_BASE=%NCU_OUTDIR%\%NCU_TAG%_%TS%"
set "CSV_OUT=%MET_OUTDIR%\%NCU_TAG%_%TS%.csv"
set "JSON_OUT=%MET_OUTDIR%\%NCU_TAG%_%TS%.json"

set "FORCE_FLAG="
if "%NCU_FORCE%"=="1" set "FORCE_FLAG=-f"

echo [1/3] Profiling kernel: %KERNEL% ...
:: NCU 실행 (Python 환경 변수 초기화로 충돌 방지)
cmd /c "set PYTHONHOME=& set PYTHONPATH=& set PYTHONNOUSERSITE=1& ncu %FORCE_FLAG% --set %NCU_SET% --target-processes all --kernel-name ""%KERNEL%"" --launch-skip %NCU_LAUNCH_SKIP% --launch-count %NCU_LAUNCH_COUNT% -o ""%REP_BASE%"" ""%EXE%"" %*"
if errorlevel 1 exit /b 1

echo [2/3] Exporting to CSV (Metrics: %NCU_METRICS%) ...
:: CSV 추출
cmd /c "set PYTHONHOME=& set PYTHONPATH=& set PYTHONNOUSERSITE=1& ncu -i ""%REP_BASE%.ncu-rep"" --page raw --csv --metrics ""%NCU_METRICS%"" --log-file ""%CSV_OUT%"""
if errorlevel 1 exit /b 1

echo [3/3] Extracting to JSON for Visual Lab ...
:: PowerShell을 이용한 JSON 변환
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\ncu_extract.ps1" -Csv "%CSV_OUT%" -Json "%JSON_OUT%"
if errorlevel 1 exit /b 1

echo.
echo [SUCCESS] Pipeline Finished.
echo   - Rep:  %REP_BASE%.ncu-rep
echo   - Csv:  %CSV_OUT%
echo   - Json: %JSON_OUT%
goto :eof

:usage
echo Usage: %~nx0 ^<kernel_name^> ^<exe_path^> [exe args...]
echo Example: %~nx0 vector_add build\bin\vector_add.exe --n 1048576
exit /b 2