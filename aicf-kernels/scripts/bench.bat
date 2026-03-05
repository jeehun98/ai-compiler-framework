@echo off
setlocal EnableExtensions

REM Usage:
REM   scripts\bench.bat [exe_name_or_path] [exe args...]
REM Examples:
REM   scripts\bench.bat add_sandbox.exe --n 16777216
REM   scripts\bench.bat build\bin\add_sandbox.exe --n 16777216

set "BUILD_DIR=build"
set "RAW_EXE=%~1"

if "%RAW_EXE%"=="" (
    set "EXE_NAME=add_sandbox.exe"
) else (
    :: 파일명만 추출 (경로가 포함되어 들어와도 파일명만 남김)
    set "EXE_NAME=%~nx1"
    shift
)

echo [bench] build: %BUILD_DIR%
cmake --build %BUILD_DIR% -j
if errorlevel 1 exit /b 1

:: 최종 실행 경로 설정
set "EXE_PATH=%BUILD_DIR%\bin\%EXE_NAME%"

if not exist "%EXE_PATH%" (
    echo [error] Cannot find executable at: %EXE_PATH%
    exit /b 1
)

echo [bench] run: "%EXE_PATH%" %*
"%EXE_PATH%" %*

exit /b %ERRORLEVEL%