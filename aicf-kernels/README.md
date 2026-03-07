🚀 AICF Kernel Development & Profiling Sandbox
본 저장소는 **AICF(AI Compiler Framework)**의 CUDA 커널(GEMM, Add 등)을 독립된 환경에서 개발, 벤치마킹 및 정밀 분석하기 위한 샌드박스입니다. bench.bat과 ncu_kernel_extract.bat을 통해 빌드-실행-분석 과정이 완전히 자동화되어 있습니다.

🛠️ 전체 실행 파이프라인
커널 최적화 루프는 아래와 같은 자동화 단계를 거칩니다.

Kernel Code: src/*.cu 커널 수정

cmake -B build -S . 수행 후 

:: F32 AdamStep 벤치마크
scripts\bench.bat adam_sandbox.exe --n 16777216 --dtype f32 --iters 100

:: F16 AdamStep 벤치마크
scripts\bench.bat adam_sandbox.exe --n 16777216 --dtype f16 --iters 100

각 커널 벤치마크 실행

scripts\bench.bat: CMake 빌드 수행 및 GPU Timer 기반 순수 커널 실행 성능(ms) 측정

scripts\ncu_kernel_extract.bat: Nsight Compute 프로파일링 수행

.ncu-rep 리포트 생성

Metric 추출 및 CSV Export

ncu_extract.ps1을 통한 JSON 변환 (Visual Lab 연동용)

📂 디렉토리 및 파일 구조
Plaintext
aicf-kernels
 ┣ scripts
 ┃ ┣ bench.bat              # 빌드 및 성능 측정 스크립트
 ┃ ┣ ncu_kernel_extract.bat # NCU 프로파일링 및 메트릭 추출
 ┃ ┗ ncu_extract.ps1        # CSV -> JSON 변환기
 ┣ out
 ┃ ┣ ncu                    # Nsight Compute 리포트 (.ncu-rep)
 ┃ ┗ metrics                # 분석 결과물 (.csv, .json)
 ┗ build
   ┗ bin                    # 빌드된 실행 파일 (.exe)
🧠 주요 스크립트 사용법
1️⃣ bench.bat (성능 측정)
CMake 빌드를 먼저 수행한 후, 지정된 인자로 커널을 실행하여 평균 실행 시간(ms)을 출력합니다.

GEMM 실행 예시:

코드 스니펫
scripts\bench.bat gemm_sandbox.exe --m 2048 --n 2048 --k 2048 --dtype f16 --iters 100
출력 결과:

Plaintext
GEMM: M=2048 N=2048 K=2048 dtype=f16 iters=100
max_abs_error: 0.277039
avg_kernel_ms: 4.67536  <-- 실제 커널 런타임 성능
2️⃣ ncu_kernel_extract.bat (정밀 분석 및 메트릭 추출)
특정 커널을 필터링하여 Nsight Compute의 상세 메트릭(SOL, Memory Throughput 등)을 추출합니다.

실행 예시:

코드 스니펫
:: scripts\ncu_kernel_extract.bat <kernel_name> <exe_path> [args...]
scripts\ncu_kernel_extract.bat gemm_f16_wmma_kernel build\bin\gemm_sandbox.exe --dtype f16 --m 2048 --n 2048 --k 2048 --iters 1
추출되는 핵심 메트릭:

sm__throughput.avg.pct_of_peak_sustained_elapsed: 연산 유닛 가동률 (Compute SOL)

dram__throughput.avg.pct_of_peak_sustained_elapsed: 메모리 대역폭 점유율 (Memory SOL)

gpu__time_duration.sum: 커널 실행 시간 상세

3️⃣ nsys.bat (타임라인 분석 - 선택 사항)
커널 간의 실행 순서, CPU-GPU 동기화 오버헤드를 확인하고자 할 때 사용합니다.

실행:

코드 스니펫
scripts\nsys.bat build\bin\gemm_sandbox.exe --dtype f16 --m 2048 --n 2048 --k 2048
🚀 권장 개발 루프 (Development Cycle)
성능 개선 시 다음 루프를 반복합니다.

코드 수정: src/gemm_sandbox.cu 등에서 Tiling 크기나 정렬 로직 수정

벤치마크: scripts\bench.bat으로 avg_kernel_ms 변화 확인

프로파일링: scripts\ncu_kernel_extract.bat으로 병목 지점(Memory vs Compute) 확인

결과 확인: out/metrics/*.json을 시각화 도구로 로드하여 하드웨어 자원 점유율 분석

🎯 이 구조의 장점
자동화: 빌드 명령어를 따로 입력할 필요 없이 bench.bat 하나로 컴파일과 실행이 완료됩니다.

정량화: max_abs_error를 통해 연산의 정확성을 보장하면서 avg_kernel_ms로 성능을 즉각 확인할 수 있습니다.

시각화 준비: 추출된 JSON 데이터는 후속 시각화 툴에서 즉시 사용 가능한 표준 포맷을 제공합니다.