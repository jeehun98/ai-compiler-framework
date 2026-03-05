🔧 전체 실행 파이프라인

커널 분석 흐름은 다음과 같다.

커널 코드
   │
   ▼
cmake build
   │
   ▼
bench.bat
   │
   ▼
ncu_kernel_extract.bat
   │
   ▼
ncu (Nsight Compute)
   │
   ▼
.ncu-rep 생성
   │
   ▼
rep → csv export
   │
   ▼
ncu_extract.ps1
   │
   ▼
json 생성
📂 생성되는 파일 구조   
aicf-kernels
 ┣ out
 ┃ ┣ ncu
 ┃ ┃ └ add_f16x2_kernel_<timestamp>.ncu-rep
 ┃ ┃
 ┃ ┗ metrics
 ┃    ┣ add_f16x2_kernel_<timestamp>.csv
 ┃    ┗ add_f16x2_kernel_<timestamp>.json
🧠 각 스크립트 역할
1️⃣ bench.bat

GPU timer 기반 순수 커널 실행 성능 측정

실행:

scripts\bench.bat add_sandbox.exe --dtype f16 --n 16777216 --iters 200

출력:

avg_kernel_ms: 0.24

이 값이 실제 runtime 성능이다.

2️⃣ ncu_kernel_extract.bat ⭐ 핵심

역할:

Nsight Compute 실행
→ 특정 kernel 필터링
→ .ncu-rep 생성
→ CSV export
→ JSON 생성

실행:

scripts\ncu_kernel_extract.bat add_f16x2_kernel build\bin\add_sandbox.exe --dtype f16 --n 16777216 --iters 10

내부 실행 흐름:

1. ncu profiling
2. .ncu-rep 생성
3. ncu -i export
4. CSV 생성
5. ncu_extract.ps1
6. JSON 생성
3️⃣ ncu_extract.ps1

역할

CSV → JSON 변환

예

CSV

gpu__time_duration.sum = 368.992 us

JSON

{
  "kernel": "add_f16x2_kernel",
  "metrics": {
    "gpu__time_duration.sum": {
      "avg": 368.992,
      "unit": "us"
    }
  }
}
4️⃣ nsys.bat (선택)

Nsight Systems trace

사용 시점

kernel timeline 분석
CPU-GPU sync
CUDA API trace

실행

scripts\nsys.bat build\bin\add_sandbox.exe --dtype f16 --n 16777216

결과

out/nsys/report_<timestamp>.qdrep
🚀 실제 커널 개발 루프

개발자는 보통 이렇게 사용한다.

1️⃣ 빌드
cmake --build build -j
2️⃣ 성능 확인
scripts\bench.bat add_sandbox.exe --dtype f16 --n 16777216
3️⃣ Nsight Compute 분석
scripts\ncu_kernel_extract.bat add_f16x2_kernel build\bin\add_sandbox.exe --dtype f16 --n 16777216
4️⃣ 결과 확인
out/metrics/*.csv
out/metrics/*.json
🎯 이 구조의 장점

이 구조는 CUDA 커널 연구 루프와 정확히 맞는다.

커널 수정
   ↓
build
   ↓
bench
   ↓
ncu profiling
   ↓
metrics export
   ↓
analysis

즉

Kernel Dev → Profiling → Metric Extraction

이 루프가 완전히 자동화된 상태다.