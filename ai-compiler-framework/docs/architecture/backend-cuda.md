# CUDA Backend Architecture

## 이 문서의 목적

이 문서는 **AI Compiler Framework의 CUDA backend가 어떤 철학과 구조로 설계되었는지** 설명한다.
일반적인 “CUDA 코드 모음”이 아니라, **컴파일러 관점의 backend**라는 점을 명확히 한다.

---

## CUDA Backend의 역할

CUDA backend는 다음 책임만 가진다.

* IR이나 Runtime이 **요청한 연산을 GPU에서 실행**
* 하드웨어(Ampere, Tensor Core, SM 구조)에 맞게 **최적화된 커널 제공**
* CUDA Graph capture / replay에 **안전한 실행 경로 유지**

반대로, CUDA backend가 **하지 않는 것**:

* 모델 구조 해석 ❌
* 연산 그래프 최적화 ❌
* 실행 순서 결정 ❌

> CUDA backend는 “결정된 계획을 정확히 실행하는 계층”이다.

---

## 디렉토리 구조

```text
src/backends/cuda/
├─ cuda_context.cu
├─ nvtx.cu
└─ ops/
   ├─ _common/
   │  ├─ shim/
   │  └─ utils/
   └─ <op_name>/
      ├─ api.hpp
      ├─ launcher.cu
      └─ kernels.cuh
```

이 구조는 **의도적으로 3-layer 패턴**을 강제한다.

---

## 1. cuda_context.cu — CUDA 실행 환경

### 역할

* CUDA runtime과의 접점
* stream / device / context 관리
* 향후 CUDA Graph capture 진입점

### 설계 원칙

* public header(`include/`)에서 `cuda_runtime.h`를 직접 include하지 않는다
* CUDA 의존성은 backend 내부로 격리

### 의미

이 파일은 단순 유틸이 아니라,

> “AI compiler가 CUDA와 대화하는 문”이다.

---

## 2. nvtx.cu — 관측 가능성 레이어

### 역할

* NVTX marker / range 삽입
* profiling을 위한 hook 제공

### 중요한 점

* **옵션화**되어 있음 (`AICF_ENABLE_NVTX`)
* CUDA Graph capture 시에도 안전하도록 설계 예정

> NVTX는 디버깅 도구가 아니라
> **컴파일러가 성능을 이해하기 위한 센서**다.

---

## 3. ops/ — 연산 구현의 핵심

### 전체 구조

```text
ops/<op_name>/
├─ api.hpp
├─ launcher.cu
└─ kernels.cuh
```

이 분리는 **절대 합치지 않는다**.

---

### 3.1 api.hpp — 외부 인터페이스

#### 역할

* Runtime / GraphExec가 호출하는 **유일한 진입점**
* shape, dtype, layout 검증
* CUDA 세부사항 은닉

#### 특징

* C++ API
* 템플릿 최소화
* 실행 정책은 노출하지 않음

> api.hpp는 “이 op가 무엇을 하는지”만 말한다.

---

### 3.2 launcher.cu — 실행 전략 결정

#### 역할

* grid / block / warp 구성
* kernel 선택 (naive / tiled / tensor core)
* shared memory 크기 계산

#### 여기서 하는 판단들

* tile size
* warp specialization 여부
* smem 사용량
* cp.async 사용 여부

> launcher는 **컴파일러의 연장선**이다.
> 단순 dispatch 코드가 아니다.

---

### 3.3 kernels.cuh — 순수 device code

#### 역할

* 실제 연산 수행
* warp-level / smem-level 최적화
* 하드웨어 특화 구현

#### 특징

* header-only
* heavily templated
* host 코드 의존 ❌

> kernel은 “연산 알고리즘 그 자체”다.

---

## 4. _common/ — 공통 인프라

### _common/shim/

```text
shim/
├─ types.hpp
├─ enums.hpp
├─ status.hpp
└─ launch.hpp
```

#### 목적

* op 간 인터페이스 통일
* enum / dtype / layout 표준화
* 캡처-safe한 launch 방식 유지

이 레이어가 없으면:

* op마다 enum 다름
* runtime 연동 지옥
* graph capture 깨짐

---

### _common/utils/

```text
utils/
├─ math.cuh
└─ reduce.cuh
```

#### 목적

* warp reduce
* numerical helpers
* 커널 간 코드 중복 제거

---

## 이 구조의 핵심 철학

### 1️⃣ 커널 최적화 ≠ 컴파일러 최적화

* kernel: 연산 내부 최적화
* compiler/runtime: 실행 흐름 최적화

CUDA backend는 **kernel 최적화까지만 책임**진다.

---

### 2️⃣ launcher가 핵심이다

* naive kernel은 언제든 교체 가능
* launcher의 결정 구조는 **컴파일러 자산**

---

### 3️⃣ Tensor Core는 backend의 문제

IR은:

* matmul
* reduction
  만 표현한다.

Tensor Core 사용 여부는 **100% backend 책임**이다.

---

## 지금 단계에서 구현된 것

* CUDA backend 골격
* NVTX stub
* CUDA object 포함 빌드 확인

아직 없는 것:

* 실제 op
* CUDA Graph capture
* Tensor Core kernel

---

## 다음 단계에서 이 문서와 연결되는 작업

1. ops/gemm 실제 구현
2. launcher에서 tile / warp policy 분기
3. runtime → backend 호출 연결
4. CUDA Graph capture-safe launch 규칙 정의

---

## 요약

이 CUDA backend는:

* 단순 CUDA wrapper ❌
* 연산 라이브러리 ❌
* **AI 컴파일러의 하드웨어 실행 계층**이다 ✔

이 구조를 이해하면, 이후
