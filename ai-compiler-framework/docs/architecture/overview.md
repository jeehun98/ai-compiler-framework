# AI Compiler Framework — Architecture Overview

## 이 문서의 목적

이 문서는 **AI Compiler Framework 전체 구조를 한 눈에 이해하기 위한 진입 문서**다.
개별 구현보다, "왜 이런 계층 구조를 가지는지"를 설명하는 데 집중한다.

---

## 프로젝트의 정체성

이 프로젝트는:

* 딥러닝 프레임워크 ❌
* 단순 CUDA 커널 모음 ❌
* 연산 라이브러리 ❌

**AI 컴파일러 + 런타임 프레임워크**다 ✔

핵심 목표는 하나다.

> 동적 AI 모델을 **정적이고 결정적인 GPU 실행 그래프**로 변환한다.

---

## 전체 계층 구조

```text
Frontend / Model Definition
        ↓
IR (Intermediate Representation)
        ↓
Runtime (Graph / CapturePlan / GraphExec)
        ↓
Backend (CUDA / Ops / Kernels)
```

각 계층은 **명확히 분리된 책임**을 가진다.

---

## 1. Frontend (범위 외)

이 프로젝트는 Frontend를 직접 구현하지 않는다.

Frontend의 역할:

* 모델 정의
* 제어 흐름(if / loop)
* 사용자 친화 API

이 프로젝트는 Frontend로부터 **IR**을 입력받는 것을 전제로 한다.

---

## 2. IR — 무엇을 할 것인가

IR은 연산을 **추상적으로 표현**한다.

IR이 표현하는 것:

* 연산 종류 (matmul, reduction, elementwise)
* 데이터 흐름
* 논리적 실행 관계

IR이 표현하지 않는 것:

* CUDA grid / block
* Tensor Core
* shared memory

> IR은 "의미"만 담고, "방법"은 담지 않는다.

---

## 3. Runtime — 어떻게 실행할 것인가

Runtime은 **실행 의미를 실제 실행 계획으로 고정**한다.

Runtime의 책임:

* 실행 단위(Graph) 정의
* 실행 전략 고정(CapturePlan)
* 실행 객체(GraphExec) 관리

Runtime의 핵심 키워드:

* deterministic
* capture-safe
* replayable

> Runtime은 컴파일러의 일부다.

---

## 4. Backend — 실제로 실행한다

Backend는 하드웨어에 특화된 실행 계층이다.

Backend의 책임:

* 커널 구현
* 하드웨어 최적화
* 실제 GPU 실행

Backend가 결정하는 것:

* tile 크기
* warp 구성
* Tensor Core 사용 여부

> Backend는 IR의 추상성을 **현실의 하드웨어로 내린다**.

---

## 디렉토리 구조 요약

```text
ai-compiler-framework/
├─ include/          # Public API (IR / Runtime 인터페이스)
├─ src/
│  ├─ ir/            # IR 구현
│  ├─ runtime/       # Graph / CapturePlan / GraphExec
│  └─ backends/
│     └─ cuda/       # CUDA backend
├─ examples/         # 실행 검증
├─ docs/             # 아키텍처 문서
└─ CMakeLists.txt
```

---

## 이 구조가 낯선 이유

대부분의 ML 프로젝트는 다음 흐름에 익숙하다.

```
model → training → inference
```

이 프로젝트는 다르다.

```
IR → compile → capture → replay
```

즉, **실행 이전에 대부분의 비용을 지불**한다.

---

## 설계 철학 요약

### 1️⃣ 실행은 반드시 결정적이어야 한다

* 동일 입력 → 동일 실행 경로
* CUDA Graph 전제 조건

### 2️⃣ 최적화는 한 번만 한다

* build 단계: 느려도 됨
* run 단계: 극도로 빠르게

### 3️⃣ 계층 간 책임은 섞지 않는다

* IR은 의미
* Runtime은 계획
* Backend는 실행

---

## 현재 구현 상태

* 전체 디렉토리 구조 확정
* C++ / CUDA 혼합 빌드 성공
* Runtime smoke test 통과
* Backend 골격 완성

아직 없는 것:

* 실제 연산(op)
* CUDA Graph capture
* IR 최적화 패스

---

## 문서 흐름 가이드

이 문서 이후 읽기 순서:

1. `runtime.md` — 실행의 중심
2. `backend-cuda.md` — CUDA 실행 계층
3. `ir-design.md` — IR 최소 단위 설계 (예정)

---

## 요약

이 프로젝트는:

* AI 모델을
* **컴파일**하여
* GPU에서 **결정적으로 실행**하기 위한

**AI 컴파일러 프레임워크**다.

이 구조를 이해하면, 이후 모든 코드와 문서가 같은 방향을 가리킨다.
