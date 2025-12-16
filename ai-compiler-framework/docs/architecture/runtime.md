# Runtime Architecture

## 이 문서의 목적

이 문서는 **AI Compiler Framework의 runtime 계층이 무엇을 책임지고, 왜 별도의 레이어로 존재하는지** 설명한다.
일반적인 ML 프레임워크의 “실행기”가 아니라, **컴파일러 관점의 runtime**라는 점이 핵심이다.

---

## Runtime의 위치

전체 흐름에서 runtime은 다음 위치에 있다.

```
Frontend / Model
        ↓
        IR (중간 표현)
        ↓
Runtime (Graph / Plan / Exec)
        ↓
Backend (CUDA / Kernel)
```

> Runtime은 **IR과 Backend 사이의 계약(contract) 레이어**다.

---

## Runtime의 핵심 책임

Runtime는 다음만을 책임진다.

* 연산 실행 단위(Graph)의 정의
* 실행 전략의 고정(CapturePlan)
* 실행 핸들(GraphExec)의 수명 관리
* 결정적(deterministic) 실행 보장

Runtime가 **하지 않는 것**:

* 커널 구현 ❌
* 연산 미세 최적화 ❌
* 하드웨어별 분기 ❌

---

## 디렉토리 구조

```text
src/runtime/
├─ graph.cpp
├─ capture_plan.cpp
└─ graph_exec.cpp
```

Public API는 다음에 정의된다.

```text
include/aicf/runtime/
├─ graph.hpp
└─ stream.hpp
```

---

## 1. Graph — 실행의 논리적 단위

### 개념

Graph는 **"무엇을 실행할 것인가"** 를 표현하는 단위다.

* 연산들의 집합
* 실행 순서
* 입력/출력 경계

Graph 자체는:

* 커널을 모른다
* CUDA를 모른다
* 실행 방법을 모른다

> Graph는 **순수한 실행 의미론**이다.

---

### Graph의 역할

* IR lowering의 결과물
* CapturePlan의 입력
* GraphExec의 원형

즉:

```
IR → Graph → (Capture) → GraphExec
```

---

## 2. CapturePlan — 실행 전략의 고정

### 왜 CapturePlan이 필요한가

CUDA Graph는 **"한 번 캡처된 실행 흐름"** 을 반복 재생하는 모델이다.
따라서 실행 전에 다음이 모두 고정되어야 한다.

* 커널 종류
* launch 순서
* stream 관계
* 메모리 사용 패턴

이 고정된 전략을 표현하는 것이 CapturePlan이다.

---

### CapturePlan의 책임

* Graph를 CUDA Graph로 변환하는 전략 보유
* capture-safe 여부 검증
* backend 호출 순서 결정

중요한 점:

> CapturePlan은 **최적화 결과물**이지, 실행 객체가 아니다.

---

## 3. GraphExec — 실행 핸들

### 개념

GraphExec는 **실제로 실행 가능한 객체**다.

* CUDA Graph 인스턴스 보유
* 재사용 가능
* 빠른 replay 지원

GraphExec는 다음을 보장한다.

* 실행 경로 불변
* 입력만 바꿔도 동일한 실행 흐름
* 런타임 오버헤드 최소화

---

### GraphExec의 수명

```text
Graph
  ↓ build
CapturePlan
  ↓ capture
GraphExec
  ↓ replay (N times)
```

> GraphExec는 "한 번 만들면 많이 쓴다"는 전제를 가진다.

---

## 4. Stream — 실행 컨텍스트

### 역할

Stream은 실행 컨텍스트를 추상화한다.

* CUDA stream 래핑
* capture-safe 정책 유지
* runtime ↔ backend 연결 고리

중요한 설계 포인트:

* public header에서는 `cudaStream_t`를 직접 노출하지 않음
* backend에서만 실제 CUDA 타입을 다룸

---

## Runtime 설계의 핵심 철학

### 1️⃣ 실행은 반드시 결정적이어야 한다

* 동일한 GraphExec → 동일한 실행
* 분기/동적 로직은 runtime 이전에 제거

이는 CUDA Graph의 전제 조건이기도 하다.

---

### 2️⃣ 최적화는 "한 번"만 한다

* CapturePlan 생성 시 비용 지불
* GraphExec 실행은 최소 오버헤드

즉:

> 느린 build, 빠른 run

---

### 3️⃣ Runtime은 컴파일러의 일부다

일반적인 런타임과 다르다.

* IR과 강하게 결합
* Backend와 계약 관계

Runtime 없이는 이 프로젝트는 **컴파일러가 아니라 커널 모음**에 불과하다.

---

## 지금 단계에서 구현된 것

* Graph / CapturePlan / GraphExec 골격
* runtime init / shutdown
* smoke 실행 확인

아직 없는 것:

* 실제 CUDA Graph capture
* 메모리 플래닝
* 동적 shape 대응

---

## 다음 단계와의 연결

1. IR → Graph lowering 구체화
2. CapturePlan 내부 구조 정의
3. Backend ops와 Graph 연결
4. CUDA Graph capture / replay 구현

---

## 요약

Runtime 계층은:

* 실행 순서를 **고정**하고
* 실행 비용을 **전처리**하며
* GPU 실행을 **결정적으로 만든다**

이 레이어를 이해하면,
이 프로젝트가 왜 단순한 CUDA 라이브러리가 아닌지 명확해진다.
