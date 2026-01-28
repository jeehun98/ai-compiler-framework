# AICF FW MVP + core_v2 엔진 아키텍처 노트 (Stage6)

이 문서는 `aicf_fw/fw`(프레임워크 UX)와 `aicf_fw/core_v2`(엔진)를 분리된 레이어로 유지하면서, **compile → warmup → capture → replay → reset**까지의 계약(Contract)과 데이터 흐름을 고정하기 위한 “1페이지” 아키텍처 메모다.
현재 `v2_fw_mvp_train_step_test.py`가 통과하는 상태를 기준으로 작성한다.

---

## 목표와 범위

### 목표

* 사용자 관점: `compile_train_step(model, opt, ...)` 호출 후 **train_step / capture / replay / reset**이 직관적이고 안정적으로 동작.
* 엔진 관점: IR/Plan/CUDA Graph 실행이 **결정론적**이고 **상태 관리**가 명확.

### 현재 범위 (Stage6 / FW MVP)

* 정적 shape 계약(예: `B, D` 고정) 기반 train_step 컴파일
* capture 후 replay(n) 반복 실행 가능
* 옵티마의 “호스트 메타(host meta)” 변경이 replay 동작에 반영 가능 (bc1/bc2 sanity)
* reset으로 상태를 초기화/복원 (복원 범위는 아래 Contract에서 고정)

---

## 레이어 분리 (책임 범위)

## 1) `aicf_fw/fw/*` (Framework UX 레이어)

**역할:** 사용자가 쓰는 API/UX를 고정한다. 에러 메시지, warmup 정책, 상태 머신을 관리한다.
**핵심 제공물:** `CompiledTrainStep` (사용자가 들고 있는 핸들)

* 입력 계약(spec) 관리

  * `B, D, device, dtype` 같은 “컴파일 타임 계약”을 명시적으로 요구/검증
* warmup 정책

  * `warmup_inputs` 제공 시 compile 단계에서 자동 warmup 수행
  * `warmup_required=True`면 warmup 없이 capture/replay 못하게 방어
* 상태 머신 관리

  * created → warmed → captured → replayable → reset
* naming/logging

  * `name="fw_mvp_train_step"` 같은 통합 네이밍
  * 그래프 begin/end, nvtx 범위 같은 사용자 친화 로그

관련 파일(현 트리 기준):

* `aicf_fw/fw/compile.py` : compile_train_step UX + 정책
* `aicf_fw/fw/train_step.py` : CompiledTrainStep 객체, 상태 전이
* `aicf_fw/fw/module.py` : fw 관점 모듈 인터페이스 보조
* `aicf_fw/fw/naming.py` : 일관된 이름 규칙

---

## 2) `aicf_fw/core_v2/*` (Engine 레이어)

**역할:** trace/IR/lower/plan/cuda_exec를 통해 “실행을 100% 결정”한다.
**핵심 제공물:** `Plan` / `Executor` / `CudaGraphExec` 류의 실행 핸들(내부 객체)

* trace (모델/옵티마 동작을 그래프화 가능한 형태로 수집)
* IR 생성 (`ir.py`)
* lowering (IR → backend ops / kernel 호출로 분해) (`lower.py`, `ops.py`)
* plan 생성 (메모리/메타 슬롯/실행 순서 고정) (`plan.py`)
* 실행 + 캡처/리플레이 (CUDA Graph) (`exec.py`, `cuda_exec.py`)
* 프린팅/디버깅 (`printer.py`)
* 결정론 보장 포인트 제공 (stream, synchronize, aliasing 규칙 등)

관련 파일(현 트리 기준):

* `aicf_fw/core_v2/trace.py`
* `aicf_fw/core_v2/ir.py`
* `aicf_fw/core_v2/lower.py`, `ops.py`
* `aicf_fw/core_v2/plan.py`
* `aicf_fw/core_v2/exec.py`, `cuda_exec.py`
* `aicf_fw/core_v2/compile.py` (엔진 단 compile 진입점)
* `aicf_fw/core_v2/printer.py`

---

## 데이터 흐름 (compile → 실행)

### 1) compile_train_step 호출 (FW)

입력: `(model, opt, B, D, device, dtype, warmup_inputs, warmup_runs, ...)`

FW는 아래를 수행:

1. 입력 계약 검증 (shape/dtype/device, required keys)
2. core_v2 compile 파이프라인 호출
3. warmup_inputs 있으면 즉시 warmup 실행하여:

   * 커널/플랜 준비
   * 캡처 가능한 정적 실행 경로 확인
4. `CompiledTrainStep` 반환

### 2) core_v2 compile 파이프라인 (Engine)

(개념 흐름)

* Trace: “train_step 1회”를 추적하기 위한 실행 경로 확보
* IR: 연산/메모리/메타(호스트 스칼라 등)를 표현
* Lower: IR을 실행 가능한 ops 시퀀스로 내림
* Plan:

  * param/state/workspace 배치 고정
  * meta slots(호스트 메타를 device 텐서로 전달하는 구멍) 고정
  * 실행 순서 고정
* Exec 객체: plan을 실행/캡처/리플레이할 핸들 제공

---

## 상태 머신 (FW에서 고정)

`CompiledTrainStep`는 다음 상태 전이를 가진다.

* **created**

  * compile 결과만 존재
  * warmup 전일 수 있음
* **warmed**

  * warmup 1회 이상 수행됨
  * 캡처에 필요한 리소스가 준비됨(정적 실행 경로 확인)
* **captured**

  * CUDA Graph 캡처 완료
  * replay 가능
* **reset**

  * 캡처/리플레이 관련 리소스 해제/초기화
  * 이후 재-warmup/재-capture 가능

권장 API 의미:

* `train_step(inputs)` : eager 실행 (graph capture 없이 실행)
* `capture(inputs)` : 해당 inputs spec으로 CUDA Graph 캡처
* `replay(n)` : 캡처된 그래프를 n번 반복 실행
* `reset()` : 캡처 상태를 풀고(필요 시) 상태를 초기화

---

## Contract (Stage6 / FW MVP에서 반드시 보장할 것)

아래는 “깨지면 안 되는” 고정 계약이다.

### Contract A — 업데이트 발생

* `train_step()` 1회 실행 후, 파라미터(예: `W0`)는 업데이트되어야 한다.

  * 테스트 기준: `|ΔW0| > 0`

### Contract B — capture/replay 업데이트 누적

* `capture()` 후 `replay(n)` 실행 시, 동일 그래프가 반복 실행되며 업데이트가 누적되어야 한다.

  * 테스트 기준: `replay(n=3)` 후 `|ΔW0| > 0` (그리고 대체로 eager 1회보다 커지는 경향)

### Contract C — host meta mutation이 replay behavior에 반영

* 호스트에서 meta 값을 바꾸면(예: `bc1_inv`, `bc2_inv`) replay 1회 업데이트량이 변해야 한다.

  * 테스트 기준: mutated vs restored의 `|ΔW0|`가 유의미하게 달라야 함

> 의도: 그래프는 “닫혀” 있지만, **메타 슬롯을 통해 동작을 조절**할 수 있어야 한다.
> (Adam bias-correction, lr, step, loss-scale 같은 값들이 여기에 해당)

### Contract D — reset의 의미(명확히 고정)

reset이 무엇을 “되돌리는지”를 고정해야 한다. Stage6에서는 아래 둘 중 하나로 확정하는 걸 추천:

* **옵션 1: reset = 캡처/그래프 리소스만 초기화** (권장)

  * 파라미터/옵티마 state는 건드리지 않음
  * capture/replay 핸들만 재구성
* **옵션 2: reset = “풀 복원”** (훈련 재현 목적)

  * 파라미터 + 옵티마 상태 + step까지 스냅샷으로 복원

지금 테스트 흐름상은 **옵션 1**이 UX가 단순하고 안전하다.
(풀 복원은 별도 `restore(snapshot)` 같은 API로 분리 추천)

---

## Host Meta 설계 원칙 (확장 대비)

### 핵심 원칙

* “호스트 메타”는 엔진이 접근하는 **명시적 레지스트리**로 관리한다.
* plan은 meta 슬롯(장치 텐서 포인터/버퍼)을 고정하고, replay 직전에 호스트가 값을 업데이트한다.

### 지금 이미 확인된 예

* `opt.bc1_inv`, `opt.bc2_inv`를 바꾸면 replay 결과가 바뀐다 → meta 경로가 살아있음

### 다음 확장 후보

* `lr`, `step`, `weight_decay`, `loss_scale`, `grad_clip_threshold`, `dropout_p`(가능하다면)

---

## 파일별 “한 줄 책임” (현재 트리 기준)

### core_v2

* `trace.py` : train_step 동작을 추적해 IR 생성의 입력을 만든다
* `ir.py` : 연산/텐서/메타/의존성을 표현하는 IR 구조
* `lower.py` : IR → 실행 가능한 ops 시퀀스로 변환
* `ops.py` : backend op 정의(커널 호출 단위), 메타/메모리 규약 포함
* `plan.py` : 메모리 배치, 메타 슬롯, 실행 순서를 고정한 실행 계획
* `exec.py` : plan 실행 핸들(런, 캡처/리플레이 연결)
* `cuda_exec.py` : CUDA stream/graph capture/replay 구체 구현
* `printer.py` : IR/plan/debug print 유틸
* `compile.py` : 위 과정을 묶어 “엔진 컴파일” 엔트리 제공

### fw

* `compile.py` : `compile_train_step` API + warmup 정책 + 상태 초기화
* `train_step.py` : `CompiledTrainStep` 구현(상태 머신, 메서드 노출)
* `module.py` : fw/nn 모듈 접점(필요 시 추상화)
* `naming.py` : 이름/태그 규칙

---

## 유지해야 할 골든 테스트 (지금 테스트를 표준으로)

* `v2_fw_mvp_train_step_test.py`는 Stage6의 “골든 테스트”다.
* 이 테스트가 보장하는 것은:

  1. eager 업데이트 발생
  2. capture/replay 업데이트 누적
  3. meta mutation에 따른 replay behavior 변경
  4. 전체 UX가 한 번에 동작(사용자 경험)

---

## 앞으로의 확장 방향 (엔진이 컴파일러답게 커지는 순서)

1. **연산 커버리지 확장**

* LN/GELU/Softmax, 더 다양한 loss, fused epilogue

2. **shape family / plan 캐싱**

* 정적 shape 유지하면서도 몇 가지 shape 그룹(B=32/64/128 등) 캐시

3. **control-flow 분기**

* 경로별 subgraph 캡처 + 런타임 dispatch (if/loop/early-exit)

4. **정확도/결정론 강화**

* TF32 정책, seed/stream 규약, aliasing/contiguous 규칙을 contract로 승격

---

## 이 문서의 “수정 금지” 구역

* 상태 머신 이름과 의미(created/warmed/captured/reset)
* Contract A/B/C (업데이트/리플레이/메타 반영)
* reset 의미(옵션 1로 고정할지 옵션 2로 고정할지 결정 후 잠금)

---

### 부록: 현재 FW MVP 테스트 결과 해석(짧게)

* eager에서 `|ΔW0| ≈ 0.0010` → 학습 업데이트 정상
* replay(n=3)에서 `|ΔW0| ≈ 0.0035` → 반복 실행 누적 정상
* meta mutated vs restored가 다름 → host meta 경로 정상
