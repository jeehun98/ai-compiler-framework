#pragma once

// v0: launcher.cu 안에서 커널 정의까지 다 들고 가는 구조를 유지하기 위한 파일.
//     (기존 gemm/launcher.cu 패턴과 동일)
// 이후에 커널 정의를 분리하고 싶으면 여기로 __global__ 선언만 옮기면 됨.
