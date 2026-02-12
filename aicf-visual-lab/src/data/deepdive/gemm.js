export const gemmDeepDive = {
  id: "GEMM",
  // App.jsx의 KernelDeepDive 컴포넌트가 사용하는 키 이름으로 변경
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "Naïve Baseline",
      throughput: "245.2 GF/s",
      description: "Global memory에서 직접 데이터를 읽어오는 방식. Memory Bound가 심각하여 연산 유닛이 대부분 유휴 상태로 방치됨.",
    },
    {
      version: "v2.0",
      tag: "Shared Memory Tiling",
      throughput: "1,420.5 GF/s",
      description: "32x32 타일링 적용. 데이터 재사용성을 극대화하여 DRAM 대역폭 요구량을 획기적으로 낮춤.",
    },
    {
      version: "v3.0",
      tag: "TensorCore & Vectorized IO",
      throughput: "3,188.9 GF/s",
      description: "Ampere 아키텍처 WMMA API 사용 및 Float4 벡터 로드 적용. Bank Conflict를 피하기 위한 패딩 기법 도입.",
    }
  ],
  // App.jsx의 KernelDeepDive 컴포넌트가 사용하는 키 이름으로 변경
  profiling_report: {
    "SM_Occupancy": "92.4%",
    "L1_Cache_Hit_Rate": "94.1%",
    "Tensor_Core_Utilization": "88.5%",
    "DRAM_Throughput": "74.2%",
    "Warp_Execution_Efficiency": "98.2%"
  },
  analysis: "Shared Memory에 타일을 적재할 때, 32개 뱅크가 동시에 접근되도록 레이아웃을 정렬함(Stride-aware Padding). 이를 통해 v2 대비 지연 시간을 15% 추가 단축함."
};