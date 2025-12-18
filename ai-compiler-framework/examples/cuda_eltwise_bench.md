ncu   --target-processes all   --nvtx   --nvtx-include "aicf::eltwise::"   --section LaunchStats   --section SpeedOfLight   --page details   .\aicf_eltwise_bench.exe



ncu   --target-processes all   --nvtx   --nvtx-include "aicf::eltwise::add_f32,aicf::eltwise::relu_f32"   --section LaunchStats   --page details   .\aicf_eltwise_bench.exe
ncu --target-processes all --nvtx --nvtx-include "regex:.*aicf::eltwise::.*" --section LaunchStats --section SpeedOfLight --page details .\aicf_eltwise_bench.exe
