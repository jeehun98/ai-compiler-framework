# python/aicf_v2/src/aicf_v2/__init__.py 내에서
import sys
from pathlib import Path

# 빌드 폴더의 바이너리를 우선적으로 찾도록 설정
# 프로젝트 루트를 찾아 build/python/aicf_cuda 경로를 추가
_build_path = Path(__file__).parents[4] / "build" / "python" / "aicf_cuda"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import _C
except ImportError:
    # 빌드 폴더에 없으면 현재 패키지 내부에 복사된 것을 시도
    from . import _C

from .tensor_spec import TensorSpec
from .model import Model, Sequential

from .layers.linear import Linear
from .layers.relu import ReLU
from .layers.relu_bwd import ReLUBwd
from .layers.add import Add
from .layers.adam_step import AdamStep
from .layers.sgd_step import SgdStep
from .layers.batchnorm import BatchNormFwd, BatchNormBwd
from .layers.layernorm import LayerNormFwd, LayerNormBwd
from .layers.reduce_sum import ReduceSum
from .layers.mse_grad import MseGrad
from .layers.copy import Copy
from .layers.grad_zero import GradZero
from .layers.step_inc import StepInc
from .layers.bias_corr import BiasCorr
from .layers.gemm_epilogue import GemmEpilogue

from .runtime.cuda_exec import CudaExecutor

__all__ = [
    "TensorSpec",
    "Model",
    "Sequential",
    "Linear",
    "GemmEpilogue",
    "ReLU",
    "ReLUBwd",
    "Add",
    "AdamStep",
    "BatchNormFwd",
    "BatchNormBwd",
    "LayerNormFwd",
    "LayerNormBwd",
    "CudaExecutor",
    "SgdStep",
    "ReduceSum",
    "MseGrad",
    "GradZero",
    "Copy",
    "StepInc",
    "BiasCorr",
]
