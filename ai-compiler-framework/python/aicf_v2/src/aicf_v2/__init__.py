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

# 기초 유틸리티 및 모델
from .tensor_spec import TensorSpec
from .model import Model, Sequential

# 레이어 (Layers)
from .layers.linear import Linear
from .layers.relu import ReLU
from .layers.add import Add
from .layers.softmax import Softmax          # [추가]
from .layers.cross_entropy import CrossEntropyLoss  # [추가]
from .layers.mse import MSELoss              # [추가] 기존 MseGrad 대신 고수준 레이어 명칭

# 학습 보조 레이어 및 옵티마이저 관련
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

# 런타임
from .runtime.cuda_exec import CudaExecutor

__all__ = [
    "TensorSpec",
    "Model",
    "Sequential",
    "Linear",
    "GemmEpilogue",
    "ReLU",
    "Softmax",            # [추가]
    "CrossEntropyLoss",   # [추가]
    "MSELoss",            # [추가]
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