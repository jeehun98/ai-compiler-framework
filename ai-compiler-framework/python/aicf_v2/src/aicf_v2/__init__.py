import sys
from pathlib import Path

# 1) 빌드 폴더의 바이너리 우선 로드 설정
_build_path = Path(__file__).parents[4] / "build" / "python" / "aicf_cuda"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import _C
except ImportError:
    from . import _C

# 2) 기초 유틸리티 및 모델
from .tensor_spec import TensorSpec
from .model import Model, Sequential

# 3) 기본 연산 레이어 (통합 이름 적용)
from .layers.linear import Linear
from .layers.relu import ReLU
from .layers.add import Add
from .layers.softmax import Softmax
from .layers.copy import Copy
from .layers.reduce_sum import ReduceSum

# 4) 정규화 및 손실 함수 (Fwd/Bwd 통합 완료)
from .layers.batchnorm import BatchNorm
from .layers.layernorm import LayerNorm
from .layers.cross_entropy import CrossEntropyLoss
from .layers.mse_loss import MSELoss
from .layers.mse_grad import MseGrad

# 5) 학습 보조 및 옵티마이저 레이어
from .layers.adam_step import AdamStep
from .layers.sgd_step import SgdStep
from .layers.grad_zero import GradZero
from .layers.step_inc import StepInc
from .layers.bias_corr import BiasCorr

# 6) 런타임
from .runtime.cuda_exec import CudaExecutor

__all__ = [
    "TensorSpec",
    "Model",
    "Sequential",
    "Linear",
    "ReLU",
    "Softmax",
    "CrossEntropyLoss",
    "MSELoss",
    "Add",
    "AdamStep",
    "SgdStep",
    "BatchNorm",    # 통합됨
    "LayerNorm",    # 통합됨
    "CudaExecutor",
    "ReduceSum",
    "MseGrad",
    "GradZero",
    "Copy",
    "StepInc",
    "BiasCorr",
]