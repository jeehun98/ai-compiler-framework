from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda.mse_loss import mse_loss as emit_mse_loss
from ..emitters.cuda.mse_grad import mse_grad as emit_mse_grad

class MSELoss(Layer):
    def __init__(self, reduction: str = "mean", name: str = "mse"):
        super().__init__(name)
        self.reduction = reduction

    def emit(self, b, y_pred, y_true, *, ctx):
        # 결과값은 항상 스칼라(f32)
        out_spec = TensorSpec(shape=(1,), dtype="f32", device=b.device)
        out_vid = b.value(f"{self.name}.out", out_spec)
        
        emit_mse_loss(b, ctx, pred=y_pred, target=y_true, out=out_vid, reduction=self.reduction)
        return out_vid



    def emit_backward(self, b, ctx, inputs, outputs, grad_y, **kwargs):
        """
        새로운 표준 규격 적용:
        - inputs[0]: y_pred (Forward 입력 1)
        - inputs[1]: y_true (Forward 입력 2)
        """
        y_pred = inputs[0]
        y_true = inputs[1]
        
        # 출력 미분값(dy)을 입력 미분값(dx)으로 변환
        g_pred = b.value(f"{self.name}.grad_input", b.values[y_pred].spec)
        
        # 이미 정의된 mse_grad 이미터 호출 (인자명: pred, target)
        from ..emitters.cuda.mse_grad import mse_grad as emit_mse_grad
        emit_mse_grad(b, ctx, pred=y_pred, target=y_true, out=g_pred)
        
        return {"input": g_pred}