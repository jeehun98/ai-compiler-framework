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



    def emit_backward(self, b, y_pred, y_true, grad_y, *, ctx):
        """
        MSE 미분: 2/N * (y_pred - y_true)
        이미터(mse_grad)의 인자명인 pred, target에 맞춰서 전달합니다.
        """
        g_pred = b.value(f"{self.name}.grad_input", b.values[y_pred].spec)
        
        # [Fix] y_pred -> pred, y_true -> target 으로 매핑
        emit_mse_grad(
            b, ctx, 
            pred=y_pred,    # 이미터의 pred 인자에 y_pred vid 전달
            target=y_true,  # 이미터의 target 인자에 y_true vid 전달
            out=g_pred
        )
        
        return {"input": g_pred}