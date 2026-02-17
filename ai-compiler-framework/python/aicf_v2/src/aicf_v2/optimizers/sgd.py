from .base import Optimizer
from ..layers.sgd_step import SgdStep # Layer로 정의된 객체

class SGD(Optimizer):
    def __init__(self, model, lr=1e-3):
        super().__init__(model)
        self.lr = lr
        # 연산 단위인 Layer 인스턴스 소유 (설정 공유)
        self.op_layer = SgdStep(name="optimizer.sgd", lr=self.lr)

    def step(self):
        # 모델이 build_backward를 통해 수집한 모든 파라미터-그라디언트 쌍 순회
        for p_vid, g_vid in self.model.parameter_grads.items():
            # Layer 규약에 따라 emit 호출 (내부에서 sgd_step 이미터 실행)
            # P와 outP를 동일하게 설정하여 In-place 유도
            self.op_layer.emit(
                self.model.b, 
                P=p_vid, G=g_vid, 
                ctx=self.model.ctx
            )