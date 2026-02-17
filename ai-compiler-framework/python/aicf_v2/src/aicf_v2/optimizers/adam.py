import torch
from .base import Optimizer
from ..layers.adam_step import AdamStep
from ..tensor_spec import TensorSpec

class Adam(Optimizer):
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # 가중치 Vid별 상태 변수(M, V) 매핑 정보
        self.param_states = {}
        
        # 전역 상태: 편향 보정용 bc1, bc2
        # (v2 설계에 따라 (1,) 형상의 state로 등록)
        self.bc1_vid = model.state("adam.bc1", TensorSpec(shape=(1,), dtype="f32"))
        self.bc2_vid = model.state("adam.bc2", TensorSpec(shape=(1,), dtype="f32"))
        
        self.op_layer = AdamStep(
            name="optimizer.adam",
            lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps
        )

    def step(self):
        b = self.model.b
        ctx = self.model.ctx
        
        for p_vid, g_vid in self.model.parameter_grads.items():
            # 1. 해당 파라미터 전용 M, V 상태가 없다면 자동 생성
            if p_vid not in self.param_states:
                p_val = b.values[p_vid]
                m_vid = self.model.state(f"{p_val.name}.m", p_val.spec)
                v_vid = self.model.state(f"{p_val.name}.v", p_val.spec)
                self.param_states[p_vid] = (m_vid, v_vid)
            
            m_vid, v_vid = self.param_states[p_vid]
            
            # 2. AdamStep 연산 추가
            # P, M, V 모두 In-place 업데이트 되도록 설계된 Layer 호출
            self.op_layer.emit(
                b, 
                P=p_vid, G=g_vid, M=m_vid, V=v_vid, 
                bc1=self.bc1_vid, bc2=self.bc2_vid, 
                ctx=ctx
            )