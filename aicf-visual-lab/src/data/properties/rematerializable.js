// src/data/theory/properties/rematerializable.js

const rematerializable = {
  id: "Rematerializable",
  group: "reconstructive",
  profileKey: "rematerializable",
  title: "Rematerializable",
  subtitle: "Reconstruction Property",
  hero: {
    lead:
      "A computation is rematerializable when an intermediate value may be discarded and later reconstructed from still-available semantic dependencies at acceptable recomputation scope.",
    canonicalLatex:
      "Y = G(X),\\qquad Y \\text{ need not be stored if } Y \\text{ can be reconstructed from } X",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Reconstructible Intermediate",
          v: "중간값 Y가 이후 시점에도 남아 있는 의존성으로부터 다시 계산 가능해야 한다.",
        },
        {
          k: "Dependency Sufficiency",
          v: "재구성에 필요한 입력 또는 상태가 보존되어 있어야 한다.",
        },
        {
          k: "Semantic Equivalence by Recompute",
          v: "저장된 intermediate 대신 재계산된 값이 동일한 semantic role을 수행해야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "저장과 재구성을 교환하는 방식으로 동일한 의미를 유지할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "checkpointing, saved-tensor elimination, memory-pressure-aware recomputation을 정당화할 수 있다.",
        },
      ],
      latex:
        "Y = G(X),\\qquad Y \\text{ need not be stored if } Y \\text{ can be reconstructed from } X",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "workflow",
          title: "Reconstructibility",
          desc: "중간값은 이후에도 접근 가능한 semantic dependencies로부터 다시 계산 가능해야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "Dependency Preservation",
          desc: "재구성에 필요한 입력, state, control condition이 손실되지 않아야 한다.",
        },
        {
          id: "03",
          icon: "target",
          title: "Equivalent Reuse",
          desc: "재구성된 값은 저장된 intermediate와 동일한 downstream semantic role을 수행해야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Activation checkpointing",
        "Saved-intermediate elimination",
        "Memory-for-recompute tradeoff",
        "Recompute-based backward support",
      ],
    },
    boundary: {
      items: [
        "재구성에 필요한 upstream dependency가 이미 소실되면 rematerialization은 불가능하다.",
        "재계산 경로가 원래 intermediate와 다른 semantic state를 만들면 불법이다.",
        "hidden randomness, side effect, external state dependency가 있으면 재구성 가능성은 제한된다.",
      ],
    },
    relatedConstructions: {
      items: ["ActivationCheckpoint", "BackwardRecompute", "NormFactorRebuild"],
    },
    relatedTransforms: {
      items: [
        "Checkpoint insertion",
        "Intermediate eviction with recompute",
        "Backward rematerialization",
      ],
    },
  },
};

export default rematerializable;