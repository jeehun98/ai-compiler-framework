const decisionTolerance = {
  id: "DecisionTolerance",
  profileKey: "decision_tolerance",
  group: "downstream-aware",
  title: "Decision Tolerance",
  subtitle: "Numeric change is allowed only while downstream decisions remain stable",

  hero: {
    lead:
      "Exact numeric equality is not always required. A transformed path remains acceptable only while downstream decisions such as argmax, top-k, sign, routing, or activation boundary stay meaningfully unchanged.",
    canonicalLatex:
      "\\mathcal{D}(\\hat{Y}, Y) \\leq \\tau \\quad \\text{under downstream-safe boundary}",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Decision-Aware Correctness",
          v: "Correctness is judged not only by numeric proximity but by stability of downstream decisions.",
        },
        {
          k: "Boundary-Sensitive Tolerance",
          v: "Small numeric drift is acceptable unless it crosses rank, sign, routing, or threshold boundaries.",
        },
        {
          k: "Use-Case-Dependent Acceptance",
          v: "Tolerance must reflect how the tensor is actually consumed later.",
        },
      ],
      latex:
        "\\arg\\max(\\hat{Y}_i)=\\arg\\max(Y_i) \\quad \\text{or} \\quad \\mathrm{TopK}(\\hat{Y}_i,k)=\\mathrm{TopK}(Y_i,k)",
      preview: [
        {
          k: "Compiler View",
          v: "A numerically different realization may still be acceptable when downstream semantics remain stable.",
        },
        {
          k: "Validation View",
          v: "Task-aware checks often matter more than raw L2 closeness.",
        },
      ],
    },

    constraints: {
      cards: [
        {
          id: "T1",
          title: "Argmax Stability",
          desc:
            "The row-wise maximum index should remain unchanged when the downstream path depends on a single winner.",
          metric: "\\arg\\max \\hat{Y}_i = \\arg\\max Y_i",
        },
        {
          id: "T2",
          title: "Top-K Preservation",
          desc:
            "Candidate membership inside the important set should remain stable for ranking-sensitive paths.",
          metric:
            "\\mathrm{TopK}(\\hat{Y}_i,k)=\\mathrm{TopK}(Y_i,k)",
        },
        {
          id: "T3",
          title: "Sign Integrity",
          desc:
            "Values outside a safety margin must not flip sign when later logic depends on sign or threshold.",
          metric:
            "\\mathrm{sign}(\\hat{y})=\\mathrm{sign}(y),\\ |y|>\\eta",
        },
        {
          id: "T4",
          title: "Relative Drift Budget",
          desc:
            "Numeric change must stay below a practical relative error threshold for the intended downstream interpretation.",
          metric:
            "\\frac{|y-\\hat{y}|}{\\max(|y|,\\beta)} < \\epsilon_{rel}",
        },
      ],
    },

    preserves: {
      items: [
        "Argmax / ranking stability when required",
        "Sign or threshold boundary integrity",
        "Task-aware correctness under numeric variation",
      ],
    },

    failure: {
      items: [
        "top-1 or top-k result changes in a ranking-sensitive downstream path",
        "sign flip changes activation or gating behavior",
        "numeric drift crosses a business-meaningful decision boundary",
      ],
    },

    signals: {
      bullets: [
        {
          k: "Selection Stability",
          v: "Classifier, router, or retrieval logic often cares more about selected identity than exact numeric value.",
        },
        {
          k: "Activation Boundary Stability",
          v: "ReLU or gate-like structures are sensitive to sign and threshold crossings.",
        },
        {
          k: "Distribution Interpretation",
          v: "A changed score distribution can alter later normalization or confidence interpretation.",
        },
      ],
    },

    applicableScenarios: {
      items: [
        "Classifier projection GEMM",
        "Retrieval / ranking score GEMM",
        "Attention score path",
        "Activation-preceding fused GEMM",
      ],
    },

    relatedConstructions: {
      items: [
        { op: "gemm", label: "Classifier / Score GEMM" },
        { op: "relu", label: "Activation Boundary" },
        { op: "softmax", label: "Ranking-sensitive Logits" },
      ],
    },

    relatedTransforms: {
      items: [
        "Mixed precision realization under argmax guard",
        "Rank-preserving approximation",
        "Sign-safe accumulation path",
        "Task-aware validation instead of exact equality",
      ],
    },
  },
};

export default decisionTolerance;