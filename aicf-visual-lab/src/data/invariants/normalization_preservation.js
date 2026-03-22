const normalizationPreservation = {
  id: "NormalizationPreservation",
  profileKey: "normalization_preservation",
  group: "normalization-safety",
  title: "Normalization Preservation",
  subtitle: "Normalized meaning must survive online and blockwise execution",

  hero: {
    lead:
      "Normalization-heavy operators such as softmax, layer norm, and RMS norm may be transformed into online or blockwise realizations only if the normalized meaning remains valid.",
    canonicalLatex:
      "\\sum_i p_i = 1 \\quad \\text{or} \\quad \\mathrm{Norm}(x)=\\text{target structure}",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Same Normalized Meaning",
          v: "Changing execution shape must not change the intended normalized interpretation of the output.",
        },
        {
          k: "Stable Denominator",
          v: "The normalization factor must remain mathematically consistent across local and global merge stages.",
        },
        {
          k: "Distribution Validity",
          v: "Probability-like or scale-normalized outputs must remain valid under transformed realization.",
        },
      ],
      latex: "\\sum_i p_i = 1",
      preview: [
        {
          k: "Compiler View",
          v: "Full-pass normalization may be rewritten into streaming or tiled form only when the normalization state is preserved.",
        },
        {
          k: "Kernel View",
          v: "Rescaling, local maxima tracking, and merged statistics must still correspond to the original normalization definition.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "scale",
          title: "Mergeable Denominator State",
          desc:
            "The local normalization state must be mergeable into the same global denominator meaning.",
          metric: "D_{global}=\\mathrm{merge}(D_1,D_2,\\dots,D_n)",
        },
        {
          id: "02",
          icon: "gauge",
          title: "Stable Rescaling",
          desc:
            "When local scale changes, the rescaling step must restore a globally consistent interpretation.",
          metric: "\\tilde{s} \\rightarrow s_{global}",
        },
        {
          id: "03",
          icon: "shield",
          title: "Validity of Output Form",
          desc:
            "The final output must still satisfy the expected normalized structure.",
          metric: "\\sum_i p_i = 1",
        },
      ],
    },

    preserves: {
      items: [
        "Normalization meaning",
        "Stable denominator interpretation",
        "Probability or scale validity",
        "Equivalent normalized output structure",
      ],
    },

    failure: {
      items: [
        "online rescaling error changes the final normalized distribution",
        "blockwise merge produces a denominator inconsistent with the full-pass definition",
        "output no longer satisfies the intended normalization property",
      ],
    },

    downstreamImpact: {
      items: [
        {
          k: "Softmax Logits",
          v: "Broken normalization changes ranking confidence and downstream selection.",
        },
        {
          k: "Norm-Based Scaling",
          v: "Incorrect norm state distorts later activation scale and gradient behavior.",
        },
      ],
    },

    relatedConstructions: {
      items: [
        { op: "softmax", label: "Softmax" },
        { op: "layer_norm", label: "LayerNorm" },
        { op: "batch_norm", label: "BatchNorm" },
      ],
    },

    relatedTransforms: {
      items: [
        "Online softmax",
        "Blockwise normalization",
        "Welford merge",
        "Streaming denominator update",
      ],
    },
  },
};

export default normalizationPreservation;