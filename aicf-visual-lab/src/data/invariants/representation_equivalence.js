const representationEquivalence = {
  id: "RepresentationEquivalence",
  profileKey: "representation_equivalence",
  group: "execution-meaning",
  title: "Representation Equivalence",
  subtitle: "Different internal forms may be used only if the same meaning is preserved",

  hero: {
    lead:
      "Internal state, intermediate structure, or numeric form may change only when the transformed representation still encodes the same intended computation.",
    canonicalLatex:
      "\\phi(X) \\equiv X \\quad \\text{with respect to observable semantics}",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Same Observable Meaning",
          v: "Changing internal representation must not change the externally interpreted computation.",
        },
        {
          k: "Form-Safe Rewrite",
          v: "Alternate forms such as scaled, factorized, packed, or fused states are legal only if they preserve meaning.",
        },
        {
          k: "Recoverable Interpretation",
          v: "The transformed form must still map back to the same mathematical intent.",
        },
      ],
      latex: "\\phi(X) \\sim X",
      preview: [
        {
          k: "Compiler View",
          v: "Representation rewrite is a semantic-preserving change of form, not a change of task.",
        },
        {
          k: "Kernel View",
          v: "Packed values, fused intermediates, and rescaled states must still correspond to the original operator meaning.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "binary",
          title: "Equivalent Form Mapping",
          desc:
            "The transformed representation must preserve the same semantic interpretation under the operator.",
          metric: "\\mathrm{eval}(\\phi(X)) = \\mathrm{eval}(X)",
        },
        {
          id: "02",
          icon: "lock",
          title: "No Hidden Semantic Shift",
          desc:
            "A representation rewrite must not silently alter normalization, indexing, or scale semantics.",
          metric: "\\phi(X) \\not\\Rightarrow \\text{semantic drift}",
        },
        {
          id: "03",
          icon: "orbit",
          title: "Recoverable Meaning",
          desc:
            "The transformed form must remain interpretable as the same original intent.",
          metric: "\\phi^{-1}(\\phi(X)) \\approx X",
        },
      ],
    },

    preserves: {
      items: [
        "Observable semantic identity",
        "Equivalent operator interpretation",
        "Safe alternate internal form",
      ],
    },

    failure: {
      items: [
        "rewritten form changes scale or indexing semantics",
        "fused or packed representation is no longer equivalent to the original computation",
        "transformed state cannot be interpreted as the same mathematical object",
      ],
    },

    downstreamImpact: {
      items: [
        {
          k: "Fused Epilogues",
          v: "Intermediate elimination is safe only if later interpretation stays equivalent.",
        },
        {
          k: "Packed / Vectorized Paths",
          v: "Packing must not alter logical value meaning or lane interpretation.",
        },
      ],
    },

    relatedConstructions: {
      items: [
        { op: "gemm", label: "GEMM / Packed Path" },
        { op: "bias_add", label: "Bias Fusion" },
        { op: "adam_step", label: "Fused Optimizer State" },
      ],
    },

    relatedTransforms: {
      items: [
        "Packed vector representation",
        "Fused epilogue form",
        "Scaled internal state",
        "Intermediate elimination with recoverable meaning",
      ],
    },
  },
};

export default representationEquivalence;