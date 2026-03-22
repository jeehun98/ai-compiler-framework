const domainPruningPreservation = {
  id: "DomainPruningPreservation",
  profileKey: "domain_pruning_preservation",
  group: "pruning",
  title: "Domain Pruning Preservation",
  subtitle: "Skipping inactive or irrelevant regions must preserve observable meaning",

  hero: {
    lead:
      "Mask-based skipping, inactive-region elimination, and domain pruning are allowed only when omitted regions do not affect the final observable result.",
    canonicalLatex:
      "f(X) = f(X_{active}) \\quad \\text{if } X_{inactive} \\text{ is semantically irrelevant}",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Irrelevant Region Elimination",
          v: "Inputs that provably do not affect the output may be skipped.",
        },
        {
          k: "Boundary-Safe Pruning",
          v: "Pruning must respect sign, mask, threshold, and structural boundaries.",
        },
        {
          k: "Observable Equivalence",
          v: "The final externally visible result must remain unchanged after pruning.",
        },
      ],
      latex: "f(X)=f(X\\setminus X_{pruned})",
      preview: [
        {
          k: "Compiler View",
          v: "Masked domains, inactive branches, and prunable regions may be removed from execution only under semantic proof.",
        },
        {
          k: "Kernel View",
          v: "Skipping work is valid only when omitted loads, multiplies, or writes cannot change downstream output.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "target",
          title: "Mask Correctness",
          desc:
            "Masked or pruned regions must truly be irrelevant under the intended semantics.",
          metric: "M(x)=0 \\Rightarrow \\text{no observable contribution}",
        },
        {
          id: "02",
          icon: "shield",
          title: "Boundary Integrity",
          desc:
            "Threshold, sign, and activation boundaries must not be crossed by pruning.",
          metric: "\\mathrm{boundary}(x) \\text{ unchanged}",
        },
        {
          id: "03",
          icon: "boxes",
          title: "No Hidden Dependency",
          desc:
            "Pruned values must not reappear indirectly through later merge or normalization.",
          metric: "\\partial y / \\partial x_{pruned} = 0",
        },
      ],
    },

    preserves: {
      items: [
        "Observable output equivalence",
        "Mask and threshold semantics",
        "Pruning-safe inactive region elimination",
      ],
    },

    failure: {
      items: [
        "a pruned value still affects later normalization or merge state",
        "mask interpretation differs across transformed execution paths",
        "pruning crosses sign or threshold boundary and changes activation behavior",
      ],
    },

    downstreamImpact: {
      items: [
        {
          k: "Activation Boundary",
          v: "Incorrect pruning can change which units activate or remain silent.",
        },
        {
          k: "Masked Attention / Sparse Paths",
          v: "Pruned regions must not leak back into score or routing behavior.",
        },
      ],
    },

    relatedConstructions: {
      items: [
        { op: "relu", label: "ReLU" },
        { op: "softmax", label: "Masked Softmax" },
        { op: "bias_add", label: "Masked/Broadcast Paths" },
      ],
    },

    relatedTransforms: {
      items: [
        "Inactive lane skipping",
        "Masked region elimination",
        "Sparse-domain pruning",
        "Early exit on provably irrelevant branch",
      ],
    },
  },
};

export default domainPruningPreservation;