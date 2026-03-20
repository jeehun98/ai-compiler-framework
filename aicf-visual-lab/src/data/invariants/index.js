import semanticConsistency from "./semantic_consistency.js";
import numericStability from "./numeric_stability.js";
import structuralPreservation from "./structural_preservation.js";

export const theoryInvariantList = [
  semanticConsistency,
  numericStability,
  structuralPreservation,
];

export const theoryInvariantIds = theoryInvariantList.map((item) => item.id);

export const theoryInvariantProfileKeys = theoryInvariantList.map(
  (item) => item.profileKey ?? item.id
);

export const theoryByInvariantId = Object.fromEntries(
  theoryInvariantList.map((item) => [item.id, item])
);

export const theoryByInvariantProfileKey = Object.fromEntries(
  theoryInvariantList.map((item) => [item.profileKey ?? item.id, item])
);

export const theoryIdToInvariantProfileKey = Object.fromEntries(
  theoryInvariantList.map((item) => [item.id, item.profileKey ?? item.id])
);

export const semanticInvariants = theoryInvariantList.filter(
  (item) => item.group === "semantic"
);

export const numericInvariants = theoryInvariantList.filter(
  (item) => item.group === "numeric"
);

export const structuralInvariants = theoryInvariantList.filter(
  (item) => item.group === "structural"
);

export const statefulInvariants = theoryInvariantList.filter(
  (item) => item.group === "stateful"
);

export const semanticInvariantIds = semanticInvariants.map((item) => item.id);
export const numericInvariantIds = numericInvariants.map((item) => item.id);
export const structuralInvariantIds = structuralInvariants.map((item) => item.id);
export const statefulInvariantIds = statefulInvariants.map((item) => item.id);

export const theoryInvariantGroups = [
  {
    id: "semantic",
    title: "Semantic Invariants",
    description:
      "Meaning-level conditions that must remain unchanged after transformation, lowering, or runtime specialization.",
    items: semanticInvariants,
  },
  {
    id: "numeric",
    title: "Numeric Invariants",
    description:
      "Conditions on numerical behavior such as stability, bounded deviation, and normalization-safe execution.",
    items: numericInvariants,
  },
  {
    id: "structural",
    title: "Structural Invariants",
    description:
      "Conditions on shape relations, dependency structure, and reduction contracts that must remain preserved.",
    items: structuralInvariants,
  },
  {
    id: "stateful",
    title: "Stateful Invariants",
    description:
      "Conditions ensuring consistent state evolution and valid state transition semantics.",
    items: statefulInvariants,
  },
];