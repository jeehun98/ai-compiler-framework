import reductionEquivalence from "./reduction_equivalence";
import normalizationPreservation from "./normalization_preservation";
import domainPruningPreservation from "./domain_pruning_preservation";
import tiledExecutionEquivalence from "./tiled_execution_equivalence";
import representationEquivalence from "./representation_equivalence";
import boundedNumericDrift from "./bounded_numeric_drift";
import decisionTolerance from "./decision_tolerance";

export const theoryInvariantGroups = [
  {
    id: "execution-meaning",
    title: "Execution Meaning Preservation",
    description:
      "Execution schedule, merge order, tiling, and representation may change, but the intended operation meaning must remain intact.",
    items: [
      reductionEquivalence,
      tiledExecutionEquivalence,
      representationEquivalence,
    ],
  },
  {
    id: "normalization-safety",
    title: "Normalization & Distribution Safety",
    description:
      "Normalization factor, denominator interpretation, and stable output form must survive online or blockwise realization.",
    items: [normalizationPreservation, boundedNumericDrift],
  },
  {
    id: "pruning",
    title: "Pruning & Boundary Safety",
    description:
      "Skipping masked or inactive regions is allowed only when it does not alter observable meaning or downstream boundary behavior.",
    items: [domainPruningPreservation],
  },
  {
    id: "downstream-aware",
    title: "Downstream Decision Tolerance",
    description:
      "Exact numeric equality is not always required, but downstream decisions such as argmax, top-k, sign, or routing must stay stable when required.",
    items: [decisionTolerance],
  },
];

export const theoryByInvariantId = Object.fromEntries(
  theoryInvariantGroups.flatMap((group) =>
    group.items.map((item) => [item.id, item])
  )
);

export const theoryIdToInvariantProfileKey = Object.fromEntries(
  theoryInvariantGroups.flatMap((group) =>
    group.items.map((item) => [item.id, item.profileKey])
  )
);

export default theoryInvariantGroups;