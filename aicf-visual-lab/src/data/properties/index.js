// src/data/theory/properties/index.js

import associativeMerge from "./associative_merge.js";
import domainPrunable from "./domain_prunable.js";
import localAccumulable from "./local_accumulable.js";
import orderRewritable from "./order_rewritable.js";
import precisionRelaxable from "./precision_relaxable.js";
import rematerializable from "./rematerializable.js";
import representationInvariant from "./representation_invariant.js";
import scheduleInvariant from "./schedule_invariant.js";
import tileComposable from "./tile_composable.js";

export const theoryPropertyList = [
  associativeMerge,
  domainPrunable,
  localAccumulable,
  orderRewritable,
  precisionRelaxable,
  rematerializable,
  representationInvariant,
  scheduleInvariant,
  tileComposable,
];

export const theoryPropertyIds = theoryPropertyList.map((item) => item.id);

export const theoryPropertyProfileKeys = theoryPropertyList.map(
  (item) => item.profileKey ?? item.id
);

export const theoryByPropertyId = Object.fromEntries(
  theoryPropertyList.map((item) => [item.id, item])
);

export const theoryByProfileKey = Object.fromEntries(
  theoryPropertyList.map((item) => [item.profileKey ?? item.id, item])
);

export const theoryIdToProfileKey = Object.fromEntries(
  theoryPropertyList.map((item) => [item.id, item.profileKey ?? item.id])
);

export const foundationalProperties = theoryPropertyList.filter(
  (item) => item.group === "foundational"
);

export const reconstructiveProperties = theoryPropertyList.filter(
  (item) => item.group === "reconstructive"
);

export const structuralProperties = theoryPropertyList.filter(
  (item) => item.group === "structural"
);

export const foundationalPropertyIds = foundationalProperties.map(
  (item) => item.id
);

export const reconstructivePropertyIds = reconstructiveProperties.map(
  (item) => item.id
);

export const structuralPropertyIds = structuralProperties.map(
  (item) => item.id
);

export const theoryPropertyGroups = [
  {
    id: "foundational",
    title: "Foundational Properties",
    description:
      "Semantic and algebraic laws that directly determine when a transformation is meaning-preserving.",
    items: foundationalProperties,
  },
  {
    id: "reconstructive",
    title: "Reconstructive Properties",
    description:
      "Properties governing whether discarded or omitted values may be recovered from preserved dependencies without changing semantic meaning.",
    items: reconstructiveProperties,
  },
  {
    id: "structural",
    title: "Structural Properties",
    description:
      "Execution and decomposition forms that become realizable when higher-level semantic properties hold.",
    items: structuralProperties,
  },
];