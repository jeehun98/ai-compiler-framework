import orderRewritable from "./order_rewritable.js";
import associativeMerge from "./associative_merge.js";
import scheduleInvariant from "./schedule_invariant.js";
import tileComposable from "./tile_composable.js";
import localAccumulable from "./local_accumulable.js";
import layoutFlexible from "./layout_flexible.js";
import domainPrunable from "./domain_prunable.js";
import rematerializable from "./rematerializable.js";
import precisionRelaxable from "./precision_relaxable.js";

export const theoryPropertyList = [
  orderRewritable,
  associativeMerge,
  scheduleInvariant,
  tileComposable,
  localAccumulable,
  layoutFlexible,
  domainPrunable,
  rematerializable,
  precisionRelaxable,
];

export const theoryPropertyIds = theoryPropertyList.map((item) => item.id);

export const theoryByPropertyId = Object.fromEntries(
  theoryPropertyList.map((item) => [item.id, item])
);