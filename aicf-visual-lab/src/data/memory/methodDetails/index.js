import onlineNormDetail from "./onlineNorm";
import weightedReductionDetail from "./weightedReduction";
import rematerializationDetail from "./rematerialization";
import tileCompatibleDetail from "./tileCompatible";

export const memoryMethodDetails = {
  "online-norm": onlineNormDetail,
  "weighted-reduction": weightedReductionDetail,
  rematerialization: rematerializationDetail,
  "tile-compatible": tileCompatibleDetail,
};