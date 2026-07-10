// Source-ablation ROI model.
//
// The AUC ladder is MEASURED on the held-out test fold by
// src/parakh/eval/ablation.py and mirrored in artifacts/ablation.json:
//   GST only 0.714 -> + AA 0.736 -> + EPFO 0.752
// Everything else on this screen is an illustrative business overlay the CFO
// controls; only the AUC lift is measured.

export interface Rail {
  key: "aa" | "epfo";
  label: string;
  // AUC with this rail's sources included, from the measured ablation ladder.
  aucWith: number;
  // AUC of the immediately preceding stage, so the marginal lift is aucWith - aucBefore.
  aucBefore: number;
}

// GST is the baseline rail every MSME consents to first, so it has no marginal
// row here; AA and EPFO are the incremental consents whose value we quantify.
export const AUC_GST_ONLY = 0.714;

export const RAILS: Rail[] = [
  {
    key: "aa",
    label: "Account Aggregator (bank statements)",
    aucWith: 0.736,
    aucBefore: AUC_GST_ONLY,
  },
  {
    key: "epfo",
    label: "EPFO (workforce)",
    aucWith: 0.752,
    aucBefore: 0.736,
  },
];

export interface RoiInputs {
  // Total MSME book under management, in rupees crore.
  bookSizeCr: number;
  // Annual slippage (fresh NPA) as a percentage of the book.
  slippagePct: number;
  // Illustrative cost per consented data pull, in rupees, per rail.
  pullCost: Record<Rail["key"], number>;
  // Illustrative share of the book re-pulled and re-scored in a year.
  accountsScoredPerYear: number;
  // Illustrative fraction of the discrimination gain that converts to
  // provisioning identified early (operational capture, not a model number).
  captureFactor: number;
}

export interface RailRoi {
  key: Rail["key"];
  label: string;
  aucBefore: number;
  aucWith: number;
  aucLift: number;
  giniLift: number;
  provisioningIdentifiedEarly: number;
  pullSpend: number;
  netValue: number;
}

export interface RoiResult {
  slippageRupees: number;
  rails: RailRoi[];
  totalProvisioningIdentifiedEarly: number;
  totalPullSpend: number;
  totalNetValue: number;
}

export const DEFAULT_INPUTS: RoiInputs = {
  bookSizeCr: 40000,
  slippagePct: 2.5,
  pullCost: { aa: 10, epfo: 5 },
  accountsScoredPerYear: 250000,
  captureFactor: 0.35,
};

const CRORE = 1e7;

// Gini = 2 * AUC - 1. A rail's marginal Gini lift is the standard measure of the
// extra rank-ordering power it buys, and we let a share of that gain (the
// illustrative capture factor) convert into slippage provisioned for early.
export function computeRoi(inputs: RoiInputs): RoiResult {
  const slippageRupees = (inputs.bookSizeCr * CRORE * inputs.slippagePct) / 100;

  const rails: RailRoi[] = RAILS.map((rail) => {
    const aucLift = rail.aucWith - rail.aucBefore;
    const giniLift = 2 * aucLift;
    const provisioningIdentifiedEarly = slippageRupees * giniLift * inputs.captureFactor;
    const pullSpend = inputs.pullCost[rail.key] * inputs.accountsScoredPerYear;
    return {
      key: rail.key,
      label: rail.label,
      aucBefore: rail.aucBefore,
      aucWith: rail.aucWith,
      aucLift,
      giniLift,
      provisioningIdentifiedEarly,
      pullSpend,
      netValue: provisioningIdentifiedEarly - pullSpend,
    };
  });

  const totalProvisioningIdentifiedEarly = rails.reduce(
    (sum, rail) => sum + rail.provisioningIdentifiedEarly,
    0,
  );
  const totalPullSpend = rails.reduce((sum, rail) => sum + rail.pullSpend, 0);

  return {
    slippageRupees,
    rails,
    totalProvisioningIdentifiedEarly,
    totalPullSpend,
    totalNetValue: totalProvisioningIdentifiedEarly - totalPullSpend,
  };
}
