// How each CUAD clause type is presented: a tier (how much attention it deserves when reviewing), a one-line
// plain-English explanation, and the fine-tuned model's held-out span F1, read straight from the eval results so
// the UI can flag the types the model is weak at.

import results from "../../../eval/results/r8-v1.json";
import { CLAUSE_TYPES, type ClauseType } from "./scan-prompt";

export type Tier = "watch" | "note" | "fact";

export const TIER_LABEL: Record<Tier, string> = { watch: "Watch", note: "Note", fact: "Key facts" };

/** Below this held-out span F1 a type is shown with a "low accuracy" tag. */
export const LOW_F1 = 0.3;

const INFO: Record<ClauseType, [Tier, string]> = {
  "Document Name": ["fact", "The name of the agreement."],
  Parties: ["fact", "Who is bound by the contract."],
  "Agreement Date": ["fact", "When the contract was signed."],
  "Effective Date": ["fact", "When the obligations start."],
  "Expiration Date": ["fact", "When the initial term ends."],
  "Governing Law": ["note", "Which jurisdiction's law applies in a dispute."],

  "Uncapped Liability": ["watch", "A party's liability has no upper limit, for example for some breaches."],
  "Non-Compete": ["watch", "Restricts a party from competing, in a market, region or time period."],
  Exclusivity: ["watch", "Requires dealing only with the other party, or blocks dealing with others."],
  "Liquidated Damages": ["watch", "A fixed sum is payable on breach or termination."],
  "Change Of Control": ["watch", "A merger or sale of a party lets the other terminate, consent or renegotiate."],
  "Anti-Assignment": ["watch", "The contract can't be transferred without consent (or notice)."],
  "Termination For Convenience": ["watch", "A party can end the contract without cause."],
  "Most Favored Nation": ["watch", "A party must get terms at least as good as any third party gets."],
  "Ip Ownership Assignment": ["watch", "Intellectual property created under the contract is assigned to a party."],
  "Joint Ip Ownership": ["watch", "Some intellectual property is owned jointly."],
  "Minimum Commitment": ["watch", "A minimum order, purchase or payment is required."],
  "Volume Restriction": ["watch", "Usage above a threshold costs more or needs consent."],
  "Price Restrictions": ["watch", "Limits on raising or lowering prices."],
  "Covenant Not To Sue": ["watch", "A party agrees not to challenge rights or bring claims."],
  "No-Solicit Of Customers": ["watch", "A party may not solicit the other's customers."],
  "No-Solicit Of Employees": ["watch", "A party may not hire or solicit the other's employees."],
  "Non-Disparagement": ["watch", "A party may not criticise the other."],
  "Rofr/Rofo/Rofn": ["watch", "A right of first refusal, offer or negotiation before dealing with others."],
  "Revenue/Profit Sharing": ["watch", "Revenue or profit must be shared with the other party."],
  "Competitive Restriction Exception": ["note", "An exception to a non-compete, exclusivity or no-solicit clause."],

  "Cap On Liability": ["note", "Liability is capped at an amount or limited in time."],
  "Audit Rights": ["note", "A party may audit the other's books or compliance."],
  Insurance: ["note", "Insurance a party must hold."],
  "Warranty Duration": ["note", "How long a warranty lasts."],
  "Renewal Term": ["note", "How and for how long the contract renews."],
  "Notice Period To Terminate Renewal": ["note", "How much notice is needed to stop a renewal."],
  "Post-Termination Services": ["note", "Obligations that continue after the contract ends."],
  "Source Code Escrow": ["note", "Source code is deposited with a third party for release on set events."],
  "Third Party Beneficiary": ["note", "Someone outside the contract can enforce it."],
  "License Grant": ["note", "A licence granted by one party to the other."],
  "Affiliate License-Licensor": ["note", "The licence covers IP owned by the licensor's affiliates."],
  "Affiliate License-Licensee": ["note", "The licence extends to the licensee's affiliates."],
  "Irrevocable Or Perpetual License": ["note", "A licence that can't be revoked or never expires."],
  "Non-Transferable License": ["note", "The licensee can't transfer or sublicense the licence."],
  "Unlimited/All-You-Can-Eat-License": ["note", "Unlimited use for a fixed price."],
};

type Score = { f1: number; tp: number; fn: number };
const BY_TYPE = results.span.by_type as Record<string, Score>;

export interface ClauseInfo {
  type: ClauseType;
  tier: Tier;
  explain: string;
  /** Held-out span F1 of the fine-tuned model (97 CUAD test contracts). */
  f1: number;
  /** Gold spans of this type in the test set; small support makes the F1 noisy. */
  support: number;
  lowAccuracy: boolean;
}

export const CLAUSES: Record<ClauseType, ClauseInfo> = Object.fromEntries(
  CLAUSE_TYPES.map((type) => {
    const [tier, explain] = INFO[type];
    const s = BY_TYPE[type] ?? { f1: 0, tp: 0, fn: 0 };
    return [type, { type, tier, explain, f1: s.f1, support: s.tp + s.fn, lowAccuracy: s.f1 < LOW_F1 }];
  }),
) as Record<ClauseType, ClauseInfo>;
