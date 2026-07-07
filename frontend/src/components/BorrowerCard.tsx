import type { Borrower } from "../data/borrowers";
import type { ConsentResponse, ScoreResponse } from "../api/types";
import { inr } from "../format";
import { DimensionRings } from "./DimensionRings";
import { ReasonList } from "./ReasonList";
import { ScoreGauge } from "./ScoreGauge";
import { WhatIfCoach } from "./WhatIfCoach";

interface Props {
  borrower: Borrower;
  score: ScoreResponse;
  consent: ConsentResponse | null;
  onConsent: () => void;
}

export function BorrowerCard({ borrower, score, consent, onConsent }: Props) {
  return (
    <div className="phone-stage">
      <div className="phone">
        <div className="phone-notch" />
        <div className="phone-screen">
      <div className="phone-head">
        <span className="phone-app">परख · Parakh</span>
        <span className="phone-firm">{borrower.name} · {borrower.location}</span>
        <span className="phone-gstin">GSTIN {borrower.gstin}</span>
      </div>

      <div className="consent-chip">
        <span>
          {consent
            ? `Data via Account Aggregator · consent valid till ${consent.consent_expiry.slice(0, 10)}`
            : "Approve Account Aggregator consent to view your card"}
        </span>
        <button onClick={onConsent}>{consent ? "Approved" : "Approve"}</button>
      </div>

      <ScoreGauge score={score.score} grade={score.grade} />
      <div className="pill-row">
        <span className="pill limit">{inr(score.prequalified_limit)} pre-qualified</span>
      </div>

      <DimensionRings dimensions={score.dimensions} />

      <div className="sub">Why this score · कारण</div>
      <ReasonList codes={score.reason_codes} />

      <WhatIfCoach features={borrower.features} />
        </div>
      </div>
    </div>
  );
}
