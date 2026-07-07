import { useState } from "react";

import { draftMemo } from "../api/client";
import type { Borrower } from "../data/borrowers";
import type { MemoResponse, ScoreResponse } from "../api/types";
import { inr, percent } from "../format";

interface Props {
  borrower: Borrower;
  score: ScoreResponse;
}

export function LenderConsole({ borrower, score }: Props) {
  const [memo, setMemo] = useState<MemoResponse | null>(null);
  const [approved, setApproved] = useState(false);
  const [drafting, setDrafting] = useState(false);

  const draft = () => {
    setDrafting(true);
    setApproved(false);
    draftMemo(borrower.name, borrower.features)
      .then(setMemo)
      .catch(() => setMemo(null))
      .finally(() => setDrafting(false));
  };

  return (
    <div className="panel">
      <h2>Lender console · i-MSME Express</h2>
      <div className="sub">{borrower.name}</div>

      <div className="summary-row">
        <span>Health score</span>
        <b>
          {score.score}/100 · {score.grade}
        </b>
      </div>
      <div className="summary-row">
        <span>Avg monthly credits</span>
        <b>{inr(borrower.features.bank_avg_monthly_credits)}</b>
      </div>
      <div className="summary-row">
        <span>GST turnover (monthly)</span>
        <b>{inr(borrower.features.gst_avg_monthly_turnover)}</b>
      </div>
      <div className="summary-row">
        <span>GST-vs-bank divergence</span>
        <b>{percent(score.divergence_gap)}</b>
      </div>
      <div className="summary-row">
        <span>Pre-qualified limit</span>
        <b>{inr(score.prequalified_limit)}</b>
      </div>

      <div className={`tripwire ${score.divergence_flag ? "alert" : "clear"}`}>
        {score.divergence_flag
          ? `Divergence alert — declared GST turnover exceeds observed bank credits by ${percent(score.divergence_gap)}. Refer to manual review before sanction.`
          : "GST and banking turnover reconcile within tolerance."}
      </div>

      <button className="action" onClick={draft} disabled={drafting}>
        {drafting ? "Drafting…" : "Draft credit memo"}
      </button>

      {memo && (
        <div className="memo">
          <pre>{memo.body}</pre>
          <div className="meta">
            {memo.status} · {memo.generated_by}
          </div>
          <button
            className="action secondary"
            onClick={() => setApproved(true)}
            disabled={approved}
          >
            {approved ? "Approved by officer" : "Approve (human-in-loop)"}
          </button>
        </div>
      )}
    </div>
  );
}
