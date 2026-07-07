import { useEffect, useState } from "react";

import { whatIf } from "../api/client";
import type { Features, WhatIfResponse } from "../api/types";
import { inr } from "../format";

export function WhatIfCoach({ features }: { features: Features }) {
  const [buffer, setBuffer] = useState<number>(Math.round(features.bank_cash_buffer_days));
  const [result, setResult] = useState<WhatIfResponse | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      whatIf(features, { bank_cash_buffer_days: buffer })
        .then(setResult)
        .catch(() => setResult(null));
    }, 250);
    return () => clearTimeout(timer);
  }, [buffer, features]);

  return (
    <div className="whatif">
      <div className="sub">What-if coach · स्कोर बढ़ाएँ</div>
      <label className="hi">Cash buffer: {buffer} days</label>
      <input
        type="range"
        min={5}
        max={90}
        value={buffer}
        onChange={(event) => setBuffer(Number(event.target.value))}
      />
      {result && (
        <div className="delta">
          <span>
            Score {result.before.score} → <b>{result.after.score}</b>
          </span>
          <span>
            Eligible {inr(result.before.limit)} → <b>{inr(result.after.limit)}</b>
          </span>
        </div>
      )}
    </div>
  );
}
