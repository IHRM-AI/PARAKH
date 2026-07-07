import { useEffect, useState } from "react";

import { monitor } from "../api/client";
import type { Features, MonitorResponse } from "../api/types";

const W = 300;
const H = 120;
const PAD = 22;

export function MonitoringPanel({ features }: { features: Features }) {
  const [data, setData] = useState<MonitorResponse | null>(null);

  useEffect(() => {
    let active = true;
    monitor(features)
      .then((response) => active && setData(response))
      .catch(() => active && setData(null));
    return () => {
      active = false;
    };
  }, [features]);

  return (
    <div className="panel">
      <h2>Portfolio monitoring</h2>
      <div className="sub">12-month score surveillance · consent 104</div>
      {!data ? (
        <div className="state">Loading monitoring…</div>
      ) : (
        <MonitorChart data={data} />
      )}
    </div>
  );
}

function MonitorChart({ data }: { data: MonitorResponse }) {
  const points = data.scores.map((value, index) => ({
    x: PAD + (index / (data.scores.length - 1)) * (W - 2 * PAD),
    y: H - PAD - (value / 100) * (H - 2 * PAD),
  }));
  const line = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  const stroke = data.deteriorating ? "#c1524e" : "#00836c";
  const flag = data.flag_month ? points[data.flag_month - 1] : null;

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
        <path d={line} fill="none" stroke={stroke} strokeWidth={2.5} />
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={2} fill={stroke} />
        ))}
        {flag && (
          <>
            <line x1={flag.x} y1={PAD - 8} x2={flag.x} y2={H - PAD} stroke="#f59e0b" strokeWidth={1} strokeDasharray="3 3" />
            <circle cx={flag.x} cy={flag.y} r={4} fill="#f59e0b" />
          </>
        )}
      </svg>
      <div className="summary-row">
        <span>Score trend</span>
        <b>
          {data.scores[0]} → {data.scores[data.scores.length - 1]}
        </b>
      </div>
      {data.flag_month ? (
        <div className="tripwire alert">
          Flagged in month {data.flag_month} — score fell below the watch threshold of {data.watch_threshold},
          ahead of any missed payment.
        </div>
      ) : (
        <div className="tripwire clear">Score stable across the last 12 months.</div>
      )}
    </>
  );
}
