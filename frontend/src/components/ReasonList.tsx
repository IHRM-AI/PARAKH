import type { ReasonCode } from "../api/types";

export function ReasonList({ codes }: { codes: ReasonCode[] }) {
  return (
    <div className="reasons">
      {codes.map((code) => (
        <div className="reason" key={code.feature}>
          <div>
            <div className="en">{code.english}</div>
            <div className="hi">{code.hindi}</div>
          </div>
          <span className={`points ${code.supports_score ? "up" : "down"}`}>
            {code.supports_score ? "+" : "−"}
            {code.points}
          </span>
        </div>
      ))}
    </div>
  );
}
