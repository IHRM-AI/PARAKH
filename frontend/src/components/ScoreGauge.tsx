interface Props {
  score: number;
  grade: string;
}

const CX = 110;
const CY = 110;
const R = 92;

function polar(angle: number) {
  const a = ((angle - 90) * Math.PI) / 180;
  return { x: CX + R * Math.cos(a), y: CY + R * Math.sin(a) };
}

function arc(from: number, to: number) {
  const p0 = polar(from);
  const p1 = polar(to);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  return `M ${p0.x} ${p0.y} A ${R} ${R} 0 ${large} 1 ${p1.x} ${p1.y}`;
}

export function ScoreGauge({ score, grade }: Props) {
  const valueAngle = -90 + (Math.max(0, Math.min(100, score)) / 100) * 180;
  const stroke = score >= 68 ? "#00836c" : score >= 45 ? "#f59e0b" : "#c1524e";

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 220 130" width="220" height="130">
        <path d={arc(-90, 90)} fill="none" stroke="#e9e5db" strokeWidth={16} strokeLinecap="round" />
        <path d={arc(-90, valueAngle)} fill="none" stroke={stroke} strokeWidth={16} strokeLinecap="round" />
      </svg>
      <div className="gauge-score">
        {score}
        <small>/100</small>
      </div>
      <div className="pill-row">
        <span className="pill grade">Grade {grade}</span>
      </div>
    </div>
  );
}
