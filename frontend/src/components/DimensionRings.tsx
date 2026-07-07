interface Props {
  dimensions: Record<string, number>;
}

const R = 22;
const CIRC = 2 * Math.PI * R;

function Ring({ label, value }: { label: string; value: number }) {
  const offset = CIRC * (1 - Math.max(0, Math.min(100, value)) / 100);
  const color = value >= 60 ? "#00836c" : value >= 40 ? "#f59e0b" : "#c1524e";
  return (
    <div className="ring">
      <svg viewBox="0 0 56 56" width="56" height="56">
        <circle cx="28" cy="28" r={R} fill="none" stroke="#e9e5db" strokeWidth="6" />
        <circle
          cx="28"
          cy="28"
          r={R}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
          transform="rotate(-90 28 28)"
        />
        <text x="28" y="33" textAnchor="middle" fontSize="16" fontWeight="700" fill="#12232e">
          {value}
        </text>
      </svg>
      <div className="label">{label}</div>
    </div>
  );
}

export function DimensionRings({ dimensions }: Props) {
  return (
    <div className="rings">
      {Object.entries(dimensions).map(([label, value]) => (
        <Ring key={label} label={label} value={value} />
      ))}
    </div>
  );
}
