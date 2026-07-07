import type { HealthResponse } from "../api/types";

export function Header({ health }: { health: HealthResponse | null }) {
  const online = health?.model_loaded === true;
  const state = health == null ? "down" : online ? "ok" : "down";
  const label = health == null ? "Backend offline" : online ? "Model loaded" : "Model not loaded";

  return (
    <header className="header">
      <div className="brand">
        <h1>
          PARA<span className="kh">KH</span>
        </h1>
        <span>Financial Health Score</span>
      </div>
      <div className="header-right">
        <span className="health-dot">
          <span className={`dot ${state}`} />
          {label}
        </span>
        <span>Powered by IDBI Bank · IDBI Innovate 2026</span>
      </div>
    </header>
  );
}
