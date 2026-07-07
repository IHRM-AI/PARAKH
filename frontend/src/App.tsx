import { useEffect, useState } from "react";

import { getHealth, requestConsent, score } from "./api/client";
import type { ConsentResponse, HealthResponse, ScoreResponse } from "./api/types";
import { BorrowerCard } from "./components/BorrowerCard";
import { ConsentPanel } from "./components/ConsentPanel";
import { Header } from "./components/Header";
import { LenderConsole } from "./components/LenderConsole";
import { BORROWERS } from "./data/borrowers";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selected, setSelected] = useState(BORROWERS[0]);
  const [card, setCard] = useState<ScoreResponse | null>(null);
  const [consent, setConsent] = useState<ConsentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    setCard(null);
    setError(null);
    score(selected.features)
      .then(setCard)
      .catch((cause) => setError(cause instanceof Error ? cause.message : "Scoring failed"));
  }, [selected]);

  const approveConsent = () => {
    requestConsent("103").then(setConsent).catch(() => setConsent(null));
  };

  return (
    <div className="app">
      <Header health={health} />

      <div className="tabs">
        {BORROWERS.map((borrower) => (
          <button
            key={borrower.id}
            className={`tab ${borrower.id === selected.id ? "active" : ""}`}
            onClick={() => {
              setSelected(borrower);
              setConsent(null);
            }}
          >
            {borrower.name}
          </button>
        ))}
      </div>

      {error && <div className="state error">{error}</div>}
      {!error && !card && <div className="state">Scoring {selected.name}…</div>}

      {card && (
        <div className="grid">
          <LenderConsole borrower={selected} score={card} />
          <BorrowerCard
            borrower={selected}
            score={card}
            consent={consent}
            onConsent={approveConsent}
          />
          <ConsentPanel consent={consent} />
        </div>
      )}

      <div className="footer">
        Built on Account Aggregator + GST rails · ULI/OCEN-ready · DPDP-compliant · Onboards New-to-Credit MSMEs
      </div>
    </div>
  );
}
