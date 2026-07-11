import { useEffect, useState } from "react";

import { getHealth, requestConsent, score } from "./api/client";
import type { ConsentResponse, HealthResponse, ScoreResponse } from "./api/types";
import { BorrowerCard } from "./components/BorrowerCard";
import { ConsentPanel } from "./components/ConsentPanel";
import { Header } from "./components/Header";
import { LenderConsole } from "./components/LenderConsole";
import { MonitoringPanel } from "./components/MonitoringPanel";
import { NewMsmeForm } from "./components/NewMsmeForm";
import { RoiCalculator } from "./components/RoiCalculator";
import type { Borrower } from "./data/borrowers";
import { BORROWERS } from "./data/borrowers";

type View = "console" | "roi";

function initialBorrower(): Borrower {
  if (typeof window === "undefined") return BORROWERS[0];
  const focus = new URLSearchParams(window.location.search).get("focus");
  return BORROWERS.find((borrower) => borrower.id === focus) ?? BORROWERS[0];
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [view, setView] = useState<View>("console");
  const [extra, setExtra] = useState<Borrower[]>([]);
  const [selected, setSelected] = useState<Borrower>(initialBorrower);
  const [adding, setAdding] = useState(false);
  const [card, setCard] = useState<ScoreResponse | null>(null);
  const [consent, setConsent] = useState<ConsentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const borrowers = [...BORROWERS, ...extra];

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (adding) return;
    setCard(null);
    setError(null);
    score(selected.features)
      .then(setCard)
      .catch((cause) => setError(cause instanceof Error ? cause.message : "Scoring failed"));
  }, [selected, adding]);

  const approveConsent = () => {
    requestConsent("103").then(setConsent).catch(() => setConsent(null));
  };

  const addMsme = (borrower: Borrower) => {
    setExtra((prev) => [...prev, borrower]);
    setSelected(borrower);
    setConsent(null);
    setAdding(false);
  };

  return (
    <div className="app">
      <Header health={health} />

      <div className="viewnav">
        <button
          className={`viewtab ${view === "console" ? "active" : ""}`}
          onClick={() => setView("console")}
        >
          Lender console
        </button>
        <button
          className={`viewtab ${view === "roi" ? "active" : ""}`}
          onClick={() => setView("roi")}
        >
          Source-ablation ROI
        </button>
      </div>

      {view === "roi" && <RoiCalculator />}

      {view === "console" && (
        <>
      <div className="tabs">
        {borrowers.map((borrower) => (
          <button
            key={borrower.id}
            className={`tab ${!adding && borrower.id === selected.id ? "active" : ""}`}
            onClick={() => {
              setSelected(borrower);
              setConsent(null);
              setAdding(false);
            }}
          >
            {borrower.name}
          </button>
        ))}
        <button className={`tab ${adding ? "active" : ""}`} onClick={() => setAdding(true)}>
          + New MSME
        </button>
      </div>

      {adding ? (
        <NewMsmeForm onAdd={addMsme} onCancel={() => setAdding(false)} />
      ) : (
        <>
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
              <div className="stack">
                <MonitoringPanel features={selected.features} />
                <ConsentPanel consent={consent} />
              </div>
            </div>
          )}
        </>
      )}
        </>
      )}

      <div className="footer">
        Built on Account Aggregator + GST rails · ULI/OCEN-ready · DPDP-compliant · Onboards New-to-Credit MSMEs
      </div>
    </div>
  );
}
