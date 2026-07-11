import { useRef, useState } from "react";

import { ApiError, extractDocument } from "../api/client";
import type { ExtractResponse, Features } from "../api/types";
import type { Borrower } from "../data/borrowers";

interface Props {
  onAdd: (borrower: Borrower) => void;
  onCancel: () => void;
}

interface Field {
  key: keyof Features;
  label: string;
  value: number;
  step?: number;
}

const FIELDS: Field[] = [
  { key: "gst_avg_monthly_turnover", label: "GST turnover / month (₹)", value: 500000, step: 10000 },
  { key: "gst_turnover_growth", label: "Turnover growth", value: 0.05, step: 0.01 },
  { key: "gst_turnover_volatility", label: "Turnover volatility", value: 0.18, step: 0.01 },
  { key: "gst_filing_punctuality", label: "GST filings on time (0–12)", value: 10 },
  { key: "gst_turnover_decline_3m", label: "3-month decline", value: 0, step: 0.01 },
  { key: "bank_avg_monthly_credits", label: "Bank credits / month (₹)", value: 460000, step: 10000 },
  { key: "bank_balance_dip_count", label: "Month-end balance dips", value: 2 },
  { key: "bank_bounce_count", label: "Cheque bounces", value: 0 },
  { key: "bank_cash_buffer_days", label: "Cash buffer (days)", value: 28 },
  { key: "bank_credit_debit_ratio", label: "Credit/debit ratio", value: 1.05, step: 0.01 },
  { key: "xf_gst_bank_gap", label: "GST-vs-bank gap", value: 0.08, step: 0.01 },
  { key: "epfo_headcount", label: "EPFO headcount", value: 6 },
  { key: "epfo_headcount_trend", label: "Headcount trend", value: 0.03, step: 0.01 },
];

const SAMPLE_PATH = `${import.meta.env.BASE_URL}samples/gupta-provisions-sample.pdf`;

type Status = { tone: "info" | "warn" | "error"; text: string } | null;

export function NewMsmeForm({ onAdd, onCancel }: Props) {
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [gstin, setGstin] = useState("");
  const [fields, setFields] = useState<Field[]>(FIELDS);
  const [status, setStatus] = useState<Status>(null);
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const update = (key: keyof Features, raw: string) =>
    setFields((prev) => prev.map((f) => (f.key === key ? { ...f, value: Number(raw) } : f)));

  const applyExtraction = (result: ExtractResponse) => {
    const found = result.fields;
    if (typeof found.name === "string") setName(found.name);
    if (typeof found.location === "string") setLocation(found.location);
    if (typeof found.gstin === "string") setGstin(found.gstin);
    setFields((prev) =>
      prev.map((f) => {
        const value = found[f.key];
        return typeof value === "number" ? { ...f, value } : f;
      }),
    );

    if (result.source === "unavailable") {
      setStatus({
        tone: "warn",
        text: result.message ?? "OCR service is offline. Try the sample document.",
      });
      return;
    }
    const count = Object.keys(found).length;
    if (result.source === "ocr") {
      setStatus({
        tone: "info",
        text: `Extracted ${count} field${count === 1 ? "" : "s"} from the document — review before computing.`,
      });
    } else {
      setStatus({
        tone: "warn",
        text: "OCR offline — loaded a sample firm. Best-effort figures, review before computing.",
      });
    }
  };

  const runExtraction = async (file: File, demo: boolean) => {
    setBusy(true);
    setStatus({ tone: "info", text: `Reading ${file.name}…` });
    try {
      applyExtraction(await extractDocument(file, demo));
    } catch (cause) {
      const message =
        cause instanceof ApiError ? cause.message : "Could not reach the extraction service.";
      setStatus({ tone: "error", text: message });
    } finally {
      setBusy(false);
    }
  };

  const onFilePicked = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void runExtraction(file, false);
  };

  const useSampleDocument = async () => {
    setBusy(true);
    setStatus({ tone: "info", text: "Loading a sample document…" });
    try {
      const response = await fetch(SAMPLE_PATH);
      if (!response.ok) throw new Error("Sample document not found");
      const blob = await response.blob();
      const file = new File([blob], "gupta-provisions-sample.pdf", { type: "application/pdf" });
      await runExtraction(file, true);
    } catch {
      setStatus({
        tone: "error",
        text: "Could not load the sample document.",
      });
      setBusy(false);
    }
  };

  const submit = () => {
    const features = Object.fromEntries(fields.map((f) => [f.key, f.value])) as Features;
    onAdd({
      id: `new-${Date.now()}`,
      name: name || "New MSME",
      location: location || "Manual entry",
      gstin: gstin || "—",
      sector: "Manual entry",
      features,
    });
  };

  return (
    <div className="panel add-form">
      <h2>Onboard a new MSME</h2>
      <div className="sub">Enter the firm's consented figures to compute its Parakh score.</div>

      <div className="upload-row">
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf,image/*"
          className="upload-input"
          onChange={onFilePicked}
        />
        <button
          type="button"
          className="action secondary upload-btn"
          disabled={busy}
          onClick={() => fileInput.current?.click()}
        >
          Upload document (PDF)
        </button>
        <button type="button" className="linklike" disabled={busy} onClick={useSampleDocument}>
          Use a sample document
        </button>
      </div>
      <div className="upload-note">
        Best-effort auto-fill from the uploaded document. The officer reviews every field before
        computing the score.
      </div>
      {status && <div className={`upload-status ${status.tone}`}>{status.text}</div>}

      <div className="add-grid">
        <label>
          Business name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Gupta Provisions" />
        </label>
        <label>
          Location
          <input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Kanpur, UP" />
        </label>
        <label>
          GSTIN
          <input value={gstin} onChange={(e) => setGstin(e.target.value)} placeholder="09ABCDE1234F1Z5" />
        </label>
        {fields.map((f) => (
          <label key={String(f.key)}>
            {f.label}
            <input type="number" value={f.value} step={f.step ?? 1} onChange={(e) => update(f.key, e.target.value)} />
          </label>
        ))}
      </div>
      <div className="add-actions">
        <button className="action" onClick={submit}>
          Compute Parakh score
        </button>
        <button className="action secondary" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
