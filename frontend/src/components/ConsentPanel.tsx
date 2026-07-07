import type { ConsentResponse } from "../api/types";

export function ConsentPanel({ consent }: { consent: ConsentResponse | null }) {
  return (
    <div className="panel">
      <h2>Account Aggregator consent</h2>
      <div className="sub">ReBIT consent artefact</div>

      {consent ? (
        <div className="artefact">
          {"{"}
          <br />
          &nbsp;&nbsp;<span className="k">"purpose"</span>: {"{"} "code": "{consent.purpose_code}", "text": "
          {consent.purpose_text}" {"}"},
          <br />
          &nbsp;&nbsp;<span className="k">"fiTypes"</span>: [{consent.fi_types.map((t) => `"${t}"`).join(", ")}],
          <br />
          &nbsp;&nbsp;<span className="k">"consentStart"</span>: "{consent.consent_start}",
          <br />
          &nbsp;&nbsp;<span className="k">"consentExpiry"</span>: "{consent.consent_expiry}",
          <br />
          &nbsp;&nbsp;<span className="k">"DataLife"</span>: {"{"} "unit": "{consent.data_life_unit}", "value":{" "}
          {consent.data_life_value} {"}"},
          <br />
          &nbsp;&nbsp;<span className="k">"fetchType"</span>: "{consent.fetch_type}"
          <br />
          {"}"}
        </div>
      ) : (
        <div className="state">Approve consent on the borrower card to fetch financial data.</div>
      )}

      <p className="note">
        Note: UPI data arrives within DEPOSIT-account narrations — it is not a separate Account Aggregator FI
        type. Every fetch and decision is written to the DPDP audit ledger.
      </p>
    </div>
  );
}
