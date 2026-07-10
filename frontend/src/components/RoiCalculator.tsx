import { useMemo, useState } from "react";

import { DEFAULT_INPUTS, computeRoi } from "../data/roi";
import type { RoiInputs } from "../data/roi";
import { inr, percent } from "../format";

function signedInr(value: number): string {
  const magnitude = inr(Math.abs(value));
  return value < 0 ? `−${magnitude}` : magnitude;
}

export function RoiCalculator() {
  const [inputs, setInputs] = useState<RoiInputs>(DEFAULT_INPUTS);
  const result = useMemo(() => computeRoi(inputs), [inputs]);

  const set = <K extends keyof RoiInputs>(key: K, value: RoiInputs[K]) =>
    setInputs((prev) => ({ ...prev, [key]: value }));

  const setPull = (rail: "aa" | "epfo", value: number) =>
    setInputs((prev) => ({ ...prev, pullCost: { ...prev.pullCost, [rail]: value } }));

  return (
    <div className="panel roi">
      <h2>Source-ablation ROI · CFO view</h2>
      <div className="sub">
        Marginal rupee value of each consented data rail, from the measured AUC lift
      </div>

      <div className="roi-inputs">
        <label>
          MSME book size (₹ Cr)
          <input
            type="number"
            min={0}
            step={1000}
            value={inputs.bookSizeCr}
            onChange={(event) => set("bookSizeCr", Number(event.target.value))}
          />
        </label>

        <label>
          Annual slippage — {percent(inputs.slippagePct / 100, 1)}
          <input
            type="range"
            min={0.5}
            max={6}
            step={0.1}
            value={inputs.slippagePct}
            onChange={(event) => set("slippagePct", Number(event.target.value))}
          />
        </label>

        <label>
          Accounts re-scored / year
          <input
            type="number"
            min={0}
            step={10000}
            value={inputs.accountsScoredPerYear}
            onChange={(event) => set("accountsScoredPerYear", Number(event.target.value))}
          />
        </label>

        <label>
          AA pull cost (₹/pull)
          <input
            type="number"
            min={0}
            step={1}
            value={inputs.pullCost.aa}
            onChange={(event) => setPull("aa", Number(event.target.value))}
          />
        </label>

        <label>
          EPFO pull cost (₹/pull)
          <input
            type="number"
            min={0}
            step={1}
            value={inputs.pullCost.epfo}
            onChange={(event) => setPull("epfo", Number(event.target.value))}
          />
        </label>

        <label>
          Capture factor — {percent(inputs.captureFactor, 0)}
          <input
            type="range"
            min={0.05}
            max={0.8}
            step={0.05}
            value={inputs.captureFactor}
            onChange={(event) => set("captureFactor", Number(event.target.value))}
          />
        </label>
      </div>

      <div className="roi-slippage">
        Annual slippage base <b>{inr(result.slippageRupees)}</b>
        <span> · GST-only baseline AUC 0.714</span>
      </div>

      <table className="roi-table">
        <thead>
          <tr>
            <th>Consented rail</th>
            <th>AUC lift</th>
            <th>Provisioning identified early</th>
            <th>Pull spend</th>
            <th>Net value</th>
          </tr>
        </thead>
        <tbody>
          {result.rails.map((rail) => (
            <tr key={rail.key}>
              <td>{rail.label}</td>
              <td className="num">
                {rail.aucBefore.toFixed(3)} → {rail.aucWith.toFixed(3)}
                <span className="lift">+{rail.aucLift.toFixed(3)}</span>
              </td>
              <td className="num">{inr(rail.provisioningIdentifiedEarly)}</td>
              <td className="num">{inr(rail.pullSpend)}</td>
              <td className={`num net ${rail.netValue >= 0 ? "up" : "down"}`}>
                {signedInr(rail.netValue)}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>AA + EPFO combined</td>
            <td className="num">0.714 → 0.752</td>
            <td className="num">{inr(result.totalProvisioningIdentifiedEarly)}</td>
            <td className="num">{inr(result.totalPullSpend)}</td>
            <td className={`num net ${result.totalNetValue >= 0 ? "up" : "down"}`}>
              {signedInr(result.totalNetValue)}
            </td>
          </tr>
        </tfoot>
      </table>

      <p className="note">
        AUC lift is <b>measured</b> on the held-out test fold (
        <code>src/parakh/eval/ablation.py</code>). Book size, slippage, pull costs,
        accounts re-scored and the capture factor are <b>illustrative</b> inputs the
        CFO controls. Provisioning identified early = annual slippage × marginal Gini
        lift (2 × AUC lift) × capture factor; net value subtracts the rail's annual
        pull spend.
      </p>
    </div>
  );
}
