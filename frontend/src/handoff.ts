// Lifecycle handoff from PARAKH (origination) to TRINETRA (monitoring). The
// same MSME carries a different id in each track; this map is the single source
// of that correspondence.
const PARAKH_TO_TRINETRA: Record<string, string> = {
  sharma: "sharma-kirana",
};

// Same-origin default behind the shared ALB (TRINETRA served at /trinetra/).
// Overridable for local or split-host setups via VITE_TRINETRA_BASE.
const TRINETRA_BASE = (import.meta.env.VITE_TRINETRA_BASE ?? "/trinetra/").replace(
  /\/?$/,
  "/",
);

export function monitoringTargetId(borrowerId: string): string | null {
  return PARAKH_TO_TRINETRA[borrowerId] ?? null;
}

export function monitoringUrl(borrowerId: string): string | null {
  const target = monitoringTargetId(borrowerId);
  return target ? `${TRINETRA_BASE}?focus=${encodeURIComponent(target)}` : null;
}
