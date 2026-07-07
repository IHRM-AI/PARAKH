import type {
  ConsentResponse,
  Features,
  HealthResponse,
  MemoResponse,
  ScoreResponse,
  WhatIfResponse,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8092").replace(
  /\/$/,
  "",
);

export type PurposeCode = "103" | "104";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readError(response: Response): Promise<string> {
  const raw = await response.text().catch(() => "");
  let detail = raw;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (typeof parsed.detail === "string") detail = parsed.detail;
  } catch {
    // Non-JSON error body; fall back to the raw text.
  }
  if (response.status === 404) {
    return "Endpoint not found on the configured backend";
  }
  if (response.status === 503) {
    return detail || "Model artifact not loaded. Train the model first.";
  }
  return detail || response.statusText || `HTTP ${response.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    throw new ApiError(
      cause instanceof Error ? cause.message : "Network request failed",
      0,
    );
  }

  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function requestConsent(
  purposeCode: PurposeCode,
): Promise<ConsentResponse> {
  return request<ConsentResponse>("/consent", {
    method: "POST",
    body: JSON.stringify({ purpose_code: purposeCode }),
  });
}

export function score(features: Features): Promise<ScoreResponse> {
  return request<ScoreResponse>("/score", {
    method: "POST",
    body: JSON.stringify({ features }),
  });
}

export function whatIf(
  features: Features,
  adjustments: Partial<Features>,
): Promise<WhatIfResponse> {
  return request<WhatIfResponse>("/whatif", {
    method: "POST",
    body: JSON.stringify({ features, adjustments }),
  });
}

export function draftMemo(
  borrower: string,
  features: Features,
): Promise<MemoResponse> {
  return request<MemoResponse>("/memo", {
    method: "POST",
    body: JSON.stringify({ borrower, features }),
  });
}
