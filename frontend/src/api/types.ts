export type FeatureKey =
  | "gst_avg_monthly_turnover"
  | "gst_turnover_growth"
  | "gst_turnover_volatility"
  | "gst_filing_punctuality"
  | "gst_turnover_decline_3m"
  | "bank_avg_monthly_credits"
  | "bank_balance_dip_count"
  | "bank_bounce_count"
  | "bank_cash_buffer_days"
  | "bank_credit_debit_ratio"
  | "xf_gst_bank_gap"
  | "epfo_headcount"
  | "epfo_headcount_trend";

export type Features = Record<FeatureKey, number>;

export type DimensionKey =
  | "Cash Flow"
  | "GST Compliance"
  | "Banking Discipline"
  | "Growth"
  | "Stability"
  | "Leverage";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ConsentResponse {
  consent_start: string;
  consent_expiry: string;
  purpose_code: string;
  purpose_text: string;
  fi_types: string[];
  data_life_unit: string;
  data_life_value: number;
  fetch_type: string;
  frequency_unit: string;
  frequency_value: number;
}

export interface ReasonCode {
  feature: FeatureKey;
  english: string;
  hindi: string;
  points: number;
  supports_score: boolean;
}

export interface ScoreResponse {
  score: number;
  grade: string;
  pd: number;
  dimensions: Record<string, number>;
  prequalified_limit: number;
  divergence_gap: number;
  divergence_flag: boolean;
  reason_codes: ReasonCode[];
}

export interface WhatIfSide {
  score: number;
  grade: string;
  limit: number;
}

export interface WhatIfResponse {
  before: WhatIfSide;
  after: WhatIfSide;
}

export interface MemoResponse {
  borrower: string;
  body: string;
  status: string;
  generated_by: string;
}
