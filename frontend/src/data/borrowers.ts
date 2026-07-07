import type { Features } from "../api/types";

export interface Borrower {
  id: string;
  name: string;
  location: string;
  gstin: string;
  sector: string;
  features: Features;
}

export const BORROWERS: Borrower[] = [
  {
    id: "sharma",
    name: "Sharma Kirana Store",
    location: "Indore, MP",
    gstin: "23ABCDE1234F1Z5",
    sector: "Retail — General store",
    features: {
      gst_avg_monthly_turnover: 380000,
      gst_turnover_growth: 0.12,
      gst_turnover_volatility: 0.12,
      gst_filing_punctuality: 12,
      gst_turnover_decline_3m: -0.05,
      bank_avg_monthly_credits: 361000,
      bank_balance_dip_count: 1,
      bank_bounce_count: 0,
      bank_cash_buffer_days: 34,
      bank_credit_debit_ratio: 1.12,
      xf_gst_bank_gap: 0.05,
      epfo_headcount: 8,
      epfo_headcount_trend: 0.08,
    },
  },
  {
    id: "deepak",
    name: "Deepak Hardware",
    location: "Nagpur, MH",
    gstin: "27FGHIJ5678K2Z9",
    sector: "Trading — Hardware",
    features: {
      gst_avg_monthly_turnover: 385405,
      gst_turnover_growth: 0.0448,
      gst_turnover_volatility: 0.2291,
      gst_filing_punctuality: 4,
      gst_turnover_decline_3m: 0.0927,
      bank_avg_monthly_credits: 308576,
      bank_balance_dip_count: 3,
      bank_bounce_count: 2,
      bank_cash_buffer_days: 33.7,
      bank_credit_debit_ratio: 0.9085,
      xf_gst_bank_gap: 0.1993,
      epfo_headcount: 4,
      epfo_headcount_trend: 0.0302,
    },
  },
  {
    id: "verma",
    name: "Verma Traders",
    location: "Kanpur, UP",
    gstin: "09KLMNO9012P3Z1",
    sector: "Wholesale — Textiles",
    features: {
      gst_avg_monthly_turnover: 900000,
      gst_turnover_growth: -0.05,
      gst_turnover_volatility: 0.35,
      gst_filing_punctuality: 7,
      gst_turnover_decline_3m: 0.1,
      bank_avg_monthly_credits: 558000,
      bank_balance_dip_count: 4,
      bank_bounce_count: 2,
      bank_cash_buffer_days: 10,
      bank_credit_debit_ratio: 0.9,
      xf_gst_bank_gap: 0.38,
      epfo_headcount: 6,
      epfo_headcount_trend: -0.05,
    },
  },
];
