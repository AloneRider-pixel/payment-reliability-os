export type Dashboard = {
  business_id: string;
  invoice_count: number;
  outstanding_receivables: number;
  cash_at_risk: number;
  predicted_invoice_count: number;
  high_risk_invoice_count: number;
};

export type CustomerRisk = {
  customer_id: number;
  customer: string;
  score: number;
  grade: string;
  late_probability: number;
  expected_delay_days: number;
  reasons: string[];
  evidence_count: number;
  confidence: string;
  outstanding_amount: number;
};

export type InvoiceRisk = {
  invoice_id: number;
  invoice_number: string;
  customer: string;
  amount: number;
  due_date: string;
  late_probability: number;
  expected_delay_days: number;
  expected_payment_date: string;
  cash_at_risk: number;
  risk_band: "HIGH" | "MEDIUM" | "LOW";
  reasons: string[];
};

export type EvaluationSummary = {
  business_id: string;
  evaluated_predictions: number;
  date_mae_days: number | null;
  mean_brier_error: number | null;
  late_classification_accuracy: number | null;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(API_BASE + path);
  if (!response.ok) throw new Error("API " + response.status);
  return response.json() as Promise<T>;
}

export const api = {
  dashboard: (businessId: string) => getJson<Dashboard>("/dashboard?business_id=" + encodeURIComponent(businessId)),
  customers: (businessId: string) => getJson<CustomerRisk[]>("/risk/customers?business_id=" + encodeURIComponent(businessId)),
  invoices: (businessId: string) => getJson<InvoiceRisk[]>("/risk/invoices?business_id=" + encodeURIComponent(businessId)),
  evaluations: (businessId: string) => getJson<EvaluationSummary>("/evaluations/summary?business_id=" + encodeURIComponent(businessId)),
};