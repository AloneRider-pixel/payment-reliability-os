export type ScoreFeatures = {
  on_time_rate: number;
  late_payment_rate: number;
  average_delay_days: number;
  median_delay_days: number;
  p90_delay_days: number;
  recent_average_delay_days: number;
  recent_trend_days: number;
  payment_consistency: number;
  invoice_amount_mean: number;
  current_outstanding_exposure: number;
  evidence_count: number;
};

export type ScoreFactor = {
  name: string;
  value: number;
  weight: number;
  contribution: number;
  direction: string;
};

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
  model_version: string;
  features: ScoreFeatures;
  factors: ScoreFactor[];
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

export type ModelStatus = {
  business_id: string;
  active: boolean;
  model_version: string;
  trained_at?: string;
  train_count?: number;
  test_count?: number;
  metrics?: {
    candidate: {
      date_mae_days: number;
      mean_brier_error: number;
      late_classification_accuracy: number;
    };
    baseline: {
      date_mae_days: number;
      mean_brier_error: number;
      late_classification_accuracy: number;
    };
    delta: {
      date_mae_days: number;
      mean_brier_error: number;
    };
  };
};

export type BacktestResult = {
  business_id: string;
  model_version: string;
  min_history: number;
  invoices_seen: number;
  backtested_invoices: number;
  skipped_cold_start: number;
  date_mae_days: number | null;
  mean_brier_error: number | null;
  late_classification_accuracy: number | null;
  observed_late_rate: number | null;
  mean_predicted_late_rate: number | null;
  mean_absolute_calibration_gap: number | null;
  calibration_bins: Array<{
    lower_bound: number;
    upper_bound: number;
    count: number;
    mean_predicted_late_rate: number;
    observed_late_rate: number;
    absolute_calibration_gap: number;
  }>;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(API_BASE + path);
  if (!response.ok) throw new Error("API " + response.status);
  return response.json() as Promise<T>;
}

export const api = {
  dashboard: (businessId: string) =>
    getJson<Dashboard>("/dashboard?business_id=" + encodeURIComponent(businessId)),
  customers: (businessId: string) =>
    getJson<CustomerRisk[]>(
      "/risk/customers?business_id=" + encodeURIComponent(businessId),
    ),
  invoices: (businessId: string) =>
    getJson<InvoiceRisk[]>(
      "/risk/invoices?business_id=" + encodeURIComponent(businessId),
    ),
  evaluations: (businessId: string) =>
    getJson<EvaluationSummary>(
      "/evaluations/summary?business_id=" + encodeURIComponent(businessId),
    ),
  modelStatus: (businessId: string) =>
    getJson<ModelStatus>(
      "/models/status?business_id=" + encodeURIComponent(businessId),
    ),
  trainModel: (businessId: string, minHistory = 3) =>
    postJson<{
      promotion_status: "promoted" | "candidate_only";
      train_count: number;
      test_count: number;
      model_version: string;
      candidate_metrics: ModelStatus["metrics"]["candidate"];
      baseline_metrics: ModelStatus["metrics"]["baseline"];
      delta: {
        date_mae_days: number;
        mean_brier_error: number;
      };
    }>(
      "/models/train?business_id=" +
        encodeURIComponent(businessId) +
        "&min_history=" +
        minHistory,
    ),
  backtest: (businessId: string, minHistory = 3) =>
    postJson<BacktestResult>(
      "/evaluations/backtest?business_id=" +
        encodeURIComponent(businessId) +
        "&min_history=" +
        minHistory,
    ),
};

async function postJson<T>(path: string): Promise<T> {
  const response = await fetch(API_BASE + path, { method: "POST" });
  if (!response.ok) throw new Error("API " + response.status);
  return response.json() as Promise<T>;
}

async function uploadFile(path: string, businessId: string, file: File) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(
    API_BASE + path + "?business_id=" + encodeURIComponent(businessId),
    {
      method: "POST",
      body: form,
    },
  );
  if (!response.ok) throw new Error("Upload API " + response.status);
  return response.json();
}

export async function uploadInvoices(businessId: string, file: File) {
  return uploadFile("/imports/invoices", businessId, file);
}

export async function uploadPayments(businessId: string, file: File) {
  return uploadFile("/imports/payments", businessId, file);
}

export async function analyzeAll(businessId: string) {
  return getJson<{ predictions_created: number; invoices_seen: number }>(
    "/risk/analyze-all?business_id=" + encodeURIComponent(businessId),
  );
}
