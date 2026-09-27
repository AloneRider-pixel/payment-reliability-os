import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import type {
  BacktestResult,
  CustomerRisk,
  Dashboard,
  EvaluationSummary,
  InvoiceRisk,
} from "./api";
import { analyzeAll, api, uploadInvoices, uploadPayments } from "./api";

const DEFAULT_BUSINESS = "demo";
const currency = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

function pct(value: number) {
  return Math.round(value * 100) + "%";
}

function riskTone(value: number) {
  return value >= 0.7 ? "danger" : value >= 0.4 ? "warning" : "healthy";
}

function signedDays(value: number) {
  if (value === 0) return "0.0d";
  return value > 0 ? "+" + value.toFixed(1) + "d" : value.toFixed(1) + "d";
}

export default function App() {
  const [businessId, setBusinessId] = useState(DEFAULT_BUSINESS);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [customers, setCustomers] = useState<CustomerRisk[]>([]);
  const [invoices, setInvoices] = useState<InvoiceRisk[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [selectedCustomerId, setSelectedCustomerId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [backtesting, setBacktesting] = useState(false);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const results = await Promise.all([
        api.dashboard(businessId),
        api.customers(businessId),
        api.invoices(businessId),
        api.evaluations(businessId),
      ]);
      setDashboard(results[0]);
      setCustomers(results[1]);
      setInvoices(results[2]);
      setEvaluation(results[3]);
    } catch {
      setError("Unable to load the API. Start FastAPI and check the business ID.");
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(
    event: ChangeEvent<HTMLInputElement>,
    kind: "invoice" | "payment",
  ) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploading(true);
    setUploadMessage("");
    setError("");
    try {
      const result =
        kind === "invoice"
          ? await uploadInvoices(businessId, file)
          : await uploadPayments(businessId, file);

      setUploadMessage(
        (kind === "invoice" ? "Invoices" : "Payments") +
          ": " +
          JSON.stringify(result),
      );
      await load();
    } catch {
      setError("Upload failed. Check the file columns and API.");
    } finally {
      setUploading(false);
    }
  }

  async function runAnalysis() {
    setUploading(true);
    setUploadMessage("");
    setError("");
    try {
      const result = await analyzeAll(businessId);
      setUploadMessage(
        "Analyzed " +
          result.invoices_seen +
          " invoices; created " +
          result.predictions_created +
          " new predictions.",
      );
      await load();
    } catch {
      setError("Portfolio analysis failed. Import invoice and payment data first.");
    } finally {
      setUploading(false);
    }
  }

  async function runBacktest() {
    setBacktesting(true);
    setError("");
    setUploadMessage("");
    try {
      const result = await api.backtest(businessId, 3);
      setBacktest(result);
      setUploadMessage(
        "Backtested " +
          result.backtested_invoices +
          " settled invoices using a minimum history of " +
          result.min_history +
          ".",
      );
    } catch {
      setError("Historical backtest failed. Import enough settled invoice history first.");
    } finally {
      setBacktesting(false);
    }
  }

  useEffect(() => {
    void load();
    setBacktest(null);
  }, [businessId]);

  const averageScore = useMemo(() => {
    if (!customers.length) return 0;
    return customers.reduce((sum, customer) => sum + customer.score, 0) / customers.length;
  }, [customers]);

  const selectedCustomer = useMemo(
    () =>
      customers.find((customer) => customer.customer_id === selectedCustomerId) ??
      customers[0] ??
      null,
    [customers, selectedCustomerId],
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Payment Reliability OS</p>
          <h1>Know who will pay late before your cash is late.</h1>
        </div>
        <div className="toolbar">
          <label>
            Business ID
            <input
              value={businessId}
              onChange={(event) => setBusinessId(event.target.value)}
            />
          </label>
          <button onClick={() => void load()} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="ingest-panel">
        <div>
          <p className="eyebrow">Portfolio onboarding</p>
          <h2>Upload your invoice and payment exports</h2>
          <p className="muted">
            CSV or XLSX. The files are normalized, matched, and scored for this
            business.
          </p>
        </div>
        <div className="upload-actions">
          <label className="upload-button">
            Invoices
            <input
              type="file"
              accept=".csv,.xlsx"
              onChange={(event) => void handleUpload(event, "invoice")}
              disabled={uploading}
            />
          </label>
          <label className="upload-button">
            Payments
            <input
              type="file"
              accept=".csv,.xlsx"
              onChange={(event) => void handleUpload(event, "payment")}
              disabled={uploading}
            />
          </label>
          <button onClick={() => void runAnalysis()} disabled={uploading}>
            {uploading ? "Analyzing…" : "Analyze portfolio"}
          </button>
        </div>
        {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
      </section>

      <main>
        <section className="metric-grid">
          <Metric
            label="Receivables"
            value={
              dashboard
                ? currency.format(dashboard.outstanding_receivables)
                : "—"
            }
          />
          <Metric
            label="Cash at risk"
            value={dashboard ? currency.format(dashboard.cash_at_risk) : "—"}
            accent="danger"
          />
          <Metric
            label="High-risk invoices"
            value={dashboard?.high_risk_invoice_count ?? "—"}
          />
          <Metric
            label="Average buyer score"
            value={customers.length ? Math.round(averageScore) + "/100" : "—"}
          />
        </section>

        <section className="main-grid">
          <div className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Risk queue</p>
                <h2>Invoices that need attention</h2>
              </div>
              <span className="pill">{invoices.length} scored</span>
            </div>
            <div className="invoice-list">
              {invoices
                .slice(0, 7)
                .map((invoice) => (
                  <InvoiceRow key={invoice.invoice_id} invoice={invoice} />
                ))}
              {!loading && invoices.length === 0 && (
                <Empty text="No scored invoices yet. Import data and analyze an invoice." />
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Buyer intelligence</p>
                <h2>Payment reliability</h2>
              </div>
              <span className="score-large">
                {Math.round(averageScore)}
                <small>/100</small>
              </span>
            </div>
            <div className="buyer-list">
              {customers
                .slice(0, 7)
                .map((customer) => (
                  <BuyerRow
                    key={customer.customer_id}
                    customer={customer}
                    selected={customer.customer_id === selectedCustomer?.customer_id}
                    onSelect={() => setSelectedCustomerId(customer.customer_id)}
                  />
                ))}
              {!loading && customers.length === 0 && <Empty text="No buyers yet." />}
            </div>
          </div>
        </section>

        {selectedCustomer && (
          <section className="panel buyer-detail-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Score explainability</p>
                <h2>{selectedCustomer.customer}</h2>
                <p className="muted">
                  {selectedCustomer.confidence} confidence ·{" "}
                  {selectedCustomer.evidence_count} settled invoices ·{" "}
                  {selectedCustomer.model_version}
                </p>
              </div>
              <div className="detail-score">
                <strong>{Math.round(selectedCustomer.score)}</strong>
                <span>{selectedCustomer.grade}</span>
              </div>
            </div>

            <div className="feature-grid">
              <Feature label="On-time payment" value={pct(selectedCustomer.features.on_time_rate)} />
              <Feature label="Late-payment rate" value={pct(selectedCustomer.features.late_payment_rate)} />
              <Feature label="Average delay" value={selectedCustomer.features.average_delay_days.toFixed(1) + "d"} />
              <Feature label="Median delay" value={selectedCustomer.features.median_delay_days.toFixed(1) + "d"} />
              <Feature label="P90 delay" value={selectedCustomer.features.p90_delay_days.toFixed(1) + "d"} />
              <Feature label="Recent trend" value={signedDays(selectedCustomer.features.recent_trend_days)} />
              <Feature label="Consistency" value={pct(selectedCustomer.features.payment_consistency)} />
              <Feature label="Outstanding" value={currency.format(selectedCustomer.features.current_outstanding_exposure)} />
            </div>

            <div className="factor-list">
              {selectedCustomer.factors.map((factor) => (
                <div className="factor-row" key={factor.name}>
                  <div>
                    <strong>{factor.name}</strong>
                    <span>
                      {factor.weight * 100}% weight · {factor.direction}
                    </span>
                  </div>
                  <div className="factor-contribution">
                    <span>{factor.contribution.toFixed(1)} pts</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="reason-list">
              {selectedCustomer.reasons.map((reason) => (
                <span key={reason}>{reason}</span>
              ))}
            </div>
          </section>
        )}

        <section className="bottom-grid">
          <div className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Model health</p>
                <h2>Prediction accuracy</h2>
              </div>
              <div className="panel-actions">
                <span className="pill">
                  {evaluation?.evaluated_predictions ?? 0} evaluated
                </span>
                <button onClick={() => void runBacktest()} disabled={backtesting || loading}>
                  {backtesting ? "Backtesting…" : "Run historical backtest"}
                </button>
              </div>
            </div>
            <div className="health-grid">
              <Stat
                label="Date MAE"
                value={
                  evaluation?.date_mae_days == null
                    ? "—"
                    : evaluation.date_mae_days + " days"
                }
              />
              <Stat
                label="Late accuracy"
                value={
                  evaluation?.late_classification_accuracy == null
                    ? "—"
                    : pct(evaluation.late_classification_accuracy)
                }
              />
              <Stat
                label="Brier error"
                value={
                  evaluation?.mean_brier_error == null
                    ? "—"
                    : evaluation.mean_brier_error.toFixed(4)
                }
              />
            </div>
            <p className="muted">
              Stored metrics use evaluated predictions. Historical backtests
              score each settled invoice using only behavior available before
              that invoice was issued.
            </p>
            {backtest && (
              <div className="backtest-summary">
                <div className="backtest-head">
                  <strong>Historical backtest</strong>
                  <span>{backtest.backtested_invoices} eligible invoices · {backtest.model_version}</span>
                </div>
                <div className="backtest-metrics">
                  <span>
                    Observed late rate <strong>{pct(backtest.observed_late_rate ?? 0)}</strong>
                  </span>
                  <span>
                    Predicted late rate <strong>{pct(backtest.mean_predicted_late_rate ?? 0)}</strong>
                  </span>
                  <span>
                    Calibration gap <strong>{pct(backtest.mean_absolute_calibration_gap ?? 0)}</strong>
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Priority</p>
                <h2>Highest-risk invoice</h2>
              </div>
            </div>
            {invoices[0] ? (
              <div className="action-card">
                <div className="action-score">
                  <span
                    className={
                      "risk-dot " + riskTone(invoices[0].late_probability)
                    }
                  />
                  {pct(invoices[0].late_probability)} late risk
                </div>
                <h3>{invoices[0].customer}</h3>
                <p>
                  {invoices[0].invoice_number} ·{" "}
                  {currency.format(invoices[0].amount)}
                </p>
                <strong>{currency.format(invoices[0].cash_at_risk)} cash at risk</strong>
                <p className="muted">
                  {invoices[0].reasons[0] ??
                    "Review the buyer payment history."}
                </p>
              </div>
            ) : (
              <Empty text="Analyze an invoice to create the first priority item." />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className={"metric-card " + (accent ?? "")}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InvoiceRow({ invoice }: { invoice: InvoiceRisk }) {
  const tone = riskTone(invoice.late_probability);
  return (
    <div className="invoice-row">
      <div className="row-main">
        <span className={"risk-dot " + tone} />
        <div>
          <strong>{invoice.customer}</strong>
          <span>{invoice.invoice_number} · due {invoice.due_date}</span>
        </div>
      </div>
      <div className="row-right">
        <strong>{currency.format(invoice.amount)}</strong>
        <span className={"risk-label " + tone}>{pct(invoice.late_probability)} late</span>
      </div>
    </div>
  );
}

function BuyerRow({
  customer,
  selected,
  onSelect,
}: {
  customer: CustomerRisk;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={"buyer-row buyer-button" + (selected ? " selected" : "")} onClick={onSelect}>
      <div>
        <strong>{customer.customer}</strong>
        <span>
          {customer.evidence_count} settled invoices ·{" "}
          {customer.confidence.toLowerCase()} confidence
        </span>
      </div>
      <div className="buyer-score">
        <strong>{Math.round(customer.score)}</strong>
        <span>{customer.grade}</span>
      </div>
    </button>
  );
}

function Feature({ label, value }: { label: string; value: string }) {
  return (
    <div className="feature-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <div className="empty">{text}</div>;
}
