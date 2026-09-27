import { useEffect, useMemo, useState } from "react";
import { api, CustomerRisk, Dashboard, EvaluationSummary, InvoiceRisk } from "./api";

const DEFAULT_BUSINESS = "demo";
const currency = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

function pct(value: number) { return Math.round(value * 100) + "%"; }
function riskTone(value: number) { return value >= 0.7 ? "danger" : value >= 0.4 ? "warning" : "healthy"; }

export default function App() {
  const [businessId, setBusinessId] = useState(DEFAULT_BUSINESS);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [customers, setCustomers] = useState<CustomerRisk[]>([]);
  const [invoices, setInvoices] = useState<InvoiceRisk[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

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

  useEffect(() => { void load(); }, [businessId]);

  const averageScore = useMemo(() => {
    if (!customers.length) return 0;
    return customers.reduce((sum, c) => sum + c.score, 0) / customers.length;
  }, [customers]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Payment Reliability OS</p>
          <h1>Know who will pay late before your cash is late.</h1>
        </div>
        <div className="toolbar">
          <label>Business ID<input value={businessId} onChange={(e) => setBusinessId(e.target.value)} /></label>
          <button onClick={() => void load()} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button>
        </div>
      </header>
      {error && <div className="error-banner">{error}</div>}
      <main>
        <section className="metric-grid">
          <Metric label="Receivables" value={dashboard ? currency.format(dashboard.outstanding_receivables) : "—"} />
          <Metric label="Cash at risk" value={dashboard ? currency.format(dashboard.cash_at_risk) : "—"} accent="danger" />
          <Metric label="High-risk invoices" value={dashboard?.high_risk_invoice_count ?? "—"} />
          <Metric label="Average buyer score" value={customers.length ? Math.round(averageScore) + "/100" : "—"} />
        </section>
        <section className="main-grid">
          <div className="panel">
            <div className="panel-header"><div><p className="eyebrow">Risk queue</p><h2>Invoices that need attention</h2></div><span className="pill">{invoices.length} scored</span></div>
            <div className="invoice-list">
              {invoices.slice(0, 7).map((invoice) => <InvoiceRow key={invoice.invoice_id} invoice={invoice} />)}
              {!loading && invoices.length === 0 && <Empty text="No scored invoices yet. Import data and analyze an invoice." />}
            </div>
          </div>
          <div className="panel">
            <div className="panel-header"><div><p className="eyebrow">Buyer intelligence</p><h2>Payment reliability</h2></div><span className="score-large">{Math.round(averageScore)}<small>/100</small></span></div>
            <div className="buyer-list">
              {customers.slice(0, 7).map((customer) => <BuyerRow key={customer.customer_id} customer={customer} />)}
              {!loading && customers.length === 0 && <Empty text="No buyers yet." />}
            </div>
          </div>
        </section>
        <section className="bottom-grid">
          <div className="panel">
            <div className="panel-header"><div><p className="eyebrow">Model health</p><h2>Prediction accuracy</h2></div><span className="pill">{evaluation?.evaluated_predictions ?? 0} evaluated</span></div>
            <div className="health-grid">
              <Stat label="Date MAE" value={evaluation?.date_mae_days == null ? "—" : evaluation.date_mae_days + " days"} />
              <Stat label="Late accuracy" value={evaluation?.late_classification_accuracy == null ? "—" : pct(evaluation.late_classification_accuracy)} />
              <Stat label="Brier error" value={evaluation?.mean_brier_error == null ? "—" : evaluation.mean_brier_error.toFixed(4)} />
            </div>
            <p className="muted">Validation metrics are shown only for settled invoices with stored predictions.</p>
          </div>
          <div className="panel">
            <div className="panel-header"><div><p className="eyebrow">Priority</p><h2>Highest-risk invoice</h2></div></div>
            {invoices[0] ? <div className="action-card">
              <div className="action-score"><span className={"risk-dot " + riskTone(invoices[0].late_probability)} />{pct(invoices[0].late_probability)} late risk</div>
              <h3>{invoices[0].customer}</h3>
              <p>{invoices[0].invoice_number} · {currency.format(invoices[0].amount)}</p>
              <strong>{currency.format(invoices[0].cash_at_risk)} cash at risk</strong>
              <p className="muted">{invoices[0].reasons[0] ?? "Review the buyer payment history."}</p>
            </div> : <Empty text="Analyze an invoice to create the first priority item." />}
          </div>
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return <div className={"metric-card " + (accent ?? "")}><span>{label}</span><strong>{value}</strong></div>;
}

function InvoiceRow({ invoice }: { invoice: InvoiceRisk }) {
  const tone = riskTone(invoice.late_probability);
  return <div className="invoice-row"><div className="row-main"><span className={"risk-dot " + tone} /><div><strong>{invoice.customer}</strong><span>{invoice.invoice_number} · due {invoice.due_date}</span></div></div><div className="row-right"><strong>{currency.format(invoice.amount)}</strong><span className={"risk-label " + tone}>{pct(invoice.late_probability)} late</span></div></div>;
}

function BuyerRow({ customer }: { customer: CustomerRisk }) {
  return <div className="buyer-row"><div><strong>{customer.customer}</strong><span>{customer.evidence_count} settled invoices · {customer.confidence.toLowerCase()} confidence</span></div><div className="buyer-score"><strong>{Math.round(customer.score)}</strong><span>{customer.grade}</span></div></div>;
}

function Stat({ label, value }: { label: string; value: string }) { return <div className="stat"><span>{label}</span><strong>{value}</strong></div>; }
function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }