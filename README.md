# Payment Reliability OS

Payment-behavior intelligence for Indian B2B businesses.

## Current product slice

Upload invoice and payment history, then get:

- Payment Reliability Score (0-100)
- A-F behavior grade
- Late-payment probability
- Expected payment delay and expected payment date
- Cash-at-risk for outstanding invoices
- Explainable score factors and underlying behavior features
- Confidence level and evidence count
- Prediction-vs-actual evaluation metrics
- Leakage-safe historical backtesting
- A temporally trained ML candidate for late-risk and payment-delay prediction
- Persistent model registry with promotion history and rollback
- Leakage-safe model drift monitoring and retraining triggers
- Deterministic receivables action engine with human-in-the-loop completion

The score is an **operational payment-behavior index**, not a regulated credit rating and not a standalone lending or underwriting decision.

## Architecture

```
Invoices + Payments
        |
        v
Normalization / Validation
        |
        v
Settled buyer history
        |
        v
Feature engine
  - on-time rate
  - late-payment rate
  - average / median / P90 delay
  - recent trend
  - payment consistency
  - invoice-size baseline
  - outstanding exposure
        |
        +-----------------------------+
        |                             |
        v                             v
Explainable baseline            Temporal ML candidate
        |                       - LogisticRegression
        |                       - Ridge delay model
        |                       - temporal train/test split
        |                       - baseline promotion gate
        |                             |
        +-------------+---------------+
                      v
               Invoice risk
                      |
                      v
                Cash at risk
                      |
                      v
            Actual payment outcome
                      |
                      v
              Model evaluation
```

## API

Key endpoints:

- `POST /imports/invoices` — CSV/XLSX ingestion
- `POST /imports/payments` — CSV/XLSX ingestion
- `POST /risk/analyze-all` — batch risk analysis for outstanding invoices
- `GET /risk/customers` — buyer scores, features, factor contributions, and exposure
- `GET /risk/invoices` — latest invoice risk queue
- `POST /evaluations/backtest` — chronological historical validation using only prior settled behavior
- `GET /evaluations/summary` — prediction quality metrics from stored predictions
- `POST /models/train` — train a temporal ML candidate and compare it with the baseline
- `GET /models/status` — show whether a promoted ML model is active
- `GET /models/drift` — measure recent feature drift against the active model's training distribution
- `GET /models/drift/latest` — retrieve the latest persisted drift snapshot
- `POST /models/retrain-if-needed` — measure drift and train a candidate only when the retraining policy is triggered
- `POST /models/rollback` — restore the previous promoted ML model
- `POST /actions/generate` — generate explainable receivables actions for outstanding scored invoices
- `GET /actions` — retrieve the open/completed/dismissed action queue
- `GET /actions/summary` — summarize open collection workload and exposure
- `POST /actions/{action_id}/status` — mark a recommended action open, completed, or dismissed
- `GET /jobs/runs` — show the latest scheduled-operation audit history
- `POST /score/buyer` — score supplied buyer history without persistence

## ML model lifecycle

Model version: `ml-v0.1`.

Training uses chronological samples. For each historical invoice, the feature history is cut off at that invoice's issue date, preventing future payment outcomes from entering the training features.

The candidate contains:

- Logistic regression for late-payment probability
- Ridge regression for expected positive payment delay
- StandardScaler parameters and model coefficients stored as JSON artifacts
- Temporal train/test metrics
- A promotion gate requiring the candidate to match or improve the baseline on both Brier error and payment-date MAE

When the gate is not met, the baseline remains active. When an ML model is active, live invoice analysis uses it automatically. Model artifacts, metrics, lineage, and lifecycle state are stored in the `model_registry` database table, so deployments do not depend on local filesystem state. Previous promoted versions are retained for rollback.

### Drift monitoring

The active model's training feature means and scales are used as the reference distribution. A recent cohort of settled invoices is reconstructed with the same leakage-safe historical feature cutoff. The service reports a mean standardized feature shift and the maximum individual feature shift.

The default operating policy recommends retraining when the aggregate drift score reaches `0.75` or any feature reaches `1.50σ`, provided the recent cohort has at least 12 eligible samples. These are operational thresholds for the prototype, not statistical significance tests.

`POST /models/retrain-if-needed` records a drift snapshot and triggers the existing temporal training pipeline only when the policy recommends retraining. The candidate still must pass the baseline promotion gate before becoming active.



### Scheduled operations

The repository includes a daily operations runner at `scripts/run_scheduled_jobs.py`. Its four stages run in dependency order:

1. Evaluate pending predictions against newly settled invoices.
2. Check model drift and run the existing temporal training/promotion pipeline when needed.
3. Refresh outstanding invoice risk predictions using the currently active model.
4. Refresh the receivables action queue.

Each business/job/date combination is protected by a unique database slot in `scheduled_job_runs`, so repeated invocations of the same daily slot are idempotent. The audit record stores start/end timestamps, completion state, and structured output.

GitHub Actions runs this workflow every day at 02:00 UTC (07:30 IST). Set the repository `DATABASE_URL` secret to the persistent application database before enabling scheduled production execution. The workflow also supports manual dispatch for one business or the full portfolio.

Scheduled execution does not automatically send collection messages or alter commercial terms; it refreshes predictions, model lifecycle state, and human-reviewable action recommendations.

### Receivables action engine

The action engine is deliberately deterministic and separate from the ML model. It converts the latest invoice prediction plus due-date state into an operational next step:

- `MONITOR` — no immediate collection action
- `PRE_DUE_REMINDER` — routine reminder
- `PRE_DUE_PRIORITY` — pre-due contact for elevated modeled risk
- `COLLECTION_FOLLOW_UP` — follow up on an overdue balance
- `PRIORITY_COLLECTION` — confirm a committed payment date for materially overdue/high-risk invoices
- `ESCALATION_REVIEW` — account-owner review for materially overdue, high-risk balances

Each action stores its priority score, outstanding exposure, model version, due-date state, and an explainable reason. Completing an action only records workflow state; the product does not automatically send collection messages or alter commercial terms.

## Validation discipline

Historical backtests are chronological. A settled invoice is scored only when it has at least the configured number of settled invoices before its issue date; future invoices and future payment outcomes are excluded from the feature history.

Backtests do not create prediction rows or evaluation records.

Treat model metrics as measurements of the supplied historical dataset, not universal accuracy claims. More data is required before using ML performance as a generalized business claim.

## Local development

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

uvicorn backend.app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Evidence policy

Never publish model accuracy, customer counts, or cash-recovery figures without reproducible evidence. Evaluation metrics must be calculated from stored predictions matched to actual settled invoice outcomes.
