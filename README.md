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

When the gate is not met, the baseline remains active. When an ML model is active, live invoice analysis uses it automatically. Runtime artifacts are stored under `.models/` and are intentionally excluded from Git.

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
