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
- Leakage-safe historical backtest with calibration buckets

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
        v
Explainable scoring baseline
        |
        +--> Buyer score + factor contributions
        +--> Invoice late-payment risk
        +--> Expected payment date
        +--> Cash at risk
        |
        v
Historical backtest
  - chronological holdout
  - date MAE
  - Brier error
  - classification accuracy
  - observed vs predicted late rate
  - calibration buckets
        |
        v
Actual payment outcome
        |
        v
Prediction-vs-actual evaluation
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
- `POST /score/buyer` — score a supplied buyer history without persistence

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

## Scoring baseline

Model version: `baseline-v1.0`.

The score is composed from explicit, inspectable contributions for late-payment rate, average delay, payment consistency, and recent trend. Cold-start buyers receive a neutral score with `N/A` grade and LOW confidence rather than a fabricated history.

The probability and score serve different purposes: the score summarizes operational reliability, while late-payment probability estimates the chance that a future invoice will be late. Neither should be represented as a regulated credit rating.

## Validation discipline

Historical backtests are chronological. A settled invoice is scored only when it has at least the configured number of settled invoices before its issue date; future invoices and future payment outcomes are excluded from the feature history. Backtests do not create prediction rows or evaluation records.

Treat model metrics as measurements of the supplied historical dataset, not universal accuracy claims.

## Evidence policy

Never publish model accuracy, customer counts, or cash-recovery figures without reproducible evidence. Evaluation metrics must be calculated from stored predictions matched to actual settled invoice outcomes.
