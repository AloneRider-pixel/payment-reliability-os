# Payment Reliability OS

Payment-behavior intelligence for Indian B2B receivables workflows.

## Product

Given invoice and payment history, the system produces:

- Payment Reliability Score (0–100) and behavior grade.
- Late-payment probability and expected delay/date.
- Cash-at-risk and explainable factor contributions.
- Confidence/evidence information.
- Leakage-safe historical backtests.
- Temporal ML candidates with baseline comparison and promotion gates.
- Persistent model registry, rollback, and drift monitoring.
- Deterministic receivables actions with human completion.

The score is an **operational payment-behavior index**. It is not a regulated credit rating or a standalone lending/underwriting decision.

## Architecture

```text
Invoices + Payments
        ↓
Normalization / validation
        ↓
Leakage-safe features
        ├───────────────┐
        ↓               ↓
Explainable baseline   Temporal ML candidate
        └───────┬───────┘
                ↓
          Invoice risk
                ↓
          Cash at risk
                ↓
       Observed payment outcome
                ↓
           Evaluation
```

The action layer is separated from the model layer so operational recommendations remain deterministic and reviewable.

## Model lifecycle

Historical features are constructed without future outcomes. Candidates are evaluated against the explainable baseline and promoted only when the configured gate is satisfied. Model lineage, metrics, rollback state, and registry metadata are persisted in the database.

Drift checks compare eligible recent cohorts against the active model's training feature distribution. Thresholds are operational prototype policies, not statistical significance tests.

## Scheduled operations

```text
reconcile outcomes
   → check drift / candidate promotion
   → refresh invoice predictions
   → refresh action queue
```

Repeated business/date/job combinations use an idempotent database slot.

## Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Persistence | PostgreSQL |
| ML | scikit-learn / temporal evaluation |
| Frontend | React, TypeScript, Vite |
| Operations | Scheduled jobs, rollback/drift controls |
| Delivery | GitHub Actions |

## Quick start

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Verification

```bash
python -m pytest -q
cd frontend
npm run build
```

CI covers backend tests and the frontend build; CodeQL, dependency review, Scorecard, and scheduled operations are separate workflows.

## Security and trust boundaries

Invoice/payment data, model artifacts, predictions, and operational actions have different trust boundaries. Do not turn model output into an automated commercial decision without an explicit policy and human-review path.

## Evidence policy

Never publish accuracy, customer counts, recovery, or cash-at-risk claims without reproducible evidence including dataset, leakage controls, metrics, environment, and producing commit.

See [docs/evaluation.md](docs/evaluation.md).

## Documentation

- [Evaluation](docs/evaluation.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Maintenance standard

Keep historical features leakage-safe, baseline gates and rollback controls intact, and financial semantics separate from model output.

## License

MIT
