# Payment Reliability OS

Payment-behavior intelligence for Indian B2B businesses.

## Product

Upload invoice and payment history to generate:

- Payment Reliability Score (0–100) and A–F behavior grade.
- Late-payment probability and expected payment delay/date.
- Cash-at-risk and explainable factor contributions.
- Confidence/evidence information.
- Leakage-safe historical backtesting.
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
Leakage-safe historical features
        ├───────────────┐
        ↓               ↓
Explainable baseline   Temporal ML candidate
        └───────┬───────┘
                ↓
          Invoice risk
                ↓
          Cash at risk
                ↓
      Actual payment outcome
                ↓
         Model evaluation
```

The action engine is deliberately separated from the ML layer so operational recommendations remain deterministic and reviewable.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /imports/invoices` | Import invoices |
| `POST /imports/payments` | Import payments |
| `GET /risk/customers` | Buyer scores/features/exposure |
| `GET /risk/invoices` | Invoice risk queue |
| `POST /evaluations/backtest` | Chronological backtest |
| `GET /evaluations/summary` | Stored prediction metrics |
| `POST /models/train` | Train/evaluate ML candidate |
| `GET /models/status` | Active model status |
| `GET /models/drift` | Drift summary |
| `POST /models/retrain-if-needed` | Policy-driven retraining |
| `POST /models/rollback` | Restore previous model |
| `POST /actions/generate` | Generate receivables actions |
| `GET /actions` | Action queue |
| `GET /jobs/runs` | Scheduled-operation audit history |

## Model lifecycle

Training is chronological: future payment outcomes are excluded from historical feature construction. Candidates are compared with the explainable baseline and promoted only when the configured evaluation gate is satisfied.

Promoted model metadata, lineage, metrics, and rollback state are persisted in the database rather than depending on local files.

Drift monitoring reconstructs a recent eligible cohort against the active model's training feature distribution. The configured thresholds are operational prototype policies, not statistical significance tests.

## Scheduled operations

`scripts/run_scheduled_jobs.py` executes, in order:

1. Reconcile pending predictions with new payment outcomes.
2. Check drift and train/promote a candidate when policy requires it.
3. Refresh outstanding invoice predictions.
4. Refresh the receivables action queue.

Repeated business/date/job combinations are protected by an idempotent database slot.

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

CI runs backend tests and the frontend production build; CodeQL, dependency review, Scorecard, and scheduled operations are separate repository workflows.

## Security and trust boundaries

Treat invoice/payment records, ML artifacts, model outputs, and operational actions as separate trust boundaries. Never convert a model score into an automated commercial decision without an explicit application policy and human-review path.

## Evidence policy

Never publish model accuracy, customer counts, recovery figures, or cash-at-risk claims without reproducible evidence. Evaluation results must identify the historical dataset, leakage controls, metrics, environment, and producing commit.

See [docs/evaluation.md](docs/evaluation.md).

## Review path

Start with evaluation and tests, then review feature cutoffs, promotion/rollback logic, drift thresholds, scheduler idempotency, and action authorization together.

## Maintenance standard

Keep historical features leakage-safe, preserve baseline gates and rollback controls, and keep financial/payment semantics separate from model output.

## License

MIT
