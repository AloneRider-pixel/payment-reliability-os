# Payment Reliability OS

Payment-behavior intelligence for Indian B2B businesses.

## MVP

Upload invoice and payment history, then get:

- Payment Reliability Score (0-100)
- A-F behavior grade
- Late-payment probability
- Expected payment delay
- Expected payment date
- Cash-at-risk
- Explainable risk factors
- Recommended operational action

This is an **operational payment-behavior score**, not a regulated credit rating or lending decision.

## MVP architecture

```
Invoices + Payments
        |
        v
Normalization / Validation
        |
        v
Buyer behavior features
        |
        v
Explainable risk baseline
        |
        +--> Buyer score
        +--> Invoice risk
        +--> Cash at risk
        +--> Recommended action
        |
        v
Actual payment outcome
        |
        v
Prediction-vs-actual evaluation
```

## Local development

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

uvicorn backend.app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Next phases

1. Historical CSV/Excel ingestion
2. Prediction evaluation and calibration
3. Real accounting-system integrations
4. Action workflows
5. ML payment-date prediction
6. Financial-partner/TReDS workflows subject to applicable regulation

## Evidence policy

Never publish model accuracy, customer counts, or cash-recovery figures without reproducible evidence.
