# Payment Reliability OS Web

React + TypeScript operator dashboard for the Payment Reliability OS API.

## Run

From repository root:

    cd frontend
    npm install
    npm run dev

The dashboard expects FastAPI at `http://127.0.0.1:8000`.

Override with `VITE_API_BASE` when needed.

## Product views

- Receivables and cash-at-risk
- High-risk invoice queue
- Buyer Payment Reliability scores
- Evidence/confidence display
- Prediction validation metrics