# Payment Reliability OS Web

React + TypeScript operator dashboard for the Payment Reliability OS API.

## Scope

The UI presents server-computed payment-behavior scores, cash-at-risk, invoice queues, evidence/confidence information, and evaluation results.

The browser is not the authority for scoring. Keep scoring/model logic on the backend so the displayed result cannot diverge from persisted evaluation and policy state.

## Development

```bash
cd frontend
npm install
npm run dev
```

Default API: `http://127.0.0.1:8000`

Override with `VITE_API_BASE` when needed.

## Verification

```bash
npm run build
```

Run backend tests from repository root as part of end-to-end contract changes.

## Security

Do not expose payment credentials, database URLs, or model artifacts to the browser. Validate API error states and preserve the semantics of evidence/confidence fields.

## Review path

Review `src/api.ts`, score presentation, cash-at-risk views, and API error handling when server contracts change.
