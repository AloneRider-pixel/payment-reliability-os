# Payment Reliability OS Web

React + TypeScript operator dashboard for Payment Reliability OS.

## Responsibilities

The frontend presents server-computed payment-behavior scores, invoice queues, cash-at-risk, evidence/confidence data, and evaluation results. It is not the authority for scoring, model promotion, or policy decisions.

## Development

```bash
cd frontend
npm install
npm run dev
```

Override `VITE_API_BASE` when the API is not at the local default.

## Verification

```bash
npm run build
```

Run backend tests from the repository root when changing shared contracts.

## Security

Never expose payment credentials, database URLs, or model artifacts to the browser. Preserve the semantics of evidence/confidence fields and treat API errors as distinct from empty data.

## Review path

Review `src/api.ts`, score presentation, cash-at-risk views, and API error handling for server-contract changes.

## License

MIT
