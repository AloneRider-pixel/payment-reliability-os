# Prediction evaluation

Predictions are evaluated only after an invoice is fully settled.

For each prediction, the system records:
- actual final payment date
- actual days relative to due date
- whether payment was late
- absolute payment-date error
- squared probability error (Brier component)

The summary endpoint aggregates evaluations for one business.

These metrics are validation measurements for the current model version. They are not evidence of predictive performance until a sufficiently large, representative historical evaluation set has been accumulated.