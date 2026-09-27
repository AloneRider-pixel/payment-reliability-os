from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.entities import Customer, Invoice
from backend.app.models import BuyerHistory, InvoiceInput


def buyer_history(session: Session, customer_id: int) -> BuyerHistory:
    customer = session.scalar(
        select(Customer)
        .options(selectinload(Customer.invoices).selectinload(Invoice.payments))
        .where(Customer.id == customer_id)
    )
    if not customer:
        raise ValueError("customer not found")

    settled = []
    for invoice in sorted(customer.invoices, key=lambda item: (item.invoice_date, item.id)):
        paid = sum(float(payment.amount) for payment in invoice.payments)
        if paid + 0.005 < float(invoice.amount):
            continue

        final_payment = max(
            (payment.payment_date for payment in invoice.payments),
            default=invoice.due_date,
        )
        settled.append(
            {
                "amount": float(invoice.amount),
                "delay": float((final_payment - invoice.due_date).days),
            }
        )

    delays = [item["delay"] for item in settled]
    late_count = sum(delay > 0 for delay in delays)
    recent = delays[-5:]
    settled_amounts = [item["amount"] for item in settled]

    outstanding = 0.0
    for invoice in customer.invoices:
        paid = sum(float(payment.amount) for payment in invoice.payments)
        outstanding += max(float(invoice.amount) - paid, 0.0)

    return BuyerHistory(
        payment_delays_days=delays,
        invoice_count=len(delays),
        late_invoice_count=late_count,
        recent_delays_days=recent,
        average_invoice_amount=(
            sum(settled_amounts) / len(settled_amounts) if settled_amounts else 0.0
        ),
        current_outstanding_amount=outstanding,
    )


def invoice_input(session: Session, invoice_id: int) -> InvoiceInput:
    invoice = session.scalar(
        select(Invoice)
        .options(selectinload(Invoice.payments), selectinload(Invoice.customer))
        .where(Invoice.id == invoice_id)
    )
    if not invoice:
        raise ValueError("invoice not found")

    return InvoiceInput(
        due_date=invoice.due_date,
        amount=float(invoice.amount),
        buyer=buyer_history(session, invoice.customer_id),
    )
