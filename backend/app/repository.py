from collections import defaultdict
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.entities import Customer, Invoice, Payment
from backend.app.models import BuyerHistory, InvoiceInput

def buyer_history(session: Session, customer_id: int) -> BuyerHistory:
    customer = session.scalar(select(Customer).options(selectinload(Customer.invoices).selectinload(Invoice.payments)).where(Customer.id == customer_id))
    if not customer: raise ValueError("customer not found")
    delays=[]; recent=[]; paid_count=0; late_count=0; total_amount=0.0
    for invoice in customer.invoices:
        paid=sum(float(p.amount) for p in invoice.payments)
        total_amount += float(invoice.amount)
        if paid + 0.005 < float(invoice.amount): continue
        paid_count += 1
        final_payment=max((p.payment_date for p in invoice.payments), default=invoice.due_date)
        delay=(final_payment - invoice.due_date).days
        delays.append(float(delay)); late_count += int(delay > 0)
    recent = delays[-5:]
    outstanding = 0.0
    for invoice in customer.invoices:
        paid=sum(float(p.amount) for p in invoice.payments)
        outstanding += max(float(invoice.amount) - paid, 0.0)
    return BuyerHistory(payment_delays_days=delays, invoice_count=paid_count, late_invoice_count=late_count, recent_delays_days=recent, average_invoice_amount=(total_amount/len(customer.invoices) if customer.invoices else 0), current_outstanding_amount=outstanding)

def invoice_input(session: Session, invoice_id: int) -> InvoiceInput:
    invoice = session.scalar(select(Invoice).options(selectinload(Invoice.payments), selectinload(Invoice.customer)).where(Invoice.id == invoice_id))
    if not invoice: raise ValueError("invoice not found")
    return InvoiceInput(due_date=invoice.due_date, amount=float(invoice.amount), buyer=buyer_history(session, invoice.customer_id))