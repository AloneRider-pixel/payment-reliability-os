import csv
import io
from datetime import date
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.entities import Business, Customer, Invoice, Payment

REQUIRED_INVOICE_COLUMNS = {"invoice_number", "customer", "invoice_date", "due_date", "amount"}
REQUIRED_PAYMENT_COLUMNS = {"invoice_number", "payment_date", "amount_paid"}

def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD") from exc

def _parse_money(value: str, field: str) -> Decimal:
    try:
        amount = Decimal(value.strip().replace(",", ""))
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if amount < 0:
        raise ValueError(f"{field} cannot be negative")
    return amount.quantize(Decimal("0.01"))

def _rows(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))

def ensure_business(session: Session, business_id: str, name: str | None = None) -> Business:
    business = session.get(Business, business_id)
    if business:
        return business
    business = Business(id=business_id, name=name or business_id)
    session.add(business)
    session.flush()
    return business

def import_invoices(session: Session, business_id: str, data: bytes) -> dict:
    rows = _rows(data)
    if not rows: return {"created": 0, "updated": 0}
    missing = REQUIRED_INVOICE_COLUMNS - set(rows[0])
    if missing: raise ValueError(f"Missing invoice columns: {', '.join(sorted(missing))}")
    ensure_business(session, business_id)
    created = updated = 0
    for row in rows:
        number = row["invoice_number"].strip()
        customer_key = row["customer"].strip()
        if not number or not customer_key: raise ValueError("invoice_number and customer are required")
        customer = session.scalar(select(Customer).where(Customer.business_id == business_id, Customer.external_key == customer_key))
        if not customer:
            customer = Customer(business_id=business_id, external_key=customer_key, name=customer_key)
            session.add(customer); session.flush()
        invoice = session.scalar(select(Invoice).where(Invoice.business_id == business_id, Invoice.invoice_number == number))
        values = {"customer_id": customer.id, "invoice_date": _parse_date(row["invoice_date"], f"{number}.invoice_date"), "due_date": _parse_date(row["due_date"], f"{number}.due_date"), "amount": _parse_money(row["amount"], f"{number}.amount")}
        if invoice:
            for key, value in values.items(): setattr(invoice, key, value)
            updated += 1
        else:
            session.add(Invoice(business_id=business_id, invoice_number=number, **values)); created += 1
    session.commit()
    return {"created": created, "updated": updated}

def import_payments(session: Session, business_id: str, data: bytes) -> dict:
    rows = _rows(data)
    if not rows: return {"created": 0}
    missing = REQUIRED_PAYMENT_COLUMNS - set(rows[0])
    if missing: raise ValueError(f"Missing payment columns: {', '.join(sorted(missing))}")
    created = 0
    for row in rows:
        number = row["invoice_number"].strip()
        invoice = session.scalar(select(Invoice).where(Invoice.business_id == business_id, Invoice.invoice_number == number))
        if not invoice: raise ValueError(f"Payment references unknown invoice: {number}")
        payment_date = _parse_date(row["payment_date"], f"{number}.payment_date")
        amount = _parse_money(row["amount_paid"], f"{number}.amount_paid")
        duplicate = session.scalar(select(Payment).where(Payment.invoice_id == invoice.id, Payment.payment_date == payment_date, Payment.amount == amount))
        if duplicate: continue
        session.add(Payment(invoice_id=invoice.id, payment_date=payment_date, amount=amount)); created += 1
    session.commit()
    return {"created": created}