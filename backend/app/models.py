from datetime import date

from pydantic import BaseModel, Field


class BuyerHistory(BaseModel):
    payment_delays_days: list[float] = Field(default_factory=list)
    invoice_count: int = 0
    late_invoice_count: int = 0
    recent_delays_days: list[float] = Field(default_factory=list)
    average_invoice_amount: float = 0
    current_outstanding_amount: float = 0


class InvoiceInput(BaseModel):
    due_date: date
    amount: float = Field(ge=0)
    buyer: BuyerHistory
