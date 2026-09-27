from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db import Base

class Business(Base):
    __tablename__ = "businesses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    customers: Mapped[list["Customer"]] = relationship(back_populates="business", cascade="all, delete-orphan")
    invoices: Mapped[list["Invoice"]] = relationship(back_populates="business", cascade="all, delete-orphan")

class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    external_key: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    business: Mapped[Business] = relationship(back_populates="customers")
    invoices: Mapped[list["Invoice"]] = relationship(back_populates="customer", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("business_id", "external_key", name="uq_customer_business_external"),)

class Invoice(Base):
    __tablename__ = "invoices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    invoice_number: Mapped[str] = mapped_column(String(128), nullable=False)
    invoice_date: Mapped[date] = mapped_column(Date, nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    business: Mapped[Business] = relationship(back_populates="invoices")
    customer: Mapped[Customer] = relationship(back_populates="invoices")
    payments: Mapped[list["Payment"]] = relationship(back_populates="invoice", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("business_id", "invoice_number", name="uq_invoice_business_number"),)

class Payment(Base):
    __tablename__ = "payments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), index=True)
    payment_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    invoice: Mapped[Invoice] = relationship(back_populates="payments")

class RiskPrediction(Base):
    __tablename__ = "risk_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    late_probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    expected_delay_days: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    expected_payment_date: Mapped[date] = mapped_column(Date, nullable=False)
    cash_at_risk: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    reasons: Mapped[str] = mapped_column(Text, nullable=False)

class PredictionEvaluation(Base):
    __tablename__ = "prediction_evaluations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("risk_predictions.id"), unique=True, index=True)
    actual_payment_date: Mapped[date] = mapped_column(Date, nullable=False)
    actual_delay_days: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_late: Mapped[bool] = mapped_column(nullable=False)
    payment_date_error_days: Mapped[int] = mapped_column(Integer, nullable=False)
    probability_brier_error: Mapped[float] = mapped_column(Numeric(10, 8), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    prediction: Mapped[RiskPrediction] = relationship()
    __table_args__ = (UniqueConstraint("prediction_id", name="uq_prediction_evaluation_prediction"),)

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate")
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    train_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[str] = mapped_column(Text, nullable=False)
    artifact: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    __table_args__ = (
        UniqueConstraint("business_id", "model_version", name="uq_model_business_version"),
    )


class ModelDriftSnapshot(Base):
    __tablename__ = "model_drift_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    measured_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_score: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    max_feature_shift: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    recommendation: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)


class CollectionAction(Base):
    __tablename__ = "collection_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), index=True)
    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("risk_predictions.id"),
        nullable=True,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    action_type: Mapped[str] = mapped_column(String(48), nullable=False)
    priority_score: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="open")
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    days_overdue: Mapped[int] = mapped_column(Integer, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "invoice_id",
            "prediction_id",
            name="uq_collection_action_prediction",
        ),
    )


class ScheduledJobRun(Base):
    __tablename__ = "scheduled_job_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    result: Mapped[str] = mapped_column(Text, nullable=False, default="")
    __table_args__ = (
        UniqueConstraint(
            "business_id",
            "job_type",
            "scheduled_for",
            name="uq_scheduled_job_business_slot",
        ),
    )
