"""Data models for invoice extraction with confidence scoring."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ExtractedField:
    """A single extracted field with its confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0  # 0-100%
    source: str = "not_found"  # extraction method used

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 95:
            return "HIGH"
        elif self.confidence >= 85:
            return "MEDIUM"
        elif self.confidence >= 70:
            return "LOW"
        return "VERY LOW"

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence_pct": round(self.confidence, 1),
            "confidence_label": self.confidence_label,
            "source": self.source,
        }


@dataclass
class LineItem:
    """A single invoice line item."""
    line_number: int = 0
    description: ExtractedField = field(default_factory=ExtractedField)
    quantity: ExtractedField = field(default_factory=ExtractedField)
    unit_price: ExtractedField = field(default_factory=ExtractedField)
    amount: ExtractedField = field(default_factory=ExtractedField)
    unit_of_measure: ExtractedField = field(default_factory=ExtractedField)
    po_line_number: ExtractedField = field(default_factory=ExtractedField)

    def to_dict(self) -> dict:
        return {
            "line_number": self.line_number,
            "description": self.description.to_dict(),
            "quantity": self.quantity.to_dict(),
            "unit_price": self.unit_price.to_dict(),
            "amount": self.amount.to_dict(),
            "unit_of_measure": self.unit_of_measure.to_dict(),
            "po_line_number": self.po_line_number.to_dict(),
        }


@dataclass
class InvoiceData:
    """Complete extracted invoice data for three-way match."""

    # --- Header fields (Invoice identification) ---
    invoice_number: ExtractedField = field(default_factory=ExtractedField)
    invoice_date: ExtractedField = field(default_factory=ExtractedField)
    due_date: ExtractedField = field(default_factory=ExtractedField)
    currency: ExtractedField = field(default_factory=ExtractedField)
    payment_terms: ExtractedField = field(default_factory=ExtractedField)

    # --- PO Reference (critical for three-way match) ---
    po_number: ExtractedField = field(default_factory=ExtractedField)

    # --- Vendor / Supplier details ---
    vendor_name: ExtractedField = field(default_factory=ExtractedField)
    vendor_address: ExtractedField = field(default_factory=ExtractedField)
    vendor_tax_id: ExtractedField = field(default_factory=ExtractedField)
    vendor_bank_details: ExtractedField = field(default_factory=ExtractedField)

    # --- Bill-to / Buyer details ---
    buyer_name: ExtractedField = field(default_factory=ExtractedField)
    buyer_address: ExtractedField = field(default_factory=ExtractedField)

    # --- Delivery / Ship-to (for GRN matching) ---
    delivery_address: ExtractedField = field(default_factory=ExtractedField)
    delivery_note_number: ExtractedField = field(default_factory=ExtractedField)

    # --- Totals ---
    subtotal: ExtractedField = field(default_factory=ExtractedField)
    tax_rate: ExtractedField = field(default_factory=ExtractedField)
    tax_amount: ExtractedField = field(default_factory=ExtractedField)
    total_amount: ExtractedField = field(default_factory=ExtractedField)

    # --- Line items ---
    line_items: list = field(default_factory=list)

    # --- Raw text ---
    raw_text: str = ""

    @property
    def overall_confidence(self) -> float:
        """Weighted average confidence across all critical fields."""
        weights = {
            "invoice_number": 3,
            "invoice_date": 2,
            "po_number": 3,
            "vendor_name": 2,
            "total_amount": 3,
            "subtotal": 1,
            "tax_amount": 1,
            "currency": 1,
        }
        total_weight = 0
        weighted_sum = 0
        for fname, weight in weights.items():
            f = getattr(self, fname)
            if isinstance(f, ExtractedField):
                weighted_sum += f.confidence * weight
                total_weight += weight
        return round(weighted_sum / total_weight, 1) if total_weight else 0.0

    def get_all_fields(self) -> dict:
        """Return all header fields as a dict."""
        result = {}
        for fname in [
            "invoice_number", "invoice_date", "due_date", "currency",
            "payment_terms", "po_number", "vendor_name", "vendor_address",
            "vendor_tax_id", "vendor_bank_details", "buyer_name",
            "buyer_address", "delivery_address", "delivery_note_number",
            "subtotal", "tax_rate", "tax_amount", "total_amount",
        ]:
            f = getattr(self, fname)
            if isinstance(f, ExtractedField):
                result[fname] = f.to_dict()
        return result

    def to_erp_json(self) -> str:
        """Export as ERP-ready JSON."""
        data = {
            "invoice_header": self.get_all_fields(),
            "line_items": [li.to_dict() for li in self.line_items],
            "extraction_metadata": {
                "overall_confidence_pct": self.overall_confidence,
                "field_count": len(self.get_all_fields()),
                "line_item_count": len(self.line_items),
            },
        }
        return json.dumps(data, indent=2)

    def to_erp_flat(self) -> list[dict]:
        """Export as flat records suitable for CSV/Excel ERP import."""
        rows = []
        header = {}
        for fname, fdata in self.get_all_fields().items():
            header[fname] = fdata["value"]
            header[f"{fname}_confidence"] = fdata["confidence_pct"]

        if self.line_items:
            for li in self.line_items:
                row = dict(header)
                row["line_number"] = li.line_number
                row["line_description"] = li.description.value
                row["line_description_confidence"] = li.description.confidence
                row["line_quantity"] = li.quantity.value
                row["line_quantity_confidence"] = li.quantity.confidence
                row["line_unit_price"] = li.unit_price.value
                row["line_unit_price_confidence"] = li.unit_price.confidence
                row["line_amount"] = li.amount.value
                row["line_amount_confidence"] = li.amount.confidence
                row["line_uom"] = li.unit_of_measure.value
                row["line_po_line"] = li.po_line_number.value
                rows.append(row)
        else:
            rows.append(header)

        return rows
