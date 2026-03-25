"""Core invoice extraction engine with confidence scoring."""

import re
from typing import Optional
import pdfplumber
from models import ExtractedField, LineItem, InvoiceData


# ---------------------------------------------------------------------------
# Regex pattern banks – each tuple: (compiled regex, confidence boost)
# ---------------------------------------------------------------------------

INVOICE_NUM_PATTERNS = [
    (re.compile(r"(?:invoice\s*(?:no|number|#|num)[.:;\s]*)\s*([A-Z0-9][\w\-/]{2,20})", re.I), 95),
    (re.compile(r"(?:inv\s*[.:#\-]\s*)([A-Z0-9][\w\-/]{2,20})", re.I), 90),
    (re.compile(r"(?:bill\s*(?:no|number|#)[.:;\s]*)\s*([A-Z0-9][\w\-/]{2,20})", re.I), 80),
]

PO_NUMBER_PATTERNS = [
    (re.compile(r"(?:purchase\s*order\s*(?:no|number|#|num)?[.:;\s]*)\s*([A-Z0-9][\w\-/]{2,20})", re.I), 95),
    (re.compile(r"(?:p\.?\s*o\.?\s*(?:no|number|#|num)?[.:;\s]*)\s*([A-Z0-9][\w\-/]{2,20})", re.I), 93),
    (re.compile(r"(?:PO[.:;\s#\-]+)([A-Z0-9][\w\-/]{2,20})", re.I), 90),
    (re.compile(r"(?:order\s*(?:no|number|#|ref)[.:;\s]*)\s*([A-Z0-9][\w\-/]{2,20})", re.I), 75),
]

DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY
    (re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b"), 90),
    # YYYY-MM-DD
    (re.compile(r"\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b"), 92),
    # Month DD, YYYY
    (re.compile(r"\b(\w+\s+\d{1,2},?\s+\d{4})\b"), 88),
    # DD Month YYYY
    (re.compile(r"\b(\d{1,2}\s+\w+\s+\d{4})\b"), 85),
]

CURRENCY_PATTERNS = [
    (re.compile(r"\b(USD|EUR|GBP|CAD|AUD|CHF|JPY|INR|CNY|NZD)\b", re.I), 95),
    (re.compile(r"[\$\£\€\¥]"), 90),
]

CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY"}

AMOUNT_PATTERNS = [
    (re.compile(r"(?:total\s*(?:amount|due)?)[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 95),
    (re.compile(r"(?:amount\s*(?:due|payable))[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 93),
    (re.compile(r"(?:grand\s*total)[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 92),
    (re.compile(r"(?:balance\s*due)[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 88),
]

SUBTOTAL_PATTERNS = [
    (re.compile(r"(?:sub\s*-?\s*total)[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 93),
    (re.compile(r"(?:net\s*(?:amount|total))[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 88),
]

TAX_PATTERNS = [
    (re.compile(r"(?:(?:sales\s*)?tax|VAT|GST|HST)[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 92),
    (re.compile(r"(?:tax\s*amount)[.:;\s]*[\$\£\€\¥]?\s*([\d,]+\.?\d*)", re.I), 90),
]

TAX_RATE_PATTERNS = [
    (re.compile(r"(?:(?:sales\s*)?tax|VAT|GST|HST)\s*(?:rate)?[.:;\s]*\(?\s*(\d+\.?\d*)\s*%", re.I), 92),
]

TAX_ID_PATTERNS = [
    (re.compile(r"(?:tax\s*id|tax\s*identification|TIN|EIN|VAT\s*(?:no|number|#|reg))[.:;\s]*([A-Z0-9\-]{5,20})", re.I), 93),
    (re.compile(r"(?:GST\s*(?:no|number|#|reg)|ABN)[.:;\s]*([A-Z0-9\-\s]{5,20})", re.I), 90),
]

PAYMENT_TERMS_PATTERNS = [
    (re.compile(r"(?:payment\s*terms?|terms\s*of\s*payment)[.:;\s]*(Net\s*\d+|Due\s*(?:on|upon)\s*\w+|COD|\d+\s*days?)", re.I), 92),
    (re.compile(r"\b(Net\s*\d+)\b", re.I), 80),
]

DELIVERY_NOTE_PATTERNS = [
    (re.compile(r"(?:delivery\s*note|DN|dispatch\s*note|GRN|goods\s*receipt)\s*(?:no|number|#|num)?[.:;\s]*([A-Z0-9][\w\-/]{2,20})", re.I), 92),
    (re.compile(r"(?:packing\s*slip|shipping\s*(?:no|ref))[.:;\s]*([A-Z0-9][\w\-/]{2,20})", re.I), 85),
]


def _first_match(text: str, patterns: list[tuple]) -> ExtractedField:
    """Try patterns in order, return first match with its confidence."""
    for pattern, confidence in patterns:
        m = pattern.search(text)
        if m:
            return ExtractedField(
                value=m.group(1).strip() if m.lastindex else m.group(0).strip(),
                confidence=confidence,
                source="regex",
            )
    return ExtractedField(value=None, confidence=0, source="not_found")


def _extract_invoice_date(text: str) -> ExtractedField:
    """Extract invoice date with context-aware matching."""
    # Look for date near 'invoice date' label first
    label_pattern = re.compile(
        r"(?:invoice\s*date|date\s*of\s*invoice|dated)[.:;\s]*(.{6,20})", re.I
    )
    m = label_pattern.search(text)
    if m:
        snippet = m.group(1)
        for pat, conf in DATE_PATTERNS:
            dm = pat.search(snippet)
            if dm:
                return ExtractedField(value=dm.group(1).strip(), confidence=min(conf + 3, 98), source="regex_labelled")

    # Look for 'date' label
    date_label = re.compile(r"(?:^|\n)\s*date[.:;\s]*(.{6,20})", re.I)
    m = date_label.search(text)
    if m:
        snippet = m.group(1)
        for pat, conf in DATE_PATTERNS:
            dm = pat.search(snippet)
            if dm:
                return ExtractedField(value=dm.group(1).strip(), confidence=conf, source="regex_labelled")

    # Fallback: first date found
    for pat, conf in DATE_PATTERNS:
        dm = pat.search(text)
        if dm:
            return ExtractedField(value=dm.group(1).strip(), confidence=max(conf - 15, 40), source="regex_fallback")

    return ExtractedField(value=None, confidence=0, source="not_found")


def _extract_due_date(text: str) -> ExtractedField:
    label_pattern = re.compile(
        r"(?:due\s*date|payment\s*due|pay\s*by)[.:;\s]*(.{6,20})", re.I
    )
    m = label_pattern.search(text)
    if m:
        snippet = m.group(1)
        for pat, conf in DATE_PATTERNS:
            dm = pat.search(snippet)
            if dm:
                return ExtractedField(value=dm.group(1).strip(), confidence=conf, source="regex_labelled")
    return ExtractedField(value=None, confidence=0, source="not_found")


def _extract_currency(text: str) -> ExtractedField:
    for pat, conf in CURRENCY_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(0).strip()
            val = CURRENCY_SYMBOLS.get(val, val.upper())
            return ExtractedField(value=val, confidence=conf, source="regex")
    return ExtractedField(value=None, confidence=0, source="not_found")


def _extract_vendor_name(text: str) -> ExtractedField:
    """Heuristic: vendor name is typically at the very top of the invoice."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Skip very short lines or lines that look like headers
    for line in lines[:8]:
        if len(line) < 3:
            continue
        if re.match(r"^(invoice|tax\s*invoice|bill|statement|page\s*\d)", line, re.I):
            continue
        if re.match(r"^\d", line):
            continue
        # Likely the vendor/company name
        return ExtractedField(value=line, confidence=70, source="heuristic_position")
    return ExtractedField(value=None, confidence=0, source="not_found")


def _extract_address_block(text: str, after_label: str) -> ExtractedField:
    """Extract multi-line address block after a label."""
    pattern = re.compile(
        rf"(?:{after_label})[.:;\s]*((?:.+\n?){{1,5}})", re.I | re.MULTILINE
    )
    m = pattern.search(text)
    if m:
        block = m.group(1).strip()
        # Take up to 4 lines
        lines = [l.strip() for l in block.split("\n") if l.strip()][:4]
        return ExtractedField(value="\n".join(lines), confidence=80, source="regex_block")
    return ExtractedField(value=None, confidence=0, source="not_found")


def _extract_line_items_from_tables(pdf_path: str) -> tuple[list[LineItem], float]:
    """Extract line items from PDF tables using pdfplumber."""
    items = []
    base_confidence = 85  # table extraction is fairly reliable

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Find header row
                header_row = None
                header_idx = -1
                for i, row in enumerate(table):
                    row_text = " ".join(str(c or "") for c in row).lower()
                    if any(kw in row_text for kw in ["description", "item", "qty", "quantity", "amount", "price", "unit"]):
                        header_row = [str(c or "").strip().lower() for c in row]
                        header_idx = i
                        break

                if header_row is None:
                    continue

                # Map columns
                col_map = {}
                for ci, col in enumerate(header_row):
                    if any(k in col for k in ["desc", "item", "particular", "product", "service"]):
                        col_map["description"] = ci
                    elif any(k in col for k in ["qty", "quantity", "qnty"]):
                        col_map["quantity"] = ci
                    elif any(k in col for k in ["unit price", "rate", "price", "unit cost"]):
                        col_map["unit_price"] = ci
                    elif any(k in col for k in ["amount", "total", "line total", "ext"]):
                        col_map["amount"] = ci
                    elif any(k in col for k in ["uom", "unit of measure", "unit"]):
                        col_map["uom"] = ci
                    elif any(k in col for k in ["po line", "po_line"]):
                        col_map["po_line"] = ci

                # Extract data rows
                for row in table[header_idx + 1:]:
                    if not row or all(not str(c or "").strip() for c in row):
                        continue

                    # Skip summary rows
                    row_text = " ".join(str(c or "") for c in row).lower()
                    if any(kw in row_text for kw in ["total", "subtotal", "sub-total", "tax", "vat", "grand"]):
                        continue

                    li = LineItem(line_number=len(items) + 1)

                    if "description" in col_map:
                        val = str(row[col_map["description"]] or "").strip()
                        if val:
                            li.description = ExtractedField(val, base_confidence, "table")
                    if "quantity" in col_map:
                        val = str(row[col_map["quantity"]] or "").strip()
                        if val:
                            li.quantity = ExtractedField(val, base_confidence, "table")
                    if "unit_price" in col_map:
                        val = str(row[col_map["unit_price"]] or "").strip()
                        if val:
                            li.unit_price = ExtractedField(val, base_confidence, "table")
                    if "amount" in col_map:
                        val = str(row[col_map["amount"]] or "").strip()
                        if val:
                            li.amount = ExtractedField(val, base_confidence, "table")
                    if "uom" in col_map:
                        val = str(row[col_map["uom"]] or "").strip()
                        if val:
                            li.unit_of_measure = ExtractedField(val, base_confidence, "table")
                    if "po_line" in col_map:
                        val = str(row[col_map["po_line"]] or "").strip()
                        if val:
                            li.po_line_number = ExtractedField(val, base_confidence, "table")

                    # Only add if at least description or amount is present
                    if li.description.value or li.amount.value:
                        items.append(li)

    return items, base_confidence


def _extract_line_items_from_text(text: str) -> list[LineItem]:
    """Fallback: extract line items from raw text using patterns."""
    items = []
    # Pattern: optional line# + description + qty + price + amount
    line_pat = re.compile(
        r"^\s*(\d+)?\s*(.{5,60}?)\s+(\d+[\.,]?\d*)\s+[\$\£\€]?([\d,]+\.?\d{0,2})\s+[\$\£\€]?([\d,]+\.?\d{0,2})\s*$",
        re.MULTILINE,
    )
    for m in line_pat.finditer(text):
        li = LineItem(line_number=len(items) + 1)
        li.description = ExtractedField(m.group(2).strip(), 65, "text_pattern")
        li.quantity = ExtractedField(m.group(3).strip(), 65, "text_pattern")
        li.unit_price = ExtractedField(m.group(4).strip(), 65, "text_pattern")
        li.amount = ExtractedField(m.group(5).strip(), 65, "text_pattern")
        items.append(li)
    return items


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_invoice(pdf_path: str) -> InvoiceData:
    """
    Main extraction function – reads a PDF invoice and returns
    structured data with per-field confidence scores.
    """
    # Step 1: Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)

    invoice = InvoiceData(raw_text=raw_text)

    if not raw_text.strip():
        # PDF might be image-only – all fields stay at 0% confidence
        return invoice

    # Step 2: Extract header fields via regex
    invoice.invoice_number = _first_match(raw_text, INVOICE_NUM_PATTERNS)
    invoice.po_number = _first_match(raw_text, PO_NUMBER_PATTERNS)
    invoice.invoice_date = _extract_invoice_date(raw_text)
    invoice.due_date = _extract_due_date(raw_text)
    invoice.currency = _extract_currency(raw_text)
    invoice.payment_terms = _first_match(raw_text, PAYMENT_TERMS_PATTERNS)

    invoice.vendor_name = _extract_vendor_name(raw_text)
    invoice.vendor_tax_id = _first_match(raw_text, TAX_ID_PATTERNS)
    invoice.vendor_address = _extract_address_block(raw_text, r"(?:from|seller|vendor|supplier|remit\s*to)")
    invoice.vendor_bank_details = _extract_address_block(raw_text, r"(?:bank\s*details?|banking\s*info|account\s*details?)")

    invoice.buyer_name = _extract_address_block(raw_text, r"(?:bill\s*to|sold\s*to|buyer|customer)")
    invoice.buyer_address = _extract_address_block(raw_text, r"(?:billing\s*address)")
    invoice.delivery_address = _extract_address_block(raw_text, r"(?:ship\s*to|deliver\s*to|delivery\s*address)")
    invoice.delivery_note_number = _first_match(raw_text, DELIVERY_NOTE_PATTERNS)

    invoice.total_amount = _first_match(raw_text, AMOUNT_PATTERNS)
    invoice.subtotal = _first_match(raw_text, SUBTOTAL_PATTERNS)
    invoice.tax_amount = _first_match(raw_text, TAX_PATTERNS)
    invoice.tax_rate = _first_match(raw_text, TAX_RATE_PATTERNS)

    # Step 3: Extract line items (prefer table extraction)
    line_items, _ = _extract_line_items_from_tables(pdf_path)
    if not line_items:
        line_items = _extract_line_items_from_text(raw_text)
    invoice.line_items = line_items

    # Step 4: Cross-validate totals for bonus confidence
    _cross_validate(invoice)

    return invoice


def _cross_validate(invoice: InvoiceData):
    """Cross-check extracted values to adjust confidence."""
    # If subtotal + tax ≈ total, boost confidence on all three
    try:
        sub = float((invoice.subtotal.value or "0").replace(",", ""))
        tax = float((invoice.tax_amount.value or "0").replace(",", ""))
        total = float((invoice.total_amount.value or "0").replace(",", ""))
        if total > 0 and abs((sub + tax) - total) < 0.02:
            invoice.subtotal.confidence = min(invoice.subtotal.confidence + 5, 99)
            invoice.tax_amount.confidence = min(invoice.tax_amount.confidence + 5, 99)
            invoice.total_amount.confidence = min(invoice.total_amount.confidence + 5, 99)
    except (ValueError, TypeError):
        pass

    # If line item amounts sum to subtotal, boost line confidence
    if invoice.line_items and invoice.subtotal.value:
        try:
            sub = float(invoice.subtotal.value.replace(",", ""))
            line_sum = sum(
                float((li.amount.value or "0").replace(",", ""))
                for li in invoice.line_items
            )
            if line_sum > 0 and abs(line_sum - sub) < 0.02:
                for li in invoice.line_items:
                    li.amount.confidence = min(li.amount.confidence + 5, 99)
        except (ValueError, TypeError):
            pass
