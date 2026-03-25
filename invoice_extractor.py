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
    # Broader: "Invoice" followed by an alphanumeric code on the same line
    (re.compile(r"invoice[.:;\s]+([A-Z0-9][\w\-/]{2,20})", re.I), 75),
    # Reference number patterns
    (re.compile(r"(?:ref(?:erence)?\s*(?:no|number|#)?[.:;\s]*)\s*([A-Z0-9][\w\-/]{2,20})", re.I), 70),
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
    # Try labelled patterns with increasing flexibility
    label_patterns = [
        re.compile(r"(?:due\s*date|payment\s*due|pay\s*by)[.:;\s]*(.{6,30})", re.I),
        # "Due Date" as a standalone label followed by a date on same or next line
        re.compile(r"due\s*date\s*[.:;\s]*(.{6,30})", re.I),
        # "Due" near a date
        re.compile(r"(?:^|\s)due[.:;\s]+(.{6,20})", re.I | re.MULTILINE),
    ]
    for label_pat in label_patterns:
        m = label_pat.search(text)
        if m:
            snippet = m.group(1)
            for pat, conf in DATE_PATTERNS:
                dm = pat.search(snippet)
                if dm:
                    return ExtractedField(value=dm.group(1).strip(), confidence=conf, source="regex_labelled")

    # Broader fallback: scan entire text for "due date" near any date
    due_date_broad = re.compile(r"due\s*date\s*[^A-Za-z]*?(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", re.I)
    m = due_date_broad.search(text)
    if m:
        return ExtractedField(value=m.group(1).strip(), confidence=82, source="regex_broad")

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
    """Extract vendor name – prefer labelled fields, fall back to heuristic."""
    # Tier 1: Look for explicit labels
    label_patterns = [
        (re.compile(r"(?:vendor|supplier|seller|from|company)\s*(?:name)?[.:;\s]+(.+)", re.I), 88),
        (re.compile(r"(?:bill\s*from|invoice\s*from)[.:;\s]+(.+)", re.I), 88),
        (re.compile(r"(?:remit\s*to)[.:;\s]+(.+)", re.I), 85),
    ]
    for pat, conf in label_patterns:
        m = pat.search(text)
        if m:
            val = m.group(1).strip().split("\n")[0].strip()
            # Skip if it looks like a date or number
            if val and not re.match(r"^[\d/\-]+$", val) and not re.match(r"^date", val, re.I):
                return ExtractedField(value=val, confidence=conf, source="regex_labelled")

    # Tier 2: Heuristic – vendor name is typically at the very top
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:10]:
        if len(line) < 3:
            continue
        # Skip lines that look like headers, dates, numbers, or common labels
        if re.match(r"^(invoice|tax\s*invoice|bill|statement|page\s*\d|date|po\b|purchase)", line, re.I):
            continue
        if re.match(r"^\d", line):
            continue
        # Skip lines that look like dates (e.g. "Date: 2/28/2026")
        if re.match(r"^date\s*[.:;\s]", line, re.I):
            continue
        if re.search(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$", line):
            continue
        # Skip lines that are mostly numbers/symbols
        alpha_chars = sum(1 for c in line if c.isalpha())
        if alpha_chars < 3:
            continue
        return ExtractedField(value=line, confidence=65, source="heuristic_position")
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

    # Column header keyword sets (order matters for disambiguation)
    COL_KEYWORDS = {
        "description": ["desc", "item", "particular", "product", "service", "name"],
        "quantity": ["qty", "quantity", "qnty", "units", "count"],
        "unit_price": ["unit price", "rate", "price per", "unit cost", "unit rate", "cost"],
        "amount": ["amount", "line total", "ext", "extended", "net amount", "line amount"],
        "uom": ["uom", "unit of measure"],
        "po_line": ["po line", "po_line", "po #"],
        "tax_rate": ["tax %", "tax rate", "vat %", "gst %"],
        "tax_amount": ["tax amount", "tax", "vat", "gst"],
    }

    def _map_columns(header_row: list[str]) -> dict:
        """Map column indices to field names with smarter matching."""
        col_map = {}
        used_indices = set()

        # First pass: exact/substring matching, longest keyword first
        for field_name, keywords in COL_KEYWORDS.items():
            # Sort keywords by length descending so "unit price" matches before "price"
            for kw in sorted(keywords, key=len, reverse=True):
                for ci, col in enumerate(header_row):
                    if ci in used_indices:
                        continue
                    if kw in col:
                        col_map[field_name] = ci
                        used_indices.add(ci)
                        break
                if field_name in col_map:
                    break

        # Handle "unit" ambiguity: if "unit" matched for uom but not unit_price,
        # and there's a "price" column, prefer "unit" as unit_price
        if "unit_price" not in col_map and "uom" in col_map:
            uom_col = header_row[col_map["uom"]]
            if "price" in uom_col or "cost" in uom_col or "rate" in uom_col:
                col_map["unit_price"] = col_map.pop("uom")

        # Handle "total" ambiguity: "total" alone maps to amount, not grand total
        if "amount" not in col_map:
            for ci, col in enumerate(header_row):
                if ci in used_indices:
                    continue
                if "total" in col and "sub" not in col and "grand" not in col:
                    col_map["amount"] = ci
                    used_indices.add(ci)
                    break

        return col_map

    def _is_number(val: str) -> bool:
        """Check if a string looks like a number (with optional currency/commas)."""
        cleaned = re.sub(r"[\$\£\€\¥,\s]", "", val)
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _clean_number(val: str) -> str:
        """Strip currency symbols for cleaner output."""
        return re.sub(r"[\$\£\€\¥]", "", val).strip()

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

                col_map = _map_columns(header_row)

                # If we couldn't map description, try positional heuristic:
                # first text column is likely description
                if "description" not in col_map and len(header_row) >= 3:
                    for ci, col in enumerate(header_row):
                        if ci not in col_map.values() and col and not any(
                            kw in col for kw in ["#", "no", "sr", "sl"]
                        ):
                            col_map["description"] = ci
                            break

                # If we still have no amount, try the last numeric column
                if "amount" not in col_map:
                    for ci in range(len(header_row) - 1, -1, -1):
                        if ci not in col_map.values():
                            col_map["amount"] = ci
                            break

                # Extract data rows
                for row in table[header_idx + 1:]:
                    if not row or all(not str(c or "").strip() for c in row):
                        continue

                    # Skip summary rows
                    row_text = " ".join(str(c or "") for c in row).lower()
                    if any(kw in row_text for kw in ["total", "subtotal", "sub-total", "grand"]):
                        continue
                    # Skip rows that look like due dates or metadata
                    if re.search(r"due\s*date|payment\s*due|pay\s*by|terms", row_text, re.I):
                        continue

                    li = LineItem(line_number=len(items) + 1)

                    def _safe_get(r, idx):
                        if idx is not None and idx < len(r):
                            return str(r[idx] or "").strip()
                        return ""

                    if "description" in col_map:
                        val = _safe_get(row, col_map["description"])
                        if val:
                            li.description = ExtractedField(val, base_confidence, "table")
                    if "quantity" in col_map:
                        val = _safe_get(row, col_map["quantity"])
                        if val:
                            li.quantity = ExtractedField(_clean_number(val), base_confidence, "table")
                    if "unit_price" in col_map:
                        val = _safe_get(row, col_map["unit_price"])
                        if val:
                            li.unit_price = ExtractedField(_clean_number(val), base_confidence, "table")
                    if "amount" in col_map:
                        val = _safe_get(row, col_map["amount"])
                        if val:
                            li.amount = ExtractedField(_clean_number(val), base_confidence, "table")
                    if "uom" in col_map:
                        val = _safe_get(row, col_map["uom"])
                        if val:
                            li.unit_of_measure = ExtractedField(val, base_confidence, "table")
                    if "po_line" in col_map:
                        val = _safe_get(row, col_map["po_line"])
                        if val:
                            li.po_line_number = ExtractedField(val, base_confidence, "table")

                    # Sanity check: if "description" looks like it has embedded numbers
                    # (e.g. all data crammed into one cell), try to split it
                    if li.description.value and not li.quantity.value and not li.amount.value:
                        desc_val = li.description.value
                        # Pattern: description followed by qty, price, amount
                        split_m = re.match(
                            r"(.+?)\s+(\d+[\.,]?\d*)\s+[\$\£\€\¥]?([\d,]+\.?\d{0,2})\s+[\$\£\€\¥]?([\d,]+\.?\d{0,2})\s*$",
                            desc_val,
                        )
                        if split_m:
                            li.description = ExtractedField(split_m.group(1).strip(), 70, "table_split")
                            li.quantity = ExtractedField(split_m.group(2).strip(), 70, "table_split")
                            li.unit_price = ExtractedField(split_m.group(3).strip(), 70, "table_split")
                            li.amount = ExtractedField(split_m.group(4).strip(), 70, "table_split")
                        else:
                            # Try: description + amount only
                            split_m2 = re.match(
                                r"(.+?)\s+[\$\£\€\¥]?([\d,]+\.?\d{0,2})\s*$",
                                desc_val,
                            )
                            if split_m2 and len(split_m2.group(1)) > 3:
                                li.description = ExtractedField(split_m2.group(1).strip(), 70, "table_split")
                                li.amount = ExtractedField(split_m2.group(2).strip(), 70, "table_split")

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

    # Step 3b: If due date is still missing, scan the full text more broadly
    if not invoice.due_date.value:
        # Sometimes "Due Date" appears in table rows or footers
        due_m = re.search(r"due\s*date\s*[^A-Za-z]*?(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", raw_text, re.I)
        if due_m:
            invoice.due_date = ExtractedField(value=due_m.group(1).strip(), confidence=78, source="regex_broad")

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
