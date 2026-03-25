"""
Invoice AP Reader – Streamlit Application
Extracts invoice data for ERP three-way match with confidence scoring.
"""

import tempfile
import os
import io
import json

import streamlit as st
import pandas as pd

from invoice_extractor import extract_invoice
from models import InvoiceData


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Invoice AP Reader",
    page_icon="📄",
    layout="wide",
)

st.title("Invoice AP Reader")
st.markdown("Upload a PDF invoice to extract data for **three-way match** (Invoice ↔ PO ↔ Goods Receipt). "
            "Each field shows a **confidence %** so you can review accuracy before importing into your ERP.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Flag fields below confidence %", 0, 100, 70,
    help="Fields below this threshold will be highlighted for manual review.",
)
export_format = st.sidebar.selectbox("Export format", ["JSON", "CSV", "Excel"])

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Invoice PDF", type=["pdf"], accept_multiple_files=False)

if uploaded_file is not None:
    # Save to temp file for pdfplumber
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Extracting invoice data..."):
            invoice: InvoiceData = extract_invoice(tmp_path)

        # ---------------------------------------------------------------
        # Overall confidence banner
        # ---------------------------------------------------------------
        overall = invoice.overall_confidence
        if overall >= 85:
            st.success(f"Overall extraction confidence: **{overall}%** — High")
        elif overall >= 65:
            st.warning(f"Overall extraction confidence: **{overall}%** — Medium (review flagged fields)")
        else:
            st.error(f"Overall extraction confidence: **{overall}%** — Low (manual review recommended)")

        # ---------------------------------------------------------------
        # Three-way match critical fields
        # ---------------------------------------------------------------
        st.header("Three-Way Match Fields")
        critical_fields = {
            "Invoice Number": invoice.invoice_number,
            "PO Number": invoice.po_number,
            "Invoice Date": invoice.invoice_date,
            "Total Amount": invoice.total_amount,
            "Vendor Name": invoice.vendor_name,
            "Currency": invoice.currency,
        }

        cols = st.columns(3)
        for i, (label, field) in enumerate(critical_fields.items()):
            with cols[i % 3]:
                val = field.value or "—"
                conf = field.confidence
                if conf >= 85:
                    badge = f"🟢 {conf}%"
                elif conf >= confidence_threshold:
                    badge = f"🟡 {conf}%"
                else:
                    badge = f"🔴 {conf}%"
                st.metric(label=f"{label}  {badge}", value=val)

        # ---------------------------------------------------------------
        # All header fields detail table
        # ---------------------------------------------------------------
        st.header("All Extracted Fields")
        all_fields = invoice.get_all_fields()
        rows = []
        for fname, fdata in all_fields.items():
            flag = "⚠️ REVIEW" if fdata["confidence_pct"] < confidence_threshold else "✅"
            rows.append({
                "Field": fname.replace("_", " ").title(),
                "Value": fdata["value"] or "—",
                "Confidence %": fdata["confidence_pct"],
                "Level": fdata["confidence_label"],
                "Status": flag,
            })

        df_fields = pd.DataFrame(rows)

        def highlight_low_confidence(row):
            if row["Confidence %"] < confidence_threshold:
                return ["background-color: #ffe0e0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_fields.style.apply(highlight_low_confidence, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------------
        # Line items
        # ---------------------------------------------------------------
        if invoice.line_items:
            st.header(f"Line Items ({len(invoice.line_items)})")
            li_rows = []
            for li in invoice.line_items:
                li_rows.append({
                    "#": li.line_number,
                    "Description": li.description.value or "—",
                    "Desc Conf%": li.description.confidence,
                    "Qty": li.quantity.value or "—",
                    "Qty Conf%": li.quantity.confidence,
                    "Unit Price": li.unit_price.value or "—",
                    "Price Conf%": li.unit_price.confidence,
                    "Amount": li.amount.value or "—",
                    "Amt Conf%": li.amount.confidence,
                })

            df_lines = pd.DataFrame(li_rows)

            def highlight_line_confidence(row):
                styles = []
                for col in row.index:
                    if "Conf%" in col and isinstance(row[col], (int, float)) and row[col] < confidence_threshold:
                        styles.append("background-color: #ffe0e0")
                    else:
                        styles.append("")
                return styles

            st.dataframe(
                df_lines.style.apply(highlight_line_confidence, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No line items detected. The invoice may not contain a structured table.")

        # ---------------------------------------------------------------
        # Export section
        # ---------------------------------------------------------------
        st.header("Export for ERP Import")

        if export_format == "JSON":
            json_data = invoice.to_erp_json()
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="invoice_extract.json",
                mime="application/json",
            )
            with st.expander("Preview JSON"):
                st.json(json.loads(json_data))

        elif export_format == "CSV":
            flat = invoice.to_erp_flat()
            df_export = pd.DataFrame(flat)
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="invoice_extract.csv",
                mime="text/csv",
            )
            with st.expander("Preview CSV"):
                st.dataframe(df_export, use_container_width=True, hide_index=True)

        elif export_format == "Excel":
            flat = invoice.to_erp_flat()
            df_export = pd.DataFrame(flat)
            buffer = io.BytesIO()
            df_export.to_excel(buffer, index=False, engine="openpyxl")
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name="invoice_extract.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # ---------------------------------------------------------------
        # Raw text (debug)
        # ---------------------------------------------------------------
        with st.expander("Raw Extracted Text"):
            st.text(invoice.raw_text or "(No text extracted – PDF may be image-only)")

    finally:
        os.unlink(tmp_path)

else:
    st.info("Upload a PDF invoice to get started.")

    st.markdown("""
    ### What this app extracts (Three-Way Match)

    | Category | Fields |
    |----------|--------|
    | **Invoice Header** | Invoice Number, Date, Due Date, Currency, Payment Terms |
    | **PO Reference** | Purchase Order Number (critical for matching) |
    | **Vendor Details** | Name, Address, Tax ID, Bank Details |
    | **Buyer Details** | Name, Address |
    | **Delivery / GRN** | Delivery Address, Delivery Note / GRN Number |
    | **Totals** | Subtotal, Tax Rate, Tax Amount, Total Amount |
    | **Line Items** | Description, Quantity, Unit Price, Amount, UOM |

    ### Confidence Scoring
    - 🟢 **HIGH (≥85%)** — Field extracted with strong pattern match
    - 🟡 **MEDIUM (70-84%)** — Reasonable match, worth verifying
    - 🔴 **LOW (<70%)** — Flagged for manual review before ERP import
    """)
