import os
import re
import json
import pdfplumber
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from difflib import SequenceMatcher
from itertools import zip_longest
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# ---- Load environment variables ----
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
POPLER_PATH = os.getenv("POPLER_PATH")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")

# ---- Configure Gemini API ----
if not GEMINI_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please set it in .env file.")
    st.stop()
else:
    genai.configure(api_key=GEMINI_KEY)

# ---- Configure Tesseract path ----
if TESSERACT_PATH:
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        st.warning(f"Tesseract path not found: {TESSERACT_PATH}. OCR may not work.")
else:
    st.info("TESSERACT_PATH not set. OCR fallback will use system default.")

MODEL_NAME = "gemini-2.5-flash"

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="Batch PO vs Invoice Comparator", layout="wide")
st.title("📄 Batch PO vs Invoice Comparator")
st.markdown("*Compare Purchase Orders with Invoices using AI-powered extraction*")

# ---- Helpers ----
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes, file_name):
    """Extract text from PDF with OCR fallback. Cached for performance."""
    from io import BytesIO
    
    text_content = ""
    config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'

    try:
        # Convert bytes to BytesIO for pdfplumber
        pdf_file = BytesIO(file_bytes)
        
        # Try pdfplumber first
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_content += t + "\n"

        # OCR fallback if no text
        if not text_content.strip():
            st.info(f"📷 Using OCR for {file_name} (no text layer detected)...")
            images = convert_from_bytes(
                file_bytes,
                poppler_path=POPLER_PATH if POPLER_PATH else None
            )
            for img in images:
                ocr_text = pytesseract.image_to_string(img, config=config)
                text_content += ocr_text + "\n"

        return text_content.strip()

    except Exception as e:
        st.error(f"❌ Failed to extract text from {file_name}: {str(e)}")
        return ""


def call_gemini_for_structure(text, doc_type="Document"):
    """Use Gemini to extract structured JSON fields."""
    if not text:
        return None, "Empty text content"
    
    prompt = f"""
Extract structured data from this {doc_type} text.
Return ONLY valid JSON with this exact structure:
{{
  "document_type": "string",
  "number": "string",
  "vendor": "string",
  "date": "string",
  "grand_total": "number or string",
  "items": [
    {{
      "description": "string",
      "qty": "number or string",
      "unit_price": "number or string",
      "total": "number or string"
    }}
  ]
}}

Text:
\"\"\"{text[:15000]}\"\"\"
"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in raw:
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw)
            if json_match:
                raw = json_match.group(1)
        elif "```" in raw:
            json_match = re.search(r"```\s*(\{[\s\S]*?\})\s*```", raw)
            if json_match:
                raw = json_match.group(1)
        
        # Extract JSON object
        m = re.search(r"(\{[\s\S]*\})", raw)
        json_text = m.group(1) if m else raw
        parsed = json.loads(json_text)
        
        # Validate structure
        if "items" not in parsed or not isinstance(parsed["items"], list):
            return None, "Invalid JSON structure: missing 'items' array"
        
        return parsed, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Gemini API error: {str(e)}"


def normalize_number(value):
    """Convert string numbers to float for comparison."""
    if value is None or value == "":
        return None
    try:
        # Remove currency symbols and commas
        clean = re.sub(r'[,$₹€£]', '', str(value).strip())
        return float(clean)
    except (ValueError, TypeError):
        return None


def extract_core_product_name(description):
    """Extract core product name, ignoring model numbers, codes, etc."""
    if not description:
        return ""
    
    # Remove common patterns: model numbers, HS codes, part numbers
    desc = description.lower()
    desc = re.sub(r'\b(model|part|hs code|code|no|ref)\s*:?\s*[\w-]+', '', desc)
    desc = re.sub(r'\b\d{6,}\b', '', desc)  # Remove long numbers (codes)
    desc = re.sub(r'[^\w\s]', ' ', desc)  # Remove special chars
    desc = ' '.join(desc.split())  # Normalize whitespace
    
    return desc.strip()


def calculate_item_match_score(po_item, inv_item):
    """
    Calculate comprehensive match score based on:
    - Core product description (40%)
    - Quantity match (20%)
    - Unit price match (20%)
    - Total match (20%)
    """
    score = 0.0
    
    # 1. Description similarity (40%)
    po_desc = extract_core_product_name(po_item.get("description", ""))
    inv_desc = extract_core_product_name(inv_item.get("description", ""))
    
    if po_desc and inv_desc:
        po_tokens = set(re.findall(r"\w+", po_desc))
        inv_tokens = set(re.findall(r"\w+", inv_desc))
        
        if po_tokens and inv_tokens:
            token_overlap = len(po_tokens & inv_tokens) / max(len(po_tokens | inv_tokens), 1)
            char_sim = SequenceMatcher(None, po_desc, inv_desc).ratio()
            desc_score = (token_overlap + char_sim) / 2
            score += desc_score * 0.4
    
    # 2. Quantity match (20%)
    po_qty = normalize_number(po_item.get("qty"))
    inv_qty = normalize_number(inv_item.get("qty"))
    
    if po_qty is not None and inv_qty is not None:
        if abs(po_qty - inv_qty) < 0.01:
            score += 0.2
        else:
            # Partial credit for close quantities
            qty_diff = abs(po_qty - inv_qty) / max(po_qty, inv_qty, 1)
            if qty_diff < 0.1:  # Within 10%
                score += 0.1
    
    # 3. Unit price match (20%)
    po_price = normalize_number(po_item.get("unit_price"))
    inv_price = normalize_number(inv_item.get("unit_price"))
    
    if po_price is not None and inv_price is not None:
        if abs(po_price - inv_price) < 0.01:
            score += 0.2
        else:
            # Partial credit for close prices
            price_diff = abs(po_price - inv_price) / max(po_price, inv_price, 1)
            if price_diff < 0.1:  # Within 10%
                score += 0.1
    
    # 4. Total match (20%)
    po_total = normalize_number(po_item.get("total"))
    inv_total = normalize_number(inv_item.get("total"))
    
    if po_total is not None and inv_total is not None:
        if abs(po_total - inv_total) < 0.01:
            score += 0.2
        else:
            # Partial credit for close totals
            total_diff = abs(po_total - inv_total) / max(po_total, inv_total, 1)
            if total_diff < 0.1:  # Within 10%
                score += 0.1
    
    return score


def compare_structures(po_struct, inv_struct, item_match_threshold=0.7):
    """Compare PO and Invoice items and return detailed comparison DataFrame."""
    rows = []
    po_items = po_struct.get("items", []) if po_struct else []
    inv_items = inv_struct.get("items", []) if inv_struct else []
    inv_used = set()

    for po_it in po_items:
        best_match = None
        best_score = 0.0
        best_idx = None

        # Find best matching invoice item using comprehensive scoring
        for idx, inv_it in enumerate(inv_items):
            if idx in inv_used:
                continue
            score = calculate_item_match_score(po_it, inv_it)
            if score > best_score:
                best_score = score
                best_match = inv_it
                best_idx = idx

        # Mark as used if above threshold
        if best_idx is not None and best_score >= item_match_threshold:
            inv_used.add(best_idx)

        inv_it = best_match
        
        # Extract values
        po_qty = po_it.get("qty", "")
        inv_qty = inv_it.get("qty", "") if inv_it else ""
        po_price = po_it.get("unit_price", "")
        inv_price = inv_it.get("unit_price", "") if inv_it else ""
        po_total = po_it.get("total", "")
        inv_total = inv_it.get("total", "") if inv_it else ""

        # Detailed matching checks
        qty_match = False
        price_match = False
        total_match = False
        
        if inv_it:
            po_qty_num = normalize_number(po_qty)
            inv_qty_num = normalize_number(inv_qty)
            po_price_num = normalize_number(po_price)
            inv_price_num = normalize_number(inv_price)
            po_total_num = normalize_number(po_total)
            inv_total_num = normalize_number(inv_total)
            
            if po_qty_num is not None and inv_qty_num is not None:
                qty_match = abs(po_qty_num - inv_qty_num) < 0.01
            
            if po_price_num is not None and inv_price_num is not None:
                price_match = abs(po_price_num - inv_price_num) < 0.01
            
            if po_total_num is not None and inv_total_num is not None:
                total_match = abs(po_total_num - inv_total_num) < 0.01
        
        # Determine overall status
        if best_score >= item_match_threshold:
            if qty_match and price_match and total_match:
                status = "✅ Perfect Match"
            elif total_match:
                status = "✅ Total Match"
            elif qty_match or price_match:
                status = "⚠️ Partial Match"
            else:
                status = "⚠️ Description Match Only"
        else:
            status = "❌ Mismatch"
        
        # Build match details
        match_details = []
        if inv_it:
            if qty_match:
                match_details.append("Qty✓")
            if price_match:
                match_details.append("Price✓")
            if total_match:
                match_details.append("Total✓")
        
        match_info = " | ".join(match_details) if match_details else "-"

        rows.append({
            "PO Item": po_it.get("description", ""),
            "PO Qty": po_qty,
            "PO Price": po_price,
            "PO Total": po_total,
            "Invoice Item": inv_it.get("description", "") if inv_it else "",
            "Invoice Qty": inv_qty,
            "Invoice Price": inv_price,
            "Invoice Total": inv_total,
            "Match Details": match_info,
            "Status": status,
            "Score": f"{best_score:.0%}",
        })

    # Add unmatched invoice items
    for idx, inv_it in enumerate(inv_items):
        if idx not in inv_used:
            rows.append({
                "PO Item": "",
                "PO Qty": "",
                "PO Price": "",
                "PO Total": "",
                "Invoice Item": inv_it.get("description", ""),
                "Invoice Qty": inv_it.get("qty", ""),
                "Invoice Price": inv_it.get("unit_price", ""),
                "Invoice Total": inv_it.get("total", ""),
                "Match Details": "-",
                "Status": "❌ Not in PO",
                "Score": "0%",
            })

    return pd.DataFrame(rows)


def smart_pair_documents(po_data, inv_data):
    """Intelligently pair PO and Invoice documents by number similarity."""
    pairs = []
    
    # If counts match, try smart pairing
    if len(po_data) == len(inv_data):
        inv_used = set()
        
        for po_name, po_struct, po_err in po_data:
            if po_err or not po_struct:
                pairs.append(((po_name, po_struct, po_err), (None, None, None)))
                continue
            
            po_number = str(po_struct.get("number", ""))
            best_match = None
            best_score = 0.0
            best_idx = None
            
            # Find best matching invoice
            for idx, (inv_name, inv_struct, inv_err) in enumerate(inv_data):
                if idx in inv_used or inv_err or not inv_struct:
                    continue
                
                inv_number = str(inv_struct.get("number", ""))
                
                # Calculate similarity based on number
                if po_number and inv_number:
                    score = SequenceMatcher(None, po_number.lower(), inv_number.lower()).ratio()
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_match = (inv_name, inv_struct, inv_err)
                    best_idx = idx
            
            if best_idx is not None and best_score > 0.3:
                inv_used.add(best_idx)
                pairs.append(((po_name, po_struct, po_err), best_match))
            else:
                # If no good match, try first unused invoice
                for idx, inv_item in enumerate(inv_data):
                    if idx not in inv_used:
                        inv_used.add(idx)
                        pairs.append(((po_name, po_struct, po_err), inv_item))
                        break
                else:
                    pairs.append(((po_name, po_struct, po_err), (None, None, None)))
        
        # Add any remaining unmatched invoices
        for idx, inv_item in enumerate(inv_data):
            if idx not in inv_used:
                pairs.append(((None, None, None), inv_item))
    else:
        # Different counts - just pair sequentially
        for po_item, inv_item in zip_longest(po_data, inv_data, fillvalue=(None, None, None)):
            pairs.append((po_item, inv_item))
    
    return pairs


# ---- UI ----
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Item matching threshold", 
    min_value=50, 
    max_value=95, 
    value=70,
    help="Minimum similarity score (%) to consider items as matching"
) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistics")
if 'stats' not in st.session_state:
    st.session_state.stats = {"total_comparisons": 0, "total_matches": 0}

st.sidebar.metric("Total Comparisons", st.session_state.stats["total_comparisons"])
st.sidebar.metric("Total Matches", st.session_state.stats["total_matches"])

# File uploaders
col1, col2 = st.columns(2)
with col1:
    po_files = st.file_uploader(
        "📥 Upload Purchase Orders (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PO documents"
    )

with col2:
    inv_files = st.file_uploader(
        "📥 Upload Invoices (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more Invoice documents"
    )

# Action buttons
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    compare_btn = st.button("🔍 Compare All Documents", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear All", use_container_width=True)

# Clear functionality
if clear_btn:
    st.session_state.clear()
    st.rerun()

# Compare functionality
if compare_btn:
    if not po_files or not inv_files:
        st.error("⚠️ Please upload at least one PO and one Invoice.")
    else:
        with st.spinner("🔄 Extracting and analyzing documents..."):
            # Process POs
            po_data = []
            for f in po_files:
                file_bytes = f.read()
                text = extract_text_from_pdf(file_bytes, f.name)
                parsed, err = call_gemini_for_structure(text, "Purchase Order")
                po_data.append((f.name, parsed, err))

            # Process Invoices
            inv_data = []
            for f in inv_files:
                file_bytes = f.read()
                text = extract_text_from_pdf(file_bytes, f.name)
                parsed, err = call_gemini_for_structure(text, "Invoice")
                inv_data.append((f.name, parsed, err))

        # Smart pairing
        pairs = smart_pair_documents(po_data, inv_data)
        
        st.markdown("---")
        st.subheader("📋 Comparison Results")
        
        total_matches = 0
        total_items = 0
        
        # Display results
        for idx, ((po_name, po_struct, po_err), (inv_name, inv_struct, inv_err)) in enumerate(pairs, 1):
            with st.expander(f"**Pair {idx}:** {po_name or 'N/A'} ↔ {inv_name or 'N/A'}", expanded=True):
                if po_err:
                    st.error(f"PO Error: {po_err}")
                if inv_err:
                    st.error(f"Invoice Error: {inv_err}")
                
                if not po_err and not inv_err and po_struct and inv_struct:
                    df = compare_structures(po_struct, inv_struct, item_match_threshold=threshold)
                    
                    # Color coding based on status
                    def color_status(row):
                        status = row["Status"]
                        if "Perfect Match" in status or "Total Match" in status:
                            return ['background-color: #d4edda; color: black; border: 1px solid #aaa'] * len(row)
                        elif "Partial Match" in status or "Description Match" in status:
                            return ['background-color: #fff3cd; color: black; border: 1px solid #aaa'] * len(row)
                        else:
                            return ['background-color: #f8d7da; color: black; border: 1px solid #aaa'] * len(row)
                    
                    styled_df = df.style.apply(color_status, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    perfect_matches = sum(df["Status"].str.contains("Perfect Match|Total Match", na=False))
                    partial_matches = sum(df["Status"].str.contains("Partial Match|Description Match", na=False))
                    mismatches = len(df) - perfect_matches - partial_matches
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Perfect/Total Matches", perfect_matches)
                    with col2:
                        st.metric("⚠️ Partial Matches", partial_matches)
                    with col3:
                        st.metric("❌ Mismatches", mismatches)
                    
                    total_matches += perfect_matches
                    total_items += len(df)
        
        # Update stats
        st.session_state.stats["total_comparisons"] += len(pairs)
        st.session_state.stats["total_matches"] += total_matches
        
        st.markdown("---")
        st.info(f"📊 Overall: {total_matches}/{total_items} items matched across all document pairs")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Gemini AI | Built with Streamlit")
st.caption("Supports text extraction with OCR fallback for scanned documents")