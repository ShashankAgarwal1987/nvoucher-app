from flask import Flask, request, render_template, send_file
import os, io, re, math
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Side, Font, PatternFill
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload cap

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

# ---------- Utility ----------
def embed_batch(texts, batch_size=64):
    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vecs.extend([d.embedding for d in resp.data])
    return vecs

def cos(a, b):
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def parse_itinerary(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    days = []
    for line in lines:
        parts = re.split(r"Day\s*\d+\s*:\s*", line, flags=re.IGNORECASE)
        payload = parts[1] if len(parts) > 1 else parts[0]
        acts = [a.strip() for a in payload.split("+") if a.strip()]
        days.append(acts)
    return days

def detect_master_cols(df: pd.DataFrame):
    norm = {c.lower().strip(): c for c in df.columns}
    if "particular" not in norm or "formatted output" not in norm:
        raise ValueError("Master must have columns: 'Particular' and 'Formatted Output'.")
    col_p = norm["particular"]
    col_f = norm["formatted output"]
    col_d = norm.get("tour description")
    return col_p, col_f, col_d

def shortlist_indices(activity_text, key_embs, key_texts, top_k=8):
    qv = embed_batch([activity_text])[0]
    scores = [(i, cos(qv, kv)) for i, kv in enumerate(key_embs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:top_k]]

def gpt_pick_formatted(activity, candidates):
    """
    candidates: list of {"particular": "...", "formatted": "..."}
    Return: EXACT 'formatted' text selected by GPT (or a safe fallback)
    """
    sys = ("You are a travel operations assistant. "
           "Choose the BEST match from the options for the given activity. "
           "Return ONLY the 'formatted' field of the chosen option. No extra text.")
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":f"Activity: {activity}\n\nOptions:\n{candidates}\n\nReturn ONLY the exact 'formatted' text."}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return "⚠️ OpenAI rate limit/quota reached."
    except AuthenticationError:
        return "⚠️ Invalid OPENAI_API_KEY."
    except APIError as e:
        return f"⚠️ OpenAI API error: {e}"

# ---------- Template helpers ----------
def find_service_table_anchor(ws):
    """
    Auto-detect the header row that contains Date/Service headers.
    Returns (header_row, col_date, col_service).
    Fallback: returns (None, 1, 2) => write from row 15, columns A/B.
    """
    header_row = None
    col_date = col_service = None
    for r in range(1, 100):
        row_vals = [str(ws.cell(r, c).value).strip().lower() if ws.cell(r, c).value else "" for c in range(1, 20)]
        # look for 'date' and 'service' (or 'service details')
        try:
            c_date = next(i+1 for i, v in enumerate(row_vals) if v in {"date", "day/date"})
        except StopIteration:
            c_date = None
        try:
            c_serv = next(i+1 for i, v in enumerate(row_vals) if v in {"service", "service details", "services"})
        except StopIteration:
            c_serv = None
        if c_date and c_serv:
            header_row = r
            col_date, col_service = c_date, c_serv
            break
    if header_row is None:
        return None, 1, 2
    return header_row, col_date, col_service

def copy_cell_style(src, dst):
    if src.has_style:
        dst.font = Font(name=src.font.name, size=src.font.size, bold=src.font.bold,
                        italic=src.font.italic, vertAlign=src.font.vertAlign,
                        underline=src.font.underline, strike=src.font.strike,
                        color=src.font.color)
        dst.fill = PatternFill(fill_type=src.fill.fill_type,
                               start_color=getattr(src.fill, "start_color", None),
                               end_color=getattr(src.fill, "end_color", None))
        if src.border:
            dst.border = Border(left=src.border.left, right=src.border.right,
                                top=src.border.top, bottom=src.border.bottom,
                                diagonal=src.border.diagonal)
        if src.alignment:
            dst.alignment = Alignment(horizontal=src.alignment.horizontal,
                                      vertical=src.alignment.vertical,
                                      wrap_text=src.alignment.wrap_text)

def inject_booking_header(ws, meta):
    """
    Optional: if your template has labels like 'Confirmation ID:', 'Passenger Name:', etc.,
    we fill the cell to the right of the label.
    If not found, we silently skip.
    """
    labels = {
        "confirmation id": meta.get("confirmation_id"),
        "passenger name": meta.get("passenger_name"),
        "guest name": meta.get("passenger_name"),
        "travel dates": meta.get("travel_dates"),
        "destination": meta.get("destination")
    }
    for r in range(1, 40):
        for c in range(1, 10):
            v = ws.cell(r, c).value
            if not isinstance(v, str):
                continue
            key = v.strip().lower().rstrip(":")
            if key in labels and labels[key]:
                ws.cell(r, c+1, labels[key])

def write_services_into_template(template_bytes, rows, meta):
    """
    rows: list of {"Date": "...", "Formatted": "..."} or {"Day": "...","Date": "...", "Formatted": "..."}
    meta: optional booking header items to inject (confirmation_id, passenger_name, travel_dates, destination)
    """
    wb = load_workbook(io.BytesIO(template_bytes))
    ws = wb.active  # use first sheet

    # Try to fill header labels if present
    inject_booking_header(ws, meta)

    # Find service table anchor
    header_row, col_date, col_service = find_service_table_anchor(ws)
    start_row = (header_row + 1) if header_row else 15

    # Detect a style row to clone (first existing data row under header, else header itself)
    style_src_row = start_row
    if ws.cell(style_src_row, col_date).value is None and ws.cell(style_src_row, col_service).value is None:
        # If empty, use header row styles as fallback
        style_src_row = header_row if header_row else start_row

    # Clear existing data below header up to a reasonable limit (optional)
    for r in range(start_row, start_row + 600):
        ws.cell(r, col_date).value = None
        ws.cell(r, col_service).value = None

    # Write new rows
    cur = start_row
    for item in rows:
        date_text = item.get("Date") or item.get("Day") or ""
        serv_text = item.get("Formatted") or item.get("Service") or ""

        # Set values
        ws.cell(cur, col_date, date_text)
        ws.cell(cur, col_service, serv_text)

        # Copy styles from style_src_row
        copy_cell_style(ws.cell(style_src_row, col_date), ws.cell(cur, col_date))
        copy_cell_style(ws.cell(style_src_row, col_service), ws.cell(cur, col_service))

        # Ensure wrapping & vertical alignment
        ws.cell(cur, col_service).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(cur, col_date).alignment    = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Keep a sensible row height for multi-line services
        ws.row_dimensions[cur].height = ws.row_dimensions[style_src_row].height or 28

        cur += 1

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out

# ---------- Web ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # Files
    template_file = request.files.get("template_file")   # your voucher .xlsx (converted from .xls)
    master_file   = request.files.get("master_file")
    if not template_file or not master_file:
        return "⚠️ Please upload both Template and Master files.", 400

    # Inputs
    start_date_str = request.form.get("start_date", "").strip()
    itinerary_text = request.form.get("itinerary", "").strip()
    confirmation_id = request.form.get("confirmation_id", "").strip()
    passenger_name  = request.form.get("passenger_name", "").strip()
    destination     = request.form.get("destination", "").strip()

    if not itinerary_text:
        return "⚠️ Please provide the itinerary text.", 400

    # Build meta header string like "01-Aug-2025 to 05-Aug-2025"
    travel_dates_text = ""
    try:
        if start_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        else:
            start_date = None
        day_plans = parse_itinerary(itinerary_text)
        if start_date:
            end_date = start_date + timedelta(days=len(day_plans)-1 if day_plans else 0)
            travel_dates_text = f"{start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}"
    except Exception:
        start_date = None
        day_plans = parse_itinerary(itinerary_text)

    # Load Master
    try:
        master_df = pd.read_excel(
            master_file,
            usecols=lambda c: c.lower().strip() in {"particular","formatted output","tour description"},
            engine="openpyxl"
        )
        # Limit for free tier
        MAX_ROWS = 800
        if len(master_df) > MAX_ROWS:
            master_df = master_df.iloc[:MAX_ROWS].copy()

        # Normalize text
        for col in master_df.columns:
            if master_df[col].dtype == object:
                master_df[col] = master_df[col].astype(str).str.slice(0, 800)

        col_p, col_f, col_d = detect_master_cols(master_df)
        # Build key text for embeddings
        keys_text = (
            master_df[col_p].astype(str) + 
            ((" | " + master_df[col_d].fillna("").astype(str)) if col_d else "")
        ).tolist()
        key_embs = embed_batch(keys_text)
    except Exception as e:
        return f"❌ Master file error: {e}", 400

    # Build AI-matched rows
    rows = []
    for day_idx, acts in enumerate(day_plans, start=1):
        # Date or just Day X if no start date
        date_text = (start_date + timedelta(days=day_idx-1)).strftime("%d-%b-%Y") if start_date else f"Day {day_idx}"
        blocks = []
        for act in acts:
            try:
                top_ids = shortlist_indices(act, key_embs, keys_text, top_k=8)
                cands = [{"particular": str(master_df.iloc[i][col_p]),
                          "formatted":  str(master_df.iloc[i][col_f])} for i in top_ids]
                choice = gpt_pick_formatted(act, cands)
                # Safety fallback
                if choice not in [c["formatted"] for c in cands]:
                    choice = cands[0]["formatted"]
                blocks.append(choice)
            except Exception as e:
                blocks.append(f"⚠️ Could not map: {act} (Error: {e})")
        rows.append({"Date": date_text, "Formatted": "\n\n".join(blocks)})

    # Inject into your template
    meta = {
        "confirmation_id": confirmation_id,
        "passenger_name": passenger_name,
        "travel_dates": travel_dates_text,
        "destination": destination
    }
    try:
        out = write_services_into_template(template_file.read(), rows, meta)
    except Exception as e:
        return f"❌ Failed to write into template: {e}", 500

    return send_file(
        out,
        as_attachment=True,
        download_name="Voucher_Output.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    # Local dev only; on Render we use gunicorn
    app.run(host="0.0.0.0", port=5000)
