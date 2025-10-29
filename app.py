from flask import Flask, request, render_template, send_file
import os
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload cap

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

# ---------- Helpers ----------
def normalize_cols(df: pd.DataFrame):
    """Map canonical names to actual column names (case/space tolerant)."""
    name_map = {c.lower().strip(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in name_map: 
                return name_map[o]
        return None
    col_particular = pick("particular")
    col_formatted  = pick("formatted output", "formatted_output", "formatted")
    col_desc       = pick("tour description", "tour_description", "description")
    if not col_particular or not col_formatted:
        raise ValueError("Master must have columns: 'Particular' and 'Formatted Output'.")
    return col_particular, col_formatted, col_desc

def embed_batch(texts, batch=64):
    """Embed texts in small batches for stability on free tier."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float32)

def cosine_sim_matrix(A, b):
    """Cosine similarity between matrix A (n,d) and vector b (d,)."""
    b = b.astype(np.float32)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(An, bn)

def clean_itinerary(text: str):
    """Split multi-day itinerary into list of list of activities (handles + and Day X:)."""
    days = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip Day X:
        if line.lower().startswith("day"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                line = parts[1].strip()
        # split by +
        acts = [a.strip() for a in line.split("+") if a.strip()]
        days.append(acts)
    return days

# ---------- Web ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # Validate inputs
    if "file" not in request.files:
        return "⚠️ No file uploaded", 400
    master_file = request.files["file"]
    if not master_file or master_file.filename == "":
        return "⚠️ Empty file", 400

    itinerary_text = (request.form.get("itinerary") or "").strip()
    if not itinerary_text:
        return "⚠️ Please paste a day-wise itinerary", 400
    start_date_str = (request.form.get("start_date") or "").strip()

    # Parse itinerary days
    days = clean_itinerary(itinerary_text)
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
    except Exception:
        start_date = None

    # Read master (keep memory small)
    try:
        df = pd.read_excel(master_file, engine="openpyxl")
    except Exception as e:
        return f"❌ Could not read Excel: {e}", 400

    try:
        col_p, col_f, col_d = normalize_cols(df)
    except ValueError as e:
        return f"❌ {e}", 400

    # Trim and cap for free tier
    MAX_ROWS = 1000
    if len(df) > MAX_ROWS:
        df = df.iloc[:MAX_ROWS].copy()

    for c in [col_p, col_f, col_d] if col_d else [col_p, col_f]:
        if c and df[c].dtype == object:
            df[c] = df[c].astype(str).str.slice(0, 800)

    # Build keys to embed (Particular + optional Description)
    keys = df[col_p].astype(str).tolist()
    if col_d:
        keys = [(p + " | " + (d or "")) for p, d in zip(df[col_p].astype(str), df[col_d].fillna("").astype(str))]

    try:
        key_vecs = embed_batch(keys, batch=64)
    except (AuthenticationError, RateLimitError, APIError) as e:
        return f"❌ OpenAI error while embedding master: {e}", 500

    # Match each activity to best Formatted Output
    rows = []  # list of {"Date":..., "Formatted":...}
    for i, activities in enumerate(days, start=1):
        block_texts = []
        for act in activities:
            try:
                qv = embed_batch([act])[0]
                sims = cosine_sim_matrix(key_vecs, qv)
                idx = int(np.argmax(sims))
                formatted = str(df.iloc[idx][col_f])
            except Exception as e:
                formatted = f"⚠️ Could not map: {act} (Error: {e})"
            block_texts.append(formatted)

        date_str = (start_date + timedelta(days=i-1)).strftime("%d-%b-%Y") if start_date else f"Day {i}"
        rows.append({"Date": date_str, "Formatted Output": "\n\n".join(block_texts)})

    # Send as Excel
    out = BytesIO()
    pd.DataFrame(rows).to_excel(out, index=False, sheet_name="Voucher", engine="openpyxl")
    out.seek(0)
    return send_file(out, download_name="voucher_output.xlsx", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
