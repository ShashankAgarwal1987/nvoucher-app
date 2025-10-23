import os
import io
import re
import math
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, request, render_template, send_file
from openai import OpenAI

app = Flask(__name__)

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"   # fast + cheap, good for classification/selection

# ---------- Utils ----------
def cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return num / den if den else 0.0

def embed_texts(texts):
    # returns list of vectors
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def parse_itinerary(itinerary_text: str):
    """
    Input example lines:
      Day 1: Arrival Cairo + Airport Transfer
      Day 2: Pyramids Day Tour + Sleeper train Cairo to Aswan
    Returns list of list of activities per day.
    """
    lines = [ln.strip() for ln in itinerary_text.splitlines() if ln.strip()]
    days = []
    for line in lines:
        parts = re.split(r"Day\s*\d+\s*:\s*", line, flags=re.IGNORECASE)
        payload = parts[1] if len(parts) > 1 else parts[0]
        acts = [a.strip() for a in payload.split("+") if a.strip()]
        days.append(acts)
    return days

def detect_columns(master_df: pd.DataFrame):
    """
    Flexible column detection for master file.
    Needs:
      - 'Particular'
      - 'Formatted Output'
    (Optional:)
      - 'Tour Description' (if present, we concat with Particular for better recall)
    """
    norm = {c.lower().strip(): c for c in master_df.columns}
    if "particular" not in norm or "formatted output" not in norm:
        raise ValueError("Master must contain columns: 'Particular' and 'Formatted Output'.")

    col_particular = norm["particular"]
    col_formatted  = norm["formatted output"]
    col_desc = norm.get("tour description", None)

    return col_particular, col_formatted, col_desc

def shortlist_with_embeddings(query_text, master_keys, master_embs, top_k=8):
    q_emb = embed_texts([query_text])[0]
    scores = [(i, cosine(q_emb, e)) for i, e in enumerate(master_embs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:top_k]]

def ask_gpt_to_pick(activity, candidates):
    """
    candidates: list of dicts like:
      {"id": idx, "particular": "...", "formatted": "..."}
    We instruct GPT to pick ONE and return EXACTLY its 'formatted' text.
    """
    sys = (
        "You match a customer itinerary activity to the BEST service from the provided list. "
        "Return EXACTLY the 'formatted' field of the chosen option with no extra text."
    )

    user = {
        "activity": activity,
        "options": candidates
    }

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":f"Activity to match:\n{activity}\n\nOptions:\n{candidates}\n\nReturn ONLY the exact 'formatted' text of the best option."}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ---------- Route ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # Inputs
    master_file = request.files.get("master_file")
    start_date_str = request.form.get("start_date", "").strip()
    itinerary_text = request.form.get("itinerary", "").strip()

    if not master_file:
        return "⚠️ Please upload the Master Excel.", 400
    if not start_date_str or not itinerary_text:
        return "⚠️ Please provide Start Date and Itinerary.", 400

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except Exception:
        return "⚠️ Start Date must be in YYYY-MM-DD format (use the date picker).", 400

    # Read master
    try:
        master_df = pd.read_excel(master_file)
        col_particular, col_formatted, col_desc = detect_columns(master_df)
        # Build keys text for embeddings
        if col_desc:
            keys_text = (master_df[col_particular].astype(str) + " | " +
                         master_df[col_desc].fillna("").astype(str)).tolist()
        else:
            keys_text = master_df[col_particular].astype(str).tolist()

        # Pre-embed master once
        master_embs = embed_texts(keys_text)
    except Exception as e:
        return f"❌ Master file error: {e}", 400

    # Parse itinerary
    day_plans = parse_itinerary(itinerary_text)
    if not day_plans:
        return "⚠️ Could not parse any day lines. Use lines like 'Day 1: ...'", 400

    # Build output rows
    rows = []
    for day_idx, activities in enumerate(day_plans, start=1):
        day_date = (start_date + timedelta(days=day_idx-1)).strftime("%d-%b-%Y")
        formatted_blocks = []

        for act in activities:
            try:
                # Shortlist top K via embeddings
                top_idx = shortlist_with_embeddings(act, keys_text, master_embs, top_k=8)
                candidates = []
                for i in top_idx:
                    candidates.append({
                        "id": int(i),
                        "particular": str(master_df.iloc[i][col_particular]),
                        "formatted":  str(master_df.iloc[i][col_formatted])
                    })

                # Ask GPT to pick ONE and return EXACT formatted text
                picked = ask_gpt_to_pick(act, candidates)

                # Safety: if GPT returns something not in candidates, fallback to the top-1 formatted
                if picked not in [c["formatted"] for c in candidates]:
                    picked = candidates[0]["formatted"]

                formatted_blocks.append(picked)
            except Exception as e:
                formatted_blocks.append(f"⚠️ Could not map: {act}  (Error: {e})")

        rows.append({
            "Day": f"Day {day_idx}",
            "Date": day_date,
            "Formatted Output": "\n\n".join(formatted_blocks)
        })

    # Export Excel
    out = io.BytesIO()
    df_out = pd.DataFrame(rows, columns=["Day","Date","Formatted Output"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Voucher")
    out.seek(0)

    return send_file(
        out,
        as_attachment=True,
        download_name="Voucher_Output.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
