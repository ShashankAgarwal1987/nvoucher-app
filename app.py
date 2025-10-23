import os
import io
import pandas as pd
from flask import Flask, request, render_template, send_file
from openai import OpenAI
from datetime import datetime, timedelta

# Init
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache for master data
master_data = None

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return sum(a*b for a, b in zip(vec1, vec2)) / (
        (sum(a*a for a in vec1) ** 0.5) * (sum(b*b for b in vec2) ** 0.5)
    )

def match_input_to_master(user_input, master_df):
    user_emb = get_embedding(user_input)
    best_score, best_row = -1, None
    for _, row in master_df.iterrows():
        master_text = str(row["Particular"]) + " " + str(row["Tour Description"])
        master_emb = get_embedding(master_text)
        score = cosine_similarity(user_emb, master_emb)
        if score > best_score:
            best_score, best_row = score, row
    return best_row["Formatted Output"] if best_row is not None else "No Match Found"

@app.route("/", methods=["GET", "POST"])
def index():
    global master_data
    if request.method == "POST":
        if "master_file" in request.files and request.files["master_file"].filename:
            file = request.files["master_file"]
            master_data = pd.read_excel(file)

        start_date = request.form.get("start_date")
        itinerary_text = request.form.get("itinerary")

        if not master_data is None and start_date and itinerary_text:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            results = []
            for i, line in enumerate(itinerary_text.split("\n")):
                line = line.strip()
                if not line:
                    continue
                day_date = start_date + timedelta(days=i)
                formatted = match_input_to_master(line, master_data)
                results.append({"Date": day_date.strftime("%d-%b-%Y"), "Service": formatted})

            df_out = pd.DataFrame(results)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Voucher")
            output.seek(0)
            return send_file(output, download_name="service_voucher.xlsx", as_attachment=True)

    return render_template("index.html")
