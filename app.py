from flask import Flask, request, render_template, send_file
import pandas as pd
from openai import OpenAI
import os
from io import BytesIO
from docx import Document
from datetime import datetime, timedelta
from configs import COUNTRY_CONFIGS
import numpy as np

app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """Get OpenAI embedding for text."""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def match_service(text, master_df):
    """
    Match service text against Master Excel using OpenAI embeddings.
    Returns closest match's formatted output.
    """
    try:
        input_embedding = get_embedding(text)
        best_score, best_row = -1, None

        for _, row in master_df.iterrows():
            ref_text = str(row["Particular"])
            ref_embedding = get_embedding(ref_text)
            score = cosine_similarity(input_embedding, ref_embedding)

            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None:
            return best_row["Formatted Output"]

        return f"⚠️ Could not match service: {text}"

    except Exception as e:
        return f"Error processing {text}: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        destination = request.form.get("destination", "Egypt")
        start_date_str = request.form.get("start_date")
        itinerary_text = request.form.get("itinerary")
        hotel_text = request.form.get("hotelinfo")

        master_file = request.files.get("master_file")
        if not master_file:
            return "⚠️ Please upload master Excel"

        if not start_date_str or not itinerary_text:
            return "⚠️ Please provide start date and itinerary."

        # load master excel
        master_df = pd.read_excel(master_file)

        # inputs
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        itinerary_lines = [line.strip() for line in itinerary_text.split("\n") if line.strip()]
        hotel_lines = [line.strip() for line in hotel_text.split("\n")] if hotel_text else []

        # Create Word doc
        doc = Document()
        doc.add_heading("Travel LYKKE – Service Voucher", level=0)

        doc.add_paragraph(f"Destination: {destination}")
        doc.add_paragraph(f"Trip Start Date: {start_date.strftime('%d-%b-%Y')}")
        doc.add_paragraph("")

        # Day wise details
        for i, line in enumerate(itinerary_lines):
            day_date = (start_date + timedelta(days=i)).strftime("%d-%b-%Y")
            doc.add_heading(f"Day {i+1} – {day_date}", level=1)

            # multiple services per day (split by +)
            services = [s.strip() for s in line.split("+")]
            for service in services:
                formatted = match_service(service, master_df)
                doc.add_paragraph(formatted)

            # hotel info
            if i < len(hotel_lines):
                doc.add_paragraph(f"Hotel: {hotel_lines[i]}")

        # Escalation + Notes from configs
        if destination in COUNTRY_CONFIGS:
            config = COUNTRY_CONFIGS[destination]
            doc.add_page_break()
            doc.add_heading("Escalation Matrix", level=1)
            doc.add_paragraph(config["escalation_matrix"])

            doc.add_heading("Special Notes", level=1)
            doc.add_paragraph(config["special_note"])

            doc.add_heading("Other Comments", level=1)
            doc.add_paragraph(config["other_comments"])

        # Save Word file
        output = BytesIO()
        doc.save(output)
        output.seek(0)

        return send_file(output, download_name="Service_Voucher.docx", as_attachment=True)

    return render_template("index.html")
