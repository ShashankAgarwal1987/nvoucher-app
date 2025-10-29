from flask import Flask, request, render_template, send_file
import pandas as pd
from openai import OpenAI
import os
from io import BytesIO
from docx import Document
from datetime import datetime, timedelta
from configs import COUNTRY_CONFIGS

app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def match_service(text, master_df):
    """
    Use embeddings + OpenAI to find closest matching formatted output.
    """
    try:
        # For now we just do simple fuzzy match
        for _, row in master_df.iterrows():
            if str(row["Particular"]).lower() in text.lower():
                return row["Formatted Output"]
        # fallback
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

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        itinerary_lines = [line.strip() for line in itinerary_text.split("\n") if line.strip()]
        hotel_lines = [line.strip() for line in hotel_text.split("\n") if hotel_text else []]

        # Create Word doc
        doc = Document()
        doc.add_heading("Travel LYKKE – Service Voucher", level=0)

        doc.add_paragraph(f"Destination: {destination}")
        doc.add_paragraph(f"Trip Start Date: {start_date.strftime('%d-%b-%Y')}")
        doc.add_paragraph("")

        # Day wise
        for i, line in enumerate(itinerary_lines):
            day_date = (start_date + timedelta(days=i)).strftime("%d-%b-%Y")
            doc.add_heading(f"Day {i+1} – {day_date}", level=1)

            # Split multiple services with +
            services = [s.strip() for s in line.split("+")]
            for service in services:
                formatted = match_service(service, master_df)
                doc.add_paragraph(formatted)

            # hotel info
            if i < len(hotel_lines):
                doc.add_paragraph(f"Hotel: {hotel_lines[i]}")

        # Escalation + Notes
        if destination in COUNTRY_CONFIGS:
            config = COUNTRY_CONFIGS[destination]
            doc.add_page_break()
            doc.add_heading("Escalation Matrix", level=1)
            doc.add_paragraph(config["escalation_matrix"])

            doc.add_heading("Special Notes", level=1)
            doc.add_paragraph(config["special_note"])

            doc.add_heading("Other Comments", level=1)
            doc.add_paragraph(config["other_comments"])

        # Save Word
        output = BytesIO()
        doc.save(output)
        output.seek(0)

        return send_file(output, download_name="Service_Voucher.docx", as_attachment=True)

    return render_template("index.html")
