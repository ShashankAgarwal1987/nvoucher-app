from flask import Flask, request, render_template, send_file
import pandas as pd
from openai import OpenAI
import os
from io import BytesIO
from datetime import datetime, timedelta

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print("Embedding error:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1. Read uploaded master file
        if "master_file" not in request.files:
            return "⚠️ No master file uploaded"
        
        master_file = request.files["master_file"]
        if master_file.filename == "":
            return "⚠️ Empty file uploaded"

        # 2. Read start date & itinerary
        start_date_str = request.form.get("start_date")
        itinerary_text = request.form.get("itinerary")

        if not start_date_str or not itinerary_text:
            return "⚠️ Missing start date or itinerary"

        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            master_df = pd.read_excel(master_file)

            outputs = []
            for i, line in enumerate(itinerary_text.split("\n")):
                line = line.strip()
                if not line:
                    continue

                day_date = start_date + timedelta(days=i)
                embedding = get_embedding(line)

                if embedding is None:
                    formatted = f"⚠️ Could not process: {line}"
                else:
                    # For now just echoing back — here you can add fuzzy/embedding match logic
                    formatted = f"Processed service: {line}"

                outputs.append({
                    "Day": f"Day {i+1}",
                    "Date": day_date.strftime("%d-%b-%Y"),
                    "Service": formatted
                })

            # Export result to Excel
            result_df = pd.DataFrame(outputs)
            output = BytesIO()
            result_df.to_excel(output, index=False)
            output.seek(0)

            return send_file(output, download_name="voucher_output.xlsx", as_attachment=True)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    return render_template("index.html")
