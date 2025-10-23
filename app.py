from flask import Flask, request, render_template, send_file
import pandas as pd
from openai import OpenAI
import os
from io import BytesIO
import gc

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """Fetch embeddings with memory safety."""
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print("Embedding error:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "⚠️ Empty file"

        try:
            # Use iterator to avoid loading full file in memory
            df_iter = pd.read_excel(file, chunksize=1)  # one row at a time

            output = BytesIO()
            writer = pd.ExcelWriter(output, engine="xlsxwriter")
            results = []

            day_counter = 1
            for chunk in df_iter:
                row = chunk.iloc[0]  # single row
                text = str(row.iloc[2]) if len(row) > 2 else ""  # 3rd column

                embedding = get_embedding(text)

                if embedding is None:
                    formatted = f"⚠️ Could not process: {text}"
                else:
                    formatted = f"Processed service: {text}"

                results.append({"Day": f"Day {day_counter}", "Formatted": formatted})
                day_counter += 1

                # Free memory
                del embedding, chunk, row
                gc.collect()

            # Convert to DataFrame and save once
            result_df = pd.DataFrame(results)
            result_df.to_excel(writer, index=False, sheet_name="Voucher")
            writer.close()

            output.seek(0)
            return send_file(output, download_name="voucher_output.xlsx", as_attachment=True)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    return render_template("index.html")
