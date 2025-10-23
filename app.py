from flask import Flask, request, render_template, send_file
import pandas as pd
from openai import OpenAI
import os
from io import BytesIO

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
        if "file" not in request.files:
            return "⚠️ No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "⚠️ Empty file"

        try:
            df = pd.read_excel(file)

            outputs = []
            for i, row in df.iterrows():
                text = str(row.iloc[2]) if len(row) > 2 else ""  # Safely get 3rd column
                embedding = get_embedding(text)

                if embedding is None:
                    formatted = f"⚠️ Could not process: {text}"
                else:
                    formatted = f"Processed service: {text}"

                outputs.append({"Day": f"Day {i+1}", "Formatted": formatted})

            result_df = pd.DataFrame(outputs)

            output = BytesIO()
            result_df.to_excel(output, index=False)
            output.seek(0)

            return send_file(output, download_name="voucher_output.xlsx", as_attachment=True)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    return render_template("index.html")
