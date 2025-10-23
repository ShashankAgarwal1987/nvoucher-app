from flask import Flask, request, render_template, send_file
import pandas as pd
import openai
import os
from io import BytesIO

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded"

        try:
            df = pd.read_excel(file)

            outputs = []
            for i, row in df.iterrows():
                text = str(row[2])  # Assuming 3rd column is "Actual Input"
                embedding = get_embedding(text)

                if embedding is None:
                    formatted = f"⚠️ Could not process: {text}"
                else:
                    # Instead of returning embedding, you’d format into voucher-style
                    formatted = f"Processed service: {text}"

                outputs.append({"Day": f"Day {i+1}", "Formatted": formatted})

            result_df = pd.DataFrame(outputs)

            # Save result to Excel
            output = BytesIO()
            result_df.to_excel(output, index=False)
            output.seek(0)

            return send_file(output, download_name="voucher_output.xlsx", as_attachment=True)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")
