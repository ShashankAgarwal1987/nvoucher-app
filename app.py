from flask import Flask, request, render_template, send_file
import pandas as pd
from openai import OpenAI
import os
from io import BytesIO
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Border, Side

app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ai_match_service(itinerary_text, master_df):
    """
    Use OpenAI embeddings to find best matching service output.
    """
    try:
        # Get embedding for input itinerary text
        emb_inp = client.embeddings.create(
            input=[itinerary_text],
            model="text-embedding-3-small"
        ).data[0].embedding

        best_score = -1
        best_output = "⚠️ No match found"
        for _, row in master_df.iterrows():
            text = str(row[2])
            emb_row = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            ).data[0].embedding

            # Cosine similarity
            score = sum(a*b for a, b in zip(emb_inp, emb_row)) / (
                (sum(a*a for a in emb_inp) ** 0.5) * (sum(b*b for b in emb_row) ** 0.5)
            )

            if score > best_score:
                best_score = score
                best_output = str(row[3]) if len(row) > 3 else text

        return best_output

    except Exception as e:
        return f"❌ Error matching: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Read uploaded files and inputs
            master_file = request.files["master_file"]
            start_date = request.form["start_date"]
            itinerary_text = request.form["itinerary"]

            if not master_file or not start_date or not itinerary_text:
                return "⚠️ Missing inputs"

            # Load master excel
            master_df = pd.read_excel(master_file)

            # Load template voucher (static, shipped with app)
            template_path = "voucher_template.xlsx"
            wb = openpyxl.load_workbook(template_path)
            ws = wb.active

            # Find Service Details header row
            header_row = None
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
                for cell in row:
                    if cell.value and "Service Details" in str(cell.value):
                        header_row = cell.row
                        break
                if header_row:
                    break

            if not header_row:
                return "❌ Could not find 'Service Details' section in template"

            # Parse itinerary lines
            lines = [line.strip() for line in itinerary_text.split("\n") if line.strip()]

            # Insert services below header row
            current_row = header_row + 1
            from datetime import datetime, timedelta
            base_date = datetime.strptime(start_date, "%Y-%m-%d")

            thin = Side(border_style="thin", color="000000")

            for i, line in enumerate(lines):
                match = ai_match_service(line, master_df)

                # Write Date
                ws.cell(row=current_row, column=1).value = (base_date + timedelta(days=i)).strftime("%d-%b-%Y")

                # Write Service Details
                ws.cell(row=current_row, column=2).value = match

                # Apply borders and wrapping
                for col in range(1, 3):
                    cell = ws.cell(row=current_row, column=col)
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    cell.border = Border(top=thin, left=thin, right=thin, bottom=thin)

                current_row += 1

            # Save output
            output = BytesIO()
            wb.save(output)
            output.seek(0)

            return send_file(output, download_name="voucher_output.xlsx", as_attachment=True)

        except Exception as e:
            return f"❌ Failed to write into template: {str(e)}"

    return render_template("index.html")
