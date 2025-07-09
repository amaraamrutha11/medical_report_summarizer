from flask import Flask, render_template, request
from summarizer import summarize_text  # Now uses BERT
from pymongo import MongoClient
import datetime
import os
import PyPDF2
import docx
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['medical_reports_db']
collection = db['reports']

# Extract text from PDF
def extract_text_from_pdf(filepath):
    text = ''
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

# Extract text from DOCX
def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return '\n'.join([para.text for para in doc.paragraphs])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text_input = request.form.get('report')
    file = request.files.get('file')
    extracted_text = ''

    if file and file.filename:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.docx'):
            extracted_text = extract_text_from_docx(filepath)
        else:
            return "Unsupported file format", 400
    elif text_input.strip():
        extracted_text = text_input.strip()
    else:
        return "No input provided", 400

    summary = summarize_text(extracted_text)

    record = {
        'original_report': extracted_text,
        'summary': summary,
        'timestamp': datetime.datetime.now()
    }
    collection.insert_one(record)

    return render_template('summary.html', summary=summary)

@app.route('/history')
def history():
    records = list(collection.find().sort('timestamp', -1))
    return render_template('history.html', records=records)

if __name__ == '__main__':
    print("\n App running at: http://127.0.0.1:5000")
    print(" History Page: http://127.0.0.1:5000/history\n")
    app.run(debug=True)
