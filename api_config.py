from .script import response

from flask import Flask, request, jsonify
from pypdf import PdfReader

app = Flask(__name__)

@app.route('/api/process-bill', methods=['POST'])
def process_bill():
    if 'bill' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['bill']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    reader = PdfReader(file)  ##input pdf from user
    parsed_pdf = ""
    for page in reader.pages:
        parsed_pdf += page.extract_text() + "\n"
    bill_overview = response(parsed_pdf)
    return jsonify({'status': 'success', 'overview': bill_overview}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)