import os
import json
import re
import tempfile
import pdfplumber
import functions_framework
from PIL import Image
import easyocr
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from flask import jsonify, make_response, request

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"

ocr_reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_file(file_path: str) -> str:
    text = ""
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:  # PDF has digital text layer
                    text += page_text + "\n"
                else:
                    # PDF page is scanned → convert to image then OCR
                    pil_image = page.to_image(resolution=150).original
                    np_image = np.array(pil_image)
                    ocr_results = ocr_reader.readtext(np_image, detail=1)
                    text += " ".join([t for _, t, _ in ocr_results])
    else:
        # Handle Image (jpg, png, jpeg)
        image = Image.open(file_path)
        np_image = np.array(image) 
        ocr_results = ocr_reader.readtext(np_image, detail=1)
        text += " ".join([t for _, t, _ in ocr_results])
    return text.strip()

def clean_extracted_text(raw_text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9:/\-\s]", " ", raw_text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# EXTRACT JSON USING LOCAL LLM
def extract_json_from_text(extracted_text: str) -> str:
    cleaned_text = clean_extracted_text(extracted_text)
    model = genai.GenerativeModel(MODEL_NAME)
    # Clean prompt
    prompt = f"""
        You are an AI that extracts structured JSON from government documents.  
        From the following text input, detect the document type and extract all relevant fields.  
        Output strictly in JSON format only, without any explanations.
    
    
        Rules:
        1. Document types: Aadhaar Card, PAN Card, Passport, Driving License, Marksheet, Invoice, Contract, Voter ID, Birth Certificate, Property Registration, Tax Return, Income Certificate.
        2. Fields: Extract the fields as mentioned in the rules. For any missing fields, use null or "".
        3. If the input is plain text:
            - Single word:
                - If it is a name → extracted_data = {{"name": "<input>"}}
                - If it is an action word → extracted_data = {{"action": "<input>", "description": "User requested to initiate <input> action."}}
                - If it is a greeting → extracted_data = {{"action": "<input>", "description": "User greeted the system with '<input>'."}}
                - Else → extracted_data = {{"message": "<input>"}}
            - Phrase/sentence → extracted_data = {{"message": "<input>"}}
        4. Random/unrelated text: If the text is random or unrelated, set document_type = "text" and provide a message.
        5. Always include compliance_status based on rules:  
            - If all fields are present → "compliant"  
            - If some fields are missing or unclear → "partial data extracted — further verification required"  
            - If document type is unrecognized → "manual review required"  
            - If sensitive data mismatch detected → "data format issue — needs correction"
        6. Output format:
        {{
            "type": "object",
            "properties": {{
                "document_type": "<detected type or 'text'>",
                "extracted_data": {{ ...fields or message ... }},
                "compliance_status": "<status based on above rules>"
            }},
            "name": "response"
        }}
    
        Input Text:
        \"\"\"{cleaned_text}\"\"\"
    """

    response = model.generate_content([prompt])
    return response.text.strip() if response and response.text else "{}"


def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Main Cloud Function
@functions_framework.http
def process(request):
        if request.method == "OPTIONS":
            return cors(make_response(("", 204)))
        
        try:
            extracted_text = ""

            # ---------- 1. File upload ----------
            if request.files:
                f = next(iter(request.files.values()))
                suffix = os.path.splitext(f.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    f.save(tmp.name)
                    extracted_text = extract_text_from_file(tmp.name)
                    os.unlink(tmp.name)  # cleanup temp file


            # ---------- 2. JSON input ----------
            if not extracted_text and request.is_json:
                data = request.get_json(silent=True)
                if data and "text" in data:
                    extracted_text = data["text"] 

            # ---------- 3. Plain text input ----------
            if not extracted_text:
                raw_text = request.data.decode("utf-8").strip()
                if raw_text:
                    extracted_text = raw_text
            
            # ---------- 4. No input ----------
            if not extracted_text:
                return cors(make_response(jsonify({"error": "No input provided"}), 400))
        

            # ---------- 5. Call AI ----------
            json_result = extract_json_from_text(extracted_text)
    
            # Try parsing AI response as JSON
            try:
                result = json.loads(json_result)
            except Exception:
                result = {"raw_ai_response": json_result}
    
            return cors(make_response(jsonify(result), 200))   
        
        except Exception as e:
            return cors(make_response(jsonify({"error": str(e)}), 500))


