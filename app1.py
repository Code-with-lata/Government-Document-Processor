import os
import json
import requests
import streamlit as st
import pdfplumber
from PIL import Image
import easyocr
import numpy as np
import uuid
import gc
import google.generativeai as genai


API_KEY = os.getenv("GEMINI_API_KEY", None)
if not API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
    API_KEY = "AIzaSyB1heo6JymJ4aOcqBRZhg3GzZwiOEpsFis"
else:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash" 

def clear_memory():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

# INITIALIZE OCR (ONLY ONCE)
if "ocr_reader" not in st.session_state:
    st.session_state.ocr_reader = easyocr.Reader(['en'], gpu=True)

reader = st.session_state.ocr_reader

def extract_text_from_pdf(file_path):
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
                    results = reader.readtext(np_image)
                    for res in results:
                        text += res[1] + " "
                    text += "\n"

    else:
        # Handle Image (jpg, png, jpeg)
        image = Image.open(file_path)
        np_image = np.array(image) 
        results = reader.readtext(np_image)
        for res in results:
            text += res[1] + " "
    return text.strip()


# EXTRACT JSON USING LOCAL LLM
def extract_json_from_text(extracted_text):
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
    return response.text.strip()

# STREAMLIT UI
st.set_page_config(page_title="Government Document Processor", layout="wide")
st.title("📄 Government Document Processor (Gemini API)")

text_input = st.text_area("Your Input (Text)", placeholder="Provide text details or describe your challenge...")
uploaded_file = st.file_uploader("Upload File", type=["pdf", "png", "jpg", "jpeg"])

if st.button("🔍 Process with Gemini API"):
    extracted_text = ""

    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            extracted_text = extract_text_from_pdf(file_path)

    elif text_input.strip():
        extracted_text = text_input.strip()

    if extracted_text:
        st.subheader("Extracted Text (Preview)")
        st.text_area("Extracted Text", extracted_text, height=200)

        with st.spinner("Processing..."):
            json_result = extract_json_from_text(extracted_text)

        st.subheader("AI Response:")
        try:
            st.json(json.loads(json_result))
        except:
            st.text(json_result)
    else:
        st.warning("Could not extract any text from this file.")