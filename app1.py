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
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "You are an AI that extracts structured data from documents. "
        "Always output a JSON object strictly in this format:\n\n"
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"document_type\": \"<detected type like Aadhaar Card, PAN Card, Passport, Driving License, Marksheet, Invoice, Contract, Text>\",\n"
        "    \"extracted_data\": { ...fields depending on document type OR fallback message only... }\n"
        "  },\n"
        "  \"compliance_status\": \"<status based on completeness and compliance rules>\",\n"
        "  \"name\": \"response\"\n"
        "}\n\n"
        "### Rules by Document Type ###\n"
        "- Aadhaar Card -> Extract: name, dob, gender, aadhaar_number, Address.\n"
        "- PAN Card -> Extract: name, father's name, dob, Pan number, signature.\n"
        "- Passport -> Extract: name, passport_number, dob, nationality, issue_date, expiry_date.\n"
        "- Driving License -> Extract: name, license_number, dob, issue_date, validity.\n"
        "- Marksheet/Examination Certificate -> Extract: Roll No, exam_type, certificate_number, Candidate Name , Mother Name, Father Name, DOB, School/College Name, Exam Year, Subjects [{Subject, Max Marks, Total Marks, Grade}], Result, Date of Issue, Place, Verification Website.\n"
        "- Invoice -> Extract: invoice_number, date, seller_name, buyer_name, items, total_amount, tax_amount.\n"
        "- Contract -> Extract: contract_id, parties_involved, start_date, end_date, key_terms.\n"
        "- Voter ID -> Extract: name, father_name, dob, gender, voter_id_number, address.\n"
        "- Birth Certificate -> Extract: child_name, father_name, mother_name, dob, place_of_birth, registration_number.\n"
        "- Property Registration -> Extract: owner_name, property_address, registration_number, date_of_registration, registrar_office.\n"
        "- Tax Return: Extract: taxpayer_name, pan_number, assessment_year, income, tax_paid, refund_status.\n"
        "- Income Certificate -> Extract: Certificate_number, Applicant_name, Father's_name, Address, Annual_income, Issue_date, Validity, Issuing_authority.\n"
        "### Rules for compliance_status ###\n"
        "- If all fields are present -> 'compliant'.\n"
        "- If document fields are fully extracted and valid -> 'Data extracted successfully for regulatory review.'\n"
        "- If some fields are missing/unclear -> 'Partial data extracted — further verification required.'\n"
        "- If document type is unrecognized -> 'Document type not identified — manual review required.'\n"
        "- If sensitive data mismatch detected -> 'Data format issue — needs correction.'\n"
        "- If type not identified but text present (non-document) ->\n"
        "  document_type='text',\n"
        "  extracted_data={\"message\": \"It appears that the input was minimal or unrelated to a document. Please provide a proper document.\"},\n"
        "  compliance_status='N/A'.\n"
        "### Important ###\n"
        "- Detect document type first.\n"
        "- Never invent or hallucinate fields.\n"
        "- If input is non-document text, only return the fallback message under extracted_data.\n"
        "- Output JSON only. Never include explanations outside JSON."
    )

    response = model.generate_content([prompt, extracted_text])
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