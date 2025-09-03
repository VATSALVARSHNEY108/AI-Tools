import io
import pandas as pd
import PyPDF2
from docx import Document
from PIL import Image
import streamlit as st

def process_text_file(uploaded_file):
    """Process uploaded text file"""
    try:
        content = uploaded_file.read().decode('utf-8')
        return content
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return None

def process_pdf_file(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def process_docx_file(uploaded_file):
    """Extract text from Word document"""
    try:
        doc = Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return None

def process_csv_file(uploaded_file):
    """Process CSV file and return DataFrame"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

def process_image_file(uploaded_file):
    """Process image file and return image bytes and PIL Image"""
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image_bytes, image
    except Exception as e:
        st.error(f"Error processing image file: {e}")
        return None, None

def get_file_type(filename):
    """Get file type from filename"""
    extension = filename.lower().split('.')[-1]
    if extension in ['txt']:
        return 'text'
    elif extension in ['pdf']:
        return 'pdf'
    elif extension in ['docx', 'doc']:
        return 'docx'
    elif extension in ['csv']:
        return 'csv'
    elif extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
        return 'image'
    else:
        return 'unknown'
