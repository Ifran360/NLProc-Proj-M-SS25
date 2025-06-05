import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts and concatenates text from a PDF file.
    Returns one string containing all text.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()
