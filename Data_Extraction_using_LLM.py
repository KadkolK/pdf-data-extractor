# ---------- Imports ----------
import os  # File path and directory utilities
import fitz  # PyMuPDF: to extract text or images from PDF
import pytesseract  # OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Path to Tesseract OCR

from PIL import Image  # Image processing (used with OCR)
import io  # Convert image data between formats
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings
from langchain_community.vectorstores import FAISS  # Vector store for semantic search
from langchain_community.llms import HuggingFacePipeline  # Interface to LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits large text into manageable chunks
from langchain.chains import RetrievalQA  # QA chain using retriever + LLM
from transformers import pipeline  # HuggingFace inference pipeline
import json  # Exporting to JSON
import re  # Regex processing
import pdfplumber  # Extracts tables and text more accurately from PDFs

# ---------- Utility Functions (Table Parsing, Cleanup) ----------
def parse_table_block(table_block):
    lines = [line for line in table_block.split('\n') if line.strip()]
    rows = []
    for line in lines:
        if '|' in line:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
        elif '\t' in line:
            row = [cell.strip() for cell in line.split('\t') if cell.strip()]
        else:
            row = [cell.strip() for cell in re.split(r' {2,}', line) if cell.strip()]
        if row:
            rows.append(row)
    rows = [row for row in rows if not all(re.match(r"^[-\s]+$", cell) for cell in row)]
    return rows

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
    return text

def extract_order_table(text):
    patterns = [
        r"(Item Description[\s\S]+?)(Grand Total|Total Amount|Amount In Words)",
        r"(#.*?)(Grand Total|Total Amount|Amount In Words)",
        r"(Description\s+Qty\s+Rate[\s\S]+?)(Total Amount|Amount In Words|\n\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return "\n".join([line.strip() for line in match.group(1).split('\n') if line.strip()])
    return ""

def extract_customer_address(text):
    match = re.search(r"Purchase Order.*?\n(.*?)\nSupplier Name:", text, re.DOTALL | re.IGNORECASE)
    if match:
        lines = [line.strip() for line in match.group(1).split('\n') if line.strip()]
        return " ".join([l for l in lines if not re.search(r'@|www\.|[0-9]{10}', l)])
    return ""

def create_faiss_index(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def extract_and_store(pdf_path, query_prompts):
    print("[*] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("[*] Creating FAISS index...")
    vector_store = create_faiss_index(text)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    hf_pipe = pipeline("text-generation", model="Qwen/Qwen1.5-1.8B-Chat", device_map="auto", do_sample=False, max_new_tokens=64)
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    results_dict = {}

    for field_name, prompt in query_prompts.items():
        print(f"[*] Extracting: {field_name}")
        result = qa_chain.invoke({"query": prompt})
        answer = result['result'].split("Helpful Answer:")[-1].split("Explanation:")[0].strip()

        if field_name == "Customer's Address":
            if not answer or len(answer) < 10:
                answer = extract_customer_address(text)

        if field_name == "Item Description":
            lines = answer.split('\n')
            cleaned = [line.strip() for line in lines if line.strip() and not re.match(r"^[-\s]*$", line)]
            answer = '\n'.join(cleaned)
            if not answer or len(parse_table_block(answer)) < 2:
                fallback = extract_order_table(text)
                if fallback:
                    answer = fallback

        if field_name == "Extra Information":
            if "does not contain any leftover" in answer.lower() or len(answer.strip()) < 15:
                answer = "N/A"

        print(f" -> {field_name}: {answer}")
        results_dict[field_name] = answer

    json_path = os.path.splitext(pdf_path)[0] + "_extracted.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"[*] Done. Saved to: {json_path}")

# ---------- Main Entry Point ----------
if __name__ == "__main__":
    pdf_file = input("Enter PDF file path: ").strip()
    if not os.path.exists(pdf_file):
        print("File not found.")
        exit(1)

    query_prompts = {  #put your query prompts here...
    }


    extract_and_store(pdf_file, query_prompts)
    print("[-------] Extraction complete [-------]")




