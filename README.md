# pdf-data-extractor
Python tool to extract structured information (tables, addresses, totals, etc.) from PDFs using OCR, FAISS embeddings, and HuggingFace LLMs. Works on scanned or digital PDFs, customizable via query prompts.

# PDF Data Extractor

A Python tool to extract structured information from PDFs using OCR, semantic search, and LLMs.  
It works on both scanned and digital PDFs.  

**Note:** The code includes example extraction logic for Purchase Orders (regex-based). The hardcoded query prompts have been removed, but some functions may still assume PO-like structures. Users can adapt or extend the extraction functions to handle other document types.

---

## Features
- Works with both scanned and digital PDFs.  
- Extracts text, tables, and structured information.  
- Uses FAISS embeddings for semantic retrieval.  
- Powered by HuggingFace LLMs for field-specific extraction.  
- Saves extracted information in JSON format.  
- Fully configurable query prompts (removed from code for generalization).

---

## Project Timeline
- Completed: July 2025  
- Uploaded to GitHub: October 2025

---
