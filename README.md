ℝ𝔸𝔾 𝕊𝕪𝕤𝕥𝕖𝕞 - PDF Question Answering System

Overview

This system allows users to upload a PDF document, and ask questions based on the content of that document. Using a retrieval-augmented generation (RAG) approach, the system extracts relevant text from the uploaded PDF and generates context-based answers using Llama3 (a language model). The system uses Chroma for embeddings, making the entire process fast and efficient.

Features

PDF Upload: Upload PDF documents to process and index.
Document Chunking: Large documents are split into smaller chunks for efficient processing.
Question Answering: Ask questions related to the uploaded PDF, and get accurate answers based on its content.
Model Integration: Uses Llama3 for language understanding and Chroma for embedding.
Technologies

Streamlit: For building the web interface.
Chroma: For persistent document storage and retrieval.
PyPDF2: For PDF file reading and text extraction.
Ollama (Llama3): For generating answers based on document content.
uuid: For unique chunk ID generation.
Prerequisites

Before running the system, make sure you have the following dependencies installed:

Python 3.7+
Pip (Package Installer)
Install Dependencies
Clone the repository or download the script.
Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
Install the required Python libraries:
pip install -r requirements.txt
Setup Environment

To run the system, you need an .env file for the environment variables. The .env file should contain the following:

How to Use

Start the Streamlit app: Run the following command to start the app:
streamlit run app.py
Upload PDF: Click on the file uploader to upload a PDF document.
Ask Questions: After the document is processed, enter your query in the input box and get the response generated from the document content.
<img width="1728" alt="Screenshot 2025-04-10 at 10 20 19 AM" src="https://github.com/user-attachments/assets/f71fde4d-4478-4de8-b5e8-8763400ef4ba" />
<img width="1728" alt="Screenshot 2025-04-10 at 10 21 29 AM" src="https://github.com/user-attachments/assets/58490f39-352f-4340-87e0-ea10438ab009" />
# llm-learnings
