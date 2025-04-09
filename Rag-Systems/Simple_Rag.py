import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import PyPDF2
import uuid

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Function to encode local image to base64 for embedding in HTML
def image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image}"

class SimpleModelSelector:
    """Handles model selection – defaulting to Llama3 and Chroma."""

    def __init__(self):
        self.llm_models = {"ollama": "Llama3"}  # Only Llama3
        self.embedding_models = {
            "chroma": {
                "name": "Chroma Default",
                "dimensions": 384,
                "model_name": None,
            }
        }

class SimplePDFProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if start > 0:
                start = start - self.chunk_overlap
            chunk = text[start:end]
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {"source": pdf_file.name},
            })
            start = end
        return chunks

class SimpleRAGSystem:
    def __init__(self, embedding_model="chroma", llm_model="ollama"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.setup_embedding_function()
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        try:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def setup_collection(self):
        collection_name = f"documents_{self.embedding_model}"
        try:
            try:
                collection = self.db.get_collection(
                    name=collection_name, embedding_function=self.embedding_fn
                )
            except:
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": self.embedding_model},
                )
                st.success(f"Created new collection: {collection_name}")
            return collection
        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e

    def add_documents(self, chunks):
        try:
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context):
        import requests
        import json

        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """

            headers = {"Content-Type": "application/json"}
            data = {
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            }

            response = requests.post("http://localhost:11434/v1/chat/completions", headers=headers, json=data)
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }

def main():
    st.set_page_config(page_title="Ask Ollama-Docs", page_icon="📄", layout="centered")

    # Custom CSS for modern UI design
    st.markdown("""
        <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color:
        color: #333333;
        transition: background-color 0.3s ease;
    }

    h1, h2, h3, .stText {
        font-weight: bold;
        color: #2d3436;  
        font-size: 40px; 
        line-height: 1.4;
    }

    /* Button styling */
    .stButton>button {
        background-color: #00b894;  
        color: white;
        border-radius: 8px;
        padding: 14px 30px;
        font-size: 20px;
        cursor: pointer;
        border: none;
        box-shadow: 0 6px 14px rgba(0, 184, 148, 0.2);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b16a; 
        transform: scale(1.05);
    }

    .stTextInput>div>input {
        border: 2px solid #6c5ce7; 
        border-radius: 8px;
        padding: 14px;
        font-size: 18px;
        color: #2d3436;
        background-color: #f7f9f9;  
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput>div>input:focus {
        border-color: #6c5ce7;  
        outline: none;
        box-shadow: 0 0 5px rgba(108, 92, 231, 0.5);
    }

    .stFileUploader>div {
        background-color: #0984e3;  
        color: white;
        padding: 14px 28px;
        border-radius: 8px;
        font-size: 18px;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stFileUploader>div:hover {
        background-color: #074fa3;  
        transform: scale(1.05);
    }

    .main-container {
        padding: 50px;
        background-color: 
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
        max-width: 1200px;
        margin: 0 auto;
        text-align: center;
    }

    .stExpanderHeader {
        font-size: 28px;
        font-weight: bold;
        color: #e74c3c;  
    }

    .stExpanderContent {
        color: #2d3436;
        font-size: 18px;
    }

    .stCaption, .stText {
        font-size: 18px;
        color: #636e72;  
        line-height: 1.6;
    }

    .stTooltip {
        background-color: #f39c12; 
        color: 
        border-radius: 6px;
        font-size: 16px;
        padding: 8px 14px;
    }

    .stLink {
        color: #0984e3;
        font-weight: bold;
        text-decoration: none;
    }

    .stLink:hover {
        color: #074fa3;
        text-decoration: underline;
    }
</style>
    """, unsafe_allow_html=True)

    st.title("ℝ𝔸𝔾 𝕊𝕪𝕤𝕥𝕖𝕞")
    st.caption("Upload a PDF document and ask any question. Llama will analyze the document content and provide you with accurate, context-based answers.")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    llm_model = "llama3"
    embedding_model = "chroma"

    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None

    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        embedding_info = st.session_state.rag_system.get_embedding_info()
        with st.sidebar:
            st.header("System Info")
            st.success(f"**LLM:** {llm_model}")
            st.info(f"**Embedding:** {embedding_info['name']} ({embedding_info['dimensions']}D)")

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    st.markdown("---")
    pdf_file = st.file_uploader("📤 Upload a PDF document", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = SimplePDFProcessor()
        with st.spinner("🔍 Processing your PDF..."):
            try:
                text = processor.read_pdf(pdf_file)
                chunks = processor.create_chunks(text, pdf_file)
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"✅ Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("💬 Ask Questions from Your Document")
        query = st.text_input("Type your question here:")

        if query:
            with st.spinner("🧠 Thinking..."):
                results = st.session_state.rag_system.query_documents(query)
                if results and results["documents"]:
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )
                    if response:
                        st.markdown("### 📝 Answer:")
                        st.success(response)

                        with st.expander("📚 Source Passages"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc)
    else:
        st.info("👆 Please upload a PDF to begin querying!")

if __name__ == "__main__":
    main()
