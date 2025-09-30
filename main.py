import os
import re
import json
import pdfplumber
import faiss
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests


class FinancialComplianceRAG:
    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.load_environment()
        
    def load_environment(self):
        """Load environment variables"""
        load_dotenv()
        self.openrouter_key = os.getenv('API_KEY')
        
    def clean_text(self, text: str) -> str:
        """Clean extracted text from PDFs"""
        # Remove page numbers (standalone digits)
        text = re.sub(r"\n?\s*\d+\s*\n?", " ", text)
        # Remove multiple newlines
        text = re.sub(r"\n{2,}", "\n", text)
        # Normalize spaces
        text = re.sub(r" +", " ", text)
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"
        return self.clean_text(full_text)
    
    def chunk_text(self, text: str, chunk_size=500, chunk_overlap=100):
        """Split text into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)
    
    def process_pdfs(self, raw_dir="data/raw", processed_dir="data/processed"):
        """Process all PDFs in the raw directory"""
        RAW_DIR = Path(raw_dir)
        PROCESSED_DIR = Path(processed_dir)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        for pdf_file in RAW_DIR.glob("*.pdf"):
            print(f"Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(pdf_file)
            chunks = self.chunk_text(text)

            # Save as JSON
            output_file = PROCESSED_DIR / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    [{"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            print(f"Saved {len(chunks)} chunks → {output_file}")
    
    def setup_vector_database(self, processed_dir="data/processed"):
        """Set up the vector database with processed documents"""
        PROCESSED_DIR = Path(processed_dir)
        
        # Load all chunks
        self.documents = []
        for file in PROCESSED_DIR.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                for c in chunks:
                    self.documents.append({
                        "text": c["text"],
                        "source": file.stem,   # keep track of original PDF
                        "chunk_id": c["chunk_id"]
                    })

        # Load small embedding model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Generate embeddings for all chunks
        print("Generating embeddings...")
        embeddings = self.model.encode([doc["text"] for doc in self.documents], show_progress_bar=True)

        # Convert to numpy array (FAISS needs float32)
        embeddings = np.array(embeddings, dtype="float32")

        # Dimensions of embeddings
        dim = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatIP(dim)

        # Add embeddings to index
        self.index.add(embeddings)

        # Save FAISS index
        faiss.write_index(self.index, "faiss_index.bin")

        # Save metadata separately
        with open("faiss_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        print("Vector database setup complete.")
    
    def load_vector_database(self):
        """Load the pre-built vector database"""
        # Load the embedding model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FAISS index
        self.index = faiss.read_index("faiss_index.bin")
        
        # Load metadata
        with open("faiss_metadata.json", "r", encoding="utf-8") as f:
            self.documents = json.load(f)
    
    def search(self, query, top_k=3):
        """Search for relevant documents given a query"""
        # Embed query
        query_vector = self.model.encode([query], convert_to_numpy=True).astype("float32")
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # safeguard
                continue
            results.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "chunk_id": self.documents[idx]["chunk_id"],
                "distance": float(dist)
            })
        return results
    
    def query_llm(self, user_question, search_results):
        """Query the LLM with the user question and search results"""
        rag_context = ""
        count = 1
        for r in search_results:
            rag_context += f"Extract {count}- "
            rag_context += f"[{r['source']} - chunk {r['chunk_id']}] (score={r['distance']:.4f}) \n"
            rag_context += f"{r['text'][:1500]} \n\n"
            count += 1

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "x-ai/grok-4-fast:free",
            "messages": [
                {"role": "user", "content": f"""
                    You are a financial and legal compliance assistant. Answer the question using the provided RAG context ONLY. Always cite the source.
                    If relevant context is not provided- say "Relevant information not found in database."
                    Question: {user_question}
                    Context: {rag_context}
            """}
            ]
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_data = response.json()

            model_reply = response_data["choices"][0]["message"]["content"]
            return model_reply

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return "Error occurred during API request."
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            print(f"Raw response content: {response.text}")
            return "Error decoding response from API."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "Unexpected error occurred."
    
    def answer_question(self, user_question, top_k=3):
        """Main method to answer a user question using RAG"""
        # Search for relevant documents
        search_results = self.search(user_question, top_k)
        
        # Query the LLM with search results
        answer = self.query_llm(user_question, search_results)
        
        return answer


def main():
    # Initialize the RAG system
    rag_system = FinancialComplianceRAG()
    
    # Check if vector database exists, if not create it
    if not (Path("faiss_index.bin").exists() and Path("faiss_metadata.json").exists()):
        print("Vector database not found. Processing PDFs and creating vector database...")
        rag_system.process_pdfs()
        rag_system.setup_vector_database()
        print("Vector database created successfully.")
    else:
        print("Loading existing vector database...")
        rag_system.load_vector_database()
        print("Vector database loaded successfully.")
    
    # Main interaction loop
    print("\nFinancial Compliance RAG System Ready!")
    print("Ask questions about financial regulations. Type 'quit' to exit.\n")
    
    while True:
        user_question = input("Your question: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question.strip():
            answer = rag_system.answer_question(user_question)
            print("\nAnswer:")
            print(answer)
            print("\n" + "="*50 + "\n")
        else:
            print("Please enter a valid question.")


if __name__ == "__main__":
    main()