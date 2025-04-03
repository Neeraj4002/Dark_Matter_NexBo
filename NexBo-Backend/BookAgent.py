import os
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import Gemini client from google.genai
from google import genai
from google.genai import types

# -----------------------------
# Configuration & Logging Setup
# -----------------------------
GEMINI_API_KEY = "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE"  # Replace with your actual Gemini API key

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Global Variables and Book Mapping
# -----------------------------
# Map book names to their file names (PDF, embeddings, and faiss index)
BOOK_INFO = {
    "Rich Dad Poor Dad": {
        "pdf": "Rich_Dad_Poor_Dad.pdf",
        "embeddings": "Rich_Dad_Poor_Dad_embeddings.npy",
        "faiss_index": "Rich_Dad_Poor_Dad_faiss.index"
    },
    "The Psychology of Money": {
        "pdf": "The_Psychology_of_Money.pdf",
        "embeddings": "The_Psychology_of_Money_embeddings.npy",
        "faiss_index": "The_Psychology_of_Money_faiss.index"
    },
    "The Total Money Makeover": {
        "pdf": "The_Total_Money_Makeover.pdf",
        "embeddings": "The_Total_Money_Makeover_embeddings.npy",
        "faiss_index": "The_Total_Money_Makeover_faiss.index"
    }
}

# Global cache for each book's processed data: {"chunks": ..., "faiss_index": ...}
BOOK_SYSTEMS = {}

# Load the Sentence Transformer model once for all books
logger.info("Loading Sentence Transformer model (all-mpnet-base-v2)...")
MODEL = SentenceTransformer("all-mpnet-base-v2")

# -----------------------------
# Utility Functions
# -----------------------------
def load_book_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info("Book loaded from PDF successfully: %s", pdf_path)
        return text
    except Exception as e:
        logger.error("Error loading PDF %s: %s", pdf_path, e)
        logger.error(traceback.format_exc())
        raise

def chunk_text(text: str, max_words: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = max(0, end - overlap)
    logger.info("Text split into %d chunks.", len(chunks))
    return chunks

def get_embedding(text: str) -> np.ndarray:
    """Generate an embedding for the given text using the Sentence Transformer model."""
    try:
        embedding = MODEL.encode(text)
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error("Error generating embedding: %s", e)
        logger.error(traceback.format_exc())
        raise

def compute_embeddings_parallel(chunks: list) -> np.ndarray:
    """Compute embeddings for each text chunk in parallel."""
    embeddings = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(get_embedding, chunk): idx for idx, chunk in enumerate(chunks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                emb = future.result()
                embeddings.append((idx, emb))
            except Exception as e:
                logger.error("Error computing embedding for chunk %d: %s", idx, e)
    embeddings.sort(key=lambda x: x[0])
    emb_array = np.array([emb for _, emb in embeddings])
    logger.info("Computed embeddings for all chunks.")
    return emb_array

def save_embeddings(embeddings: np.ndarray, filename: str):
    """Save embeddings to a file using NumPy."""
    try:
        np.save(filename, embeddings)
        logger.info("Embeddings saved to %s.", filename)
    except Exception as e:
        logger.error("Error saving embeddings: %s", e)
        logger.error(traceback.format_exc())

def load_embeddings(filename: str):
    """Load embeddings from a file if available."""
    if os.path.exists(filename):
        try:
            embeddings = np.load(filename)
            logger.info("Embeddings loaded from %s.", filename)
            return embeddings
        except Exception as e:
            logger.error("Error loading embeddings: %s", e)
            logger.error(traceback.format_exc())
    return None

def save_faiss_index(index, filename: str):
    """Save FAISS index to disk."""
    try:
        faiss.write_index(index, filename)
        logger.info("FAISS index saved to %s.", filename)
    except Exception as e:
        logger.error("Error saving FAISS index: %s", e)
        logger.error(traceback.format_exc())

def load_faiss_index(filename: str):
    """Load FAISS index from disk if available."""
    if os.path.exists(filename):
        try:
            index = faiss.read_index(filename)
            logger.info("FAISS index loaded from %s.", filename)
            return index
        except Exception as e:
            logger.error("Error loading FAISS index: %s", e)
            logger.error(traceback.format_exc())
    else:
        logger.info("No FAISS index file found at %s.", filename)
    return None

def build_faiss_index(embeddings: np.ndarray):
    """Build and return a FAISS index for the given embeddings."""
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    logger.info("FAISS index built with %d vectors.", index.ntotal)
    return index

def retrieve_relevant_chunks(query: str, book_chunks: list, faiss_index, top_k: int = 3) -> list:
    """Retrieve the top_k most relevant text chunks for a given query from the provided book data."""
    try:
        query_emb = get_embedding(query)
        query_emb = np.expand_dims(query_emb, axis=0)
        distances, indices = faiss_index.search(query_emb, top_k)
        retrieved = [book_chunks[i] for i in indices[0]]
        logger.info("Retrieved %d relevant chunks for the query.", len(retrieved))
        return retrieved
    except Exception as e:
        logger.error("Error during retrieval: %s", e)
        logger.error(traceback.format_exc())
        raise

def generate_response_with_gemini(query: str, context_chunks: list, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate an answer using the Gemini‑2.0‑Flash API via generate_content."""
    try:
        context = "\n\n".join(context_chunks)
        prompt = f"Using the following context from the book:\n\n{context}\n\nAnswer the question: {query}"
        logger.info("Sending prompt to Gemini API using generate_content.")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",  # Adjust if needed
            config=types.GenerateContentConfig(
                system_instruction=prompt
            ),
            contents=query
        )
        answer = response.text.strip()  # Adjust based on actual response structure
        return answer
    except Exception as e:
        logger.error("Error generating response from Gemini: %s", e)
        logger.error(traceback.format_exc())
        raise

def initialize_system_for_book(book_name: str, pdf_path: str, embeddings_file: str, faiss_index_file: str) -> dict:
    """
    Initialize the system for a given book.
    Loads the PDF, chunks the text, and computes or loads embeddings and the FAISS index.
    Returns a dictionary with "chunks" and "faiss_index".
    """
    try:
        logger.info("Initializing system for book '%s' from %s", book_name, pdf_path)
        book_text = load_book_from_pdf(pdf_path)
        book_chunks = chunk_text(book_text)
        
        # Load cached embeddings if available; otherwise compute them.
        embeddings = load_embeddings(embeddings_file)
        if embeddings is None:
            logger.info("No cached embeddings found for %s. Computing embeddings...", embeddings_file)
            embeddings = compute_embeddings_parallel(book_chunks)
            save_embeddings(embeddings, embeddings_file)
        else:
            logger.info("Using cached embeddings from %s.", embeddings_file)
        
        # Load or build the FAISS index
        faiss_index = load_faiss_index(faiss_index_file)
        if faiss_index is None:
            logger.info("No cached FAISS index found. Building index...")
            faiss_index = build_faiss_index(embeddings)
            save_faiss_index(faiss_index, faiss_index_file)
        else:
            logger.info("Using cached FAISS index from %s.", faiss_index_file)
        
        logger.info("System initialization complete for book '%s'.", book_name)
        return {"chunks": book_chunks, "faiss_index": faiss_index}
    except Exception as e:
        logger.error("Failed to initialize system for book '%s': %s", book_name, e)
        logger.error(traceback.format_exc())
        raise

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    bookName: str

@app.post("/bookchat")
async def book_chat(request: ChatRequest):
    """
    Endpoint to handle chat queries for a given book.
    Expected JSON payload: {"message": "Hi", "bookName": "The Psychology of Money"}
    """
    book_name = request.bookName
    message = request.message

    if book_name not in BOOK_INFO:
        raise HTTPException(status_code=400, detail="Invalid bookName provided.")

    # Get the file info for the requested book
    book_files = BOOK_INFO[book_name]
    pdf_path = book_files["pdf"]
    embeddings_file = book_files["embeddings"]
    faiss_index_file = book_files["faiss_index"]

    # Initialize system for the book if not already done
    if book_name not in BOOK_SYSTEMS:
        try:
            BOOK_SYSTEMS[book_name] = initialize_system_for_book(book_name, pdf_path, embeddings_file, faiss_index_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error initializing system for the book.")

    book_data = BOOK_SYSTEMS[book_name]
    try:
        retrieved_chunks = retrieve_relevant_chunks(message, book_data["chunks"], book_data["faiss_index"], top_k=3)
        answer = generate_response_with_gemini(message, retrieved_chunks)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing chat query.")

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



# import os
# import logging
# import traceback
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from flask import Flask, request, jsonify
# import PyPDF2
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Import Gemini client from google.genai
# from google import genai
# from google.genai import types

# # -----------------------------
# # Configuration & Logging Setup
# # -----------------------------
# GEMINI_API_KEY = "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE"  # Replace with your actual Gemini API key

# # Initialize Gemini client
# gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # -----------------------------
# # Global Variables (for current book)
# # -----------------------------
# BOOK_CHUNKS = []         # List of text chunks for the currently loaded book
# FAISS_INDEX = None       # FAISS index for current book embeddings
# EMBEDDING_DIM = None     # Dimensionality of the embeddings
# MODEL = None             # Sentence Transformer model
# CURRENT_BASE_FILENAME = None  # Base name of the current PDF for caching

# # -----------------------------
# # Utility Functions
# # -----------------------------
# def load_book_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     try:
#         text = ""
#         with open(pdf_path, "rb") as file:
#             reader = PyPDF2.PdfReader(file)
#             for page in reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#         logger.info("Book loaded from PDF successfully.")
#         return text
#     except Exception as e:
#         logger.error("Error loading PDF: %s", e)
#         logger.error(traceback.format_exc())
#         raise

# def chunk_text(text, max_words=500, overlap=50):
#     """
#     Split text into overlapping chunks.
#     :param max_words: Target number of words per chunk.
#     :param overlap: Number of words to overlap between chunks.
#     """
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + max_words
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start = max(0, end - overlap)
#     logger.info("Text split into %d chunks.", len(chunks))
#     return chunks

# def get_embedding(text):
#     """
#     Generate an embedding for the given text using the Sentence Transformer model.
#     """
#     try:
#         embedding = MODEL.encode(text)
#         return np.array(embedding, dtype=np.float32)
#     except Exception as e:
#         logger.error("Error generating embedding: %s", e)
#         logger.error(traceback.format_exc())
#         raise

# def compute_embeddings_parallel(chunks):
#     """
#     Compute embeddings for each text chunk in parallel.
#     Returns a numpy array of embeddings.
#     """
#     embeddings = []
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         future_to_idx = {executor.submit(get_embedding, chunk): idx for idx, chunk in enumerate(chunks)}
#         for future in as_completed(future_to_idx):
#             idx = future_to_idx[future]
#             try:
#                 emb = future.result()
#                 embeddings.append((idx, emb))
#             except Exception as e:
#                 logger.error("Error computing embedding for chunk %d: %s", idx, e)
#     embeddings.sort(key=lambda x: x[0])
#     emb_array = np.array([emb for _, emb in embeddings])
#     logger.info("Computed embeddings for all chunks.")
#     return emb_array

# def save_embeddings(embeddings, filename):
#     """Save embeddings to a file using NumPy."""
#     try:
#         np.save(filename, embeddings)
#         logger.info("Embeddings saved to %s.", filename)
#     except Exception as e:
#         logger.error("Error saving embeddings: %s", e)
#         logger.error(traceback.format_exc())

# def save_faiss_index(index, filename):
#     """Save FAISS index to disk."""
#     try:
#         faiss.write_index(index, filename)
#         logger.info("FAISS index saved to %s.", filename)
#     except Exception as e:
#         logger.error("Error saving FAISS index: %s", e)
#         logger.error(traceback.format_exc())

# def load_embeddings(filename):
#     """Load embeddings from a file if available."""
#     if os.path.exists(filename):
#         try:
#             embeddings = np.load(filename)
#             logger.info("Embeddings loaded from %s.", filename)
#             return embeddings
#         except Exception as e:
#             logger.error("Error loading embeddings: %s", e)
#             logger.error(traceback.format_exc())
#     return None


# def load_faiss_index(filename):
#     """Load FAISS index from disk if available."""
#     if os.path.exists(filename):
#         try:
#             index = faiss.read_index(filename)
#             logger.info("FAISS index loaded from %s.", filename)
#             return index
#         except Exception as e:
#             logger.error("Error loading FAISS index: %s", e)
#             logger.error(traceback.format_exc())
#     else:
#         logger.info("No FAISS index file found at %s.", filename)
#     return None

# def build_faiss_index(embeddings):
#     """
#     Build and return a FAISS index for the given embeddings.
#     """
#     global EMBEDDING_DIM
#     EMBEDDING_DIM = embeddings.shape[1]
#     index = faiss.IndexFlatL2(EMBEDDING_DIM)
#     index.add(embeddings)
#     logger.info("FAISS index built with %d vectors.", index.ntotal)
#     return index

# def retrieve_relevant_chunks(query, top_k=3):
#     """
#     Retrieve the top_k most relevant text chunks for a given query.
#     """
#     try:
#         query_emb = get_embedding(query)
#         query_emb = np.expand_dims(query_emb, axis=0)
#         distances, indices = FAISS_INDEX.search(query_emb, top_k)
#         retrieved = [BOOK_CHUNKS[i] for i in indices[0]]
#         logger.info("Retrieved %d relevant chunks for the query.", len(retrieved))
#         return retrieved
#     except Exception as e:
#         logger.error("Error during retrieval: %s", e)
#         logger.error(traceback.format_exc())
#         raise

# def generate_response_with_gemini(query, context_chunks, max_tokens=200, temperature=0.7):
#     """
#     Generate an answer using the Gemini‑2.0‑Flash API via generate_content.
#     """
#     try:
#         context = "\n\n".join(context_chunks)
#         prompt = f"Using the following context from the book:\n\n{context}\n\nAnswer the question: {query}"
#         logger.info("Sending prompt to Gemini API using generate_content.")
#         response = gemini_client.models.generate_content(
#             model="gemini-2.0-flash",  # Adjust if needed
#             config=types.GenerateContentConfig(
#                 system_instruction=prompt
#             ),
#             contents=query
#         )
#         answer = response.text.strip()  # Adjust based on actual response structure
#         return answer
#     except Exception as e:
#         logger.error("Error generating response from Gemini: %s", e)
#         logger.error(traceback.format_exc())
#         raise

# def initialize_embeddings_for_book(chunks, embeddings_filename):
#     """
#     Load cached embeddings if available; otherwise compute and save them.
#     """
#     embeddings = load_embeddings(embeddings_filename)
#     if embeddings is None:
#         logger.info("No cached embeddings found for %s. Computing embeddings...", embeddings_filename)
#         embeddings = compute_embeddings_parallel(chunks)
#         save_embeddings(embeddings, embeddings_filename)
#     else:
#         logger.info("Using cached embeddings from %s.", embeddings_filename)
#     return embeddings

# def initialize_faiss_index_for_book(embeddings, index_filename):
#     """
#     Load cached FAISS index if available; otherwise build and save the index.
#     """
#     index = load_faiss_index(index_filename)
#     if index is None:
#         logger.info("No cached FAISS index found. Building index...")
#         index = build_faiss_index(embeddings)
#         save_faiss_index(index, index_filename)
#     else:
#         logger.info("Using cached FAISS index from %s.", index_filename)
#     return index

# # -----------------------------
# # Book Initialization Routine
# # -----------------------------
# def initialize_system_for_book(pdf_path):
#     """
#     Initialize the system for a given PDF book.
#     This includes loading the book, chunking, computing (or loading) embeddings,
#     and building (or loading) a FAISS index.
#     """
#     global BOOK_CHUNKS, FAISS_INDEX, MODEL, CURRENT_BASE_FILENAME

#     try:
#         logger.info("Initializing system for book: %s", pdf_path)
#         # Load and process the book from PDF
#         book_text = load_book_from_pdf(pdf_path)
#         BOOK_CHUNKS = chunk_text(book_text)
        
#         # Derive a base filename from the PDF path for caching
#         CURRENT_BASE_FILENAME = os.path.splitext(os.path.basename(pdf_path))[0]
#         embeddings_filename = f"{CURRENT_BASE_FILENAME}_embeddings.npy"
#         index_filename = f"{CURRENT_BASE_FILENAME}_faiss.index"
        
#         # Load Sentence Transformer model (using public model "all-mpnet-base-v2")
#         logger.info("Loading Sentence Transformer model (all-mpnet-base-v2)...")
#         MODEL = SentenceTransformer("all-mpnet-base-v2")
        
#         # Initialize embeddings and FAISS index
#         embeddings = initialize_embeddings_for_book(BOOK_CHUNKS, embeddings_filename)
#         FAISS_INDEX = initialize_faiss_index_for_book(embeddings, index_filename)
#         logger.info("System initialization complete for book: %s", pdf_path)
#     except Exception as e:
#         logger.error("Failed to initialize system for book: %s", e)
#         logger.error(traceback.format_exc())
#         raise

# # -----------------------------
# # Flask App Setup
# # -----------------------------
# app = Flask(__name__)

# @app.route("/chat", methods=["POST"])
# def chat_endpoint():
#     """
#     Flask endpoint to answer queries based on the currently loaded book.
#     Expected JSON payload: {"query": "Your question here"}
#     """
#     try:
#         data = request.get_json()
#         query = data.get("query", "")
#         if not query:
#             return jsonify({"error": "Query not provided."}), 400
        
#         retrieved_chunks = retrieve_relevant_chunks(query, top_k=3)
#         answer = generate_response_with_gemini(query, retrieved_chunks)
#         return jsonify({"answer": answer})
#     except Exception as e:
#         logger.error("Error in /chat endpoint: %s", e)
#         logger.error(traceback.format_exc())
#         return jsonify({"error": "Internal server error."}), 500

# # -----------------------------
# # Main Entry Point
# # -----------------------------
# if __name__ == "__main__":
#     # For testing, you can specify a PDF book path.
#     # To switch between books, call initialize_system_for_book() with the new PDF path.
#     PDF_BOOK_PATH = "The_Total_Money_Makeover.pdf"  # Update this path accordingly
    
#     try:
#         # Initialize system for the selected book
#         initialize_system_for_book(PDF_BOOK_PATH)
        
#         # Command-line interactive testing:
#         while True:
#             user_query = input("\nEnter your question (or type 'exit' to quit): ")
#             if user_query.strip().lower() == "exit":
#                 break
#             try:
#                 retrieved_chunks = retrieve_relevant_chunks(user_query, top_k=3)
#                 answer = generate_response_with_gemini(user_query, retrieved_chunks)
#                 print("\nAnswer:\n", answer)
#             except Exception as q_e:
#                 print("Error processing your query. Please try again.")
        
#         # To run the Flask app, uncomment the following line:
#         # app.run(host="0.0.0.0", port=5000, debug=False)
#     except Exception as init_e:
#         logger.error("Exiting due to initialization error.")
