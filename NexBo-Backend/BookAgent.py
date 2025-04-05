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
