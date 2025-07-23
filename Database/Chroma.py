from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from Preprocess.PDFprocess import PDFprocessor
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import torch
embeddings = HuggingFaceEmbeddings(
    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
    model_kwargs={"device": "cuda"}  # or "cpu"
)
def build_rag_index(pdf:str,embedding=embeddings,collection_name:str="10ms_collection",persist_directory:str="./chroma_langchain_db",chunk_size:int=200,chunk_overlap:int=20):
    """
    Build a RAG index from a PDF file.
    
    Args:
        pdf (str): Path to the PDF file.
        embedding: Embedding function to use.
        collection_name (str): Name of the collection in Chroma.
        persist_directory (str): Directory to persist the Chroma database.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    """
    # Initialize PDF processor
    proc=PDFprocessor(pdf_path=pdf)
    proc.load_pdf()
    proc.clean()
    proc.clean_bangla(split_sentences=True)
    #proc.visualize_paragraph_lengths()
    chunks=proc.chunk(chunk_size,chunk_overlap)
    #proc.visualize_chunk_lengths()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,  #save data locally
    )
    vector_store.add_documents(documents=chunks)
    return vector_store,chunks
def extract_embeddings(chunks, embedding=embeddings) -> np.ndarray:
    """
    Given a list of LangChain Document chunks and an embedding wrapper,
    return a 2D NumPy array of shape (n_chunks, embedding_dim).
    """
    texts = [doc.page_content for doc in chunks]
    # embed_documents returns a list of vectors
    vecs = embedding.embed_documents(texts)
    return np.stack(vecs, axis=0)
def extract_embeddings_batched(chunks, embedding=embeddings, batch_size=256):
    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = [c.page_content for c in chunks[i : i+batch_size]]
        vecs = embedding.embed_documents(batch)
        all_vecs.append(torch.tensor(vecs))  # temp on CPU
    return torch.cat(all_vecs, dim=0).numpy()
