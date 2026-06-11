from byaldi import RAGMultiModalModel
from src.utils.load_settings import create_logger, load_yaml
import os
import torch
import streamlit as st

@st.cache_resource
def get_cached_multimodal_model(index_name: str, device: str) -> RAGMultiModalModel:
    """Loads and caches a multimodal RAG model from a specified index.

    Args:
        index_name: Path or identifier of the pre-built index.
        device: Device to load the model onto (e.g., "mps", "cuda").

    Returns:
        The initialized multimodal RAG model instance.
    """
    logger.info(f"Loading ColPali model weights into {device} memory...")
    return RAGMultiModalModel.from_index(index_name, device=device)

logger = create_logger("Visual multimodal retriever")
config = load_yaml("config/main.yaml")

def retrieve_visual_content(user_query: str, retriever_top_k: int) -> tuple[str, list[str], list[float]]:
    """Performs visual semantic search using ColPali.

    Args:
        user_query: The search query from the user.
        retriever_top_k: The number of top relevant document pages to retrieve.

    Returns:
        A tuple containing:
            - formatted_context: Image paths joined by newlines.
            - raw_chunks: List of raw image file paths.
            - similarity_scores: List of floats representing MaxSim scores.
    """
    logger.info("Performing multimodal retrieval using ColPali (Late Interaction)...")
    
    index_name = config["vector_db"]["qdrant_collection_name"]
    interim_images_dir = config["vector_db"]["interim_images_dir"]

    if torch.backends.mps.is_available():
        target_device = "mps"
    else:
        target_device = "cpu"
    
    RAG = get_cached_multimodal_model(index_name=index_name, device=target_device)
    
    results = RAG.search(user_query, k=retriever_top_k)
    
    raw_chunks = []
    similarity_scores = []
    
    for res in results:
        image_path = os.path.join(interim_images_dir, f"page_{res.doc_id}.jpg")
        
        raw_chunks.append(image_path)
        similarity_scores.append(float(res.score))
        
    formatted_context = "\n".join(raw_chunks)
    
    logger.info(f"Multimodal retrieval completed. Found {len(raw_chunks)} relevant pages.")
    return formatted_context, raw_chunks, similarity_scores