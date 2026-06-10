import os
from byaldi import RAGMultiModalModel
from src.utils.load_settings import load_yaml, create_logger
import torch

logger = create_logger("Multimodal Database")
config = load_yaml("config/main.yaml")

def create_image_vector_db(images_dir: str, index_name: str = "colpali_index") -> RAGMultiModalModel:
    """
    Loads pre-rendered page images from a directory, generates multimodal 
    embeddings using ColPali, and builds a late-interaction vector index.
    
    Args:
        images_dir: Path to the directory containing the page images.
        index_name: Name of the folder where the resulting index will be saved.
        
    Returns:
        The loaded and populated Byaldi RAG instance.
    """
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        logger.error(f"Images directory '{images_dir}' is empty or does not exist!")
        raise FileNotFoundError(f"No source images found in {images_dir}")

    logger.info(f"Initiating Multimodal Vector DB creation using ColPali...")
    logger.info(f"Source images directory: '{images_dir}'")
    logger.info(f"Target index name: '{index_name}'")

    logger.info("Loading ColPali visual language model (this may take a moment)...")
    
    model_name = config["vector_db"].get("multimodal_model", "vidore/colqwen2-v1.0")

    if torch.backends.mps.is_available():
        target_device = "mps"
        logger.info("Apple Silicon GPU (MPS) detected. Running indexer on GPU!")
    else:
        target_device = "cpu"
        logger.info("MPS not available. Falling back to CPU mode (slower but stable).")
    RAG = RAGMultiModalModel.from_pretrained(model_name, device=target_device)

    logger.info("Generating multimodal multi-vector embeddings and filling the index...")
    
    RAG.index(
        input_path=images_dir,
        index_name=index_name,
        store_collection_with_index=True, 
        overwrite=True
    )

    logger.info(f"Successfully populated multimodal vector database. Index saved to '.byaldi/{index_name}/'")
    return RAG