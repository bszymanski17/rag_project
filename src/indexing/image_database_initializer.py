import os
from byaldi import RAGMultiModalModel
from src.utils.load_settings import load_yaml, create_logger
import torch
import re

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

    logger.info("Initiating Multimodal Vector DB creation using ColPali...")

    all_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]

    sorted_image_paths = sorted(
        all_files,
        key=lambda x: int(re.search(r"page_(\d+)", os.path.basename(x)).group(1))
    )

    doc_ids = [
        int(re.search(r"page_(\d+)", os.path.basename(p)).group(1))
        for p in sorted_image_paths
    ]

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
    input_path=sorted_image_paths[0],
    index_name=index_name,
    doc_ids=[doc_ids[0]],
    store_collection_with_index=True,
    overwrite=True
)

# Reszta plików dodawana po jednym
    for path, doc_id in zip(sorted_image_paths[1:], doc_ids[1:]):
        RAG.add_to_index(
            input_item=path,
            store_collection_with_index=True,
            doc_id=doc_id
        )

    logger.info(f"Successfully populated multimodal vector database. Index saved to '.byaldi/{index_name}/'")
    return RAG