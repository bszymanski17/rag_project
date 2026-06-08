import os

from src.utils.load_settings import load_yaml
from src.utils.logger_config import create_logger
from src.indexing.database_initializer import create_vector_database

logger = create_logger("Vector database")
config = load_yaml("config/main.yaml")

def initialize_database() -> bool:
    """Initialize the vector database from disk or build a new instance.

    Returns:
        bool: True if the database already exists or was successfully built,
              False if an error occurred during the creation process.
    """
    if os.path.exists(config["vector_db"]["db_path"]) and os.listdir(config["vector_db"]["db_path"]):
        logger.info("Vector database already exists.")
        return True
    else:
        try:
            logger.info("Creating vector database with embeddings...")

            if config["vector_db"]["chunk_approach"] == "baseline":
                from src.indexing.chunk_strategies.baseline.chunker import chunk_pdf
                chunks = chunk_pdf(config["vector_db"]["knowladge_path"], chunk_size=config["vector_db"]["chunk_size"], chunk_overlap=config["vector_db"]["chunk_overlap"])
            elif config["vector_db"]["chunk_approach"] == "multimodal_split":
                from src.indexing.chunk_strategies.multimodal_split.chunker import chunk_pdf
                chunks = chunk_pdf(pdf_path=config["vector_db"]["knowladge_path"], blacklist=config["vector_db"]["elements_blacklist"])
            elif config["vector_db"]["chunk_approach"] == "multimodal_inline":
                from src.indexing.chunk_strategies.multimodal_inline.chunker import chunk_pdf
                chunks = chunk_pdf(pdf_path=config["vector_db"]["knowladge_path"], blacklist=config["vector_db"]["elements_blacklist"])
            elif config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
                from src.indexing.chunk_strategies.multimodal_inline.page_chunker import chunk_pdf
                chunks = chunk_pdf(pdf_path=config["vector_db"]["knowladge_path"], blacklist=config["vector_db"]["elements_blacklist"])
            else:
                logger.error("Invalid chunk approach.")
                raise ValueError("Invalid chunk approach.")
            
            chroma_db = create_vector_database(chunks, config["vector_db"]["db_path"], emb_model=config['models']['embedding_model'])
            
            logger.info("Vector database created successfully.")
            return True
        
        except Exception as e:
            logger.error(f"Error during creating vector database: {str(e)}")            
            return False
        


