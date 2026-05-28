from src.db_handler.chunk_doc import chunk_doc
from src.db_handler.create_db import create_vector_database
from src.utils.tools import create_logger, load_yaml
import os

logger = create_logger("Vector database")

"""
This script acts as the main pipeline orchestrator to build the vector database
for the RAG application:
    1. Extracts text from the raw financial PDF report.
    2. Transforms the text into chunks.
    3. Generates vector embeddings and loads them into a persistent Chroma database.


"""
config = load_yaml("config/config.yaml")

def initialize_database() -> bool:
    """
    If a persistent and non-empty directory already exists, it skips the build process. Otherwise, 
    it reads the raw knowledge base document, splits it into smaller text chunks, 
    computes semantic embeddings, and creates a new Chroma DB instance.

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
            
            chunks = chunk_doc(config["vector_db"]["knowladge_path"], chunk_size=config["vector_db"]["chunk_size"], chunk_overlap=config["vector_db"]["chunk_overlap"])
            chroma_db = create_vector_database(chunks, config["vector_db"]["db_path"], emb_model=config['models']['embedding_model'])
            
            logger.info("Vector database created successfully.")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during creating vector database: {str(e)}")            
            return False

