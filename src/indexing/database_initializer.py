from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

from src.utils.load_settings import load_env, load_yaml
from src.utils.logger_config import create_logger


gcp_conf, _ = load_env()
config = load_yaml("config/main.yaml")
logger = create_logger("Vector database")

def create_vector_database(chunks, path:str, emb_model:str, batch_size:int = 50):
    """Initializes a Chroma vector database from text chunks in batches.

    Args:
        chunks: A list of LangChain `Document` objects to be indexed.
        path: Local directory path where the Chroma DB files will be saved.
        emb_model: Name of embedding model
        batch_size (optional): Number of documents to upload per batch. 

    Returns:
        Chroma: A fully populated LangChain `Chroma` vector database instance.
    """
    
    total_chunks = len(chunks)
    logger.info(f"Initiating vector database creation at path: '{path}'")
    db_chroma = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(
        model=emb_model, 
        project=gcp_conf.project_id, 
        vertexai=True
    ), 
        persist_directory=path,
        collection_metadata={"hnsw:space": config["vector_db"]["similarity_metric"]} 
    )

    for i in range(0, total_chunks, batch_size):
        time.sleep(1)
        next_batch = chunks[i : i + batch_size]
        logger.info(f"Indexing chunks ({i}/{total_chunks})...")
        db_chroma.add_documents(documents=next_batch)

    logger.info(f"Successfully populated vector database. Indexed {total_chunks} chunks.")

    return db_chroma