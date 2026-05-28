from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from src.utils.tools import create_logger, load_env, load_yaml
import time

gcp_conf, _ = load_env()

logger = create_logger("Vector database")

def create_vector_database(chunks, path:str, emb_model:str, batch_size:int = 50):
    """
    Initializes a Chroma vector database from text chunks in batches.

    Args:
        chunks: A list of LangChain `Document` objects to be indexed.
        path: Local directory path where the Chroma DB files will be saved.
        emb_model: Name of embedding model
        batch_size (optional): Number of documents to upload per batch. 

    Returns:
        Chroma: A fully populated LangChain `Chroma` vector database instance.
    """
    
    total_chunks = len(chunks)
    db_chroma = Chroma(
        embedding_function=VertexAIEmbeddings(model_name=emb_model, project=gcp_conf.project_id), 
        persist_directory=path,
        collection_metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, total_chunks, batch_size):
        time.sleep(1)
        next_batch = chunks[i : i + batch_size]
        logger.info(f"Indexing chunks ({i}/{total_chunks})...")
        db_chroma.add_documents(documents=next_batch)

    return db_chroma