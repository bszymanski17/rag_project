from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant, FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores.utils import DistanceStrategy

from src.utils.load_settings import load_env, load_yaml
from src.utils.logger_config import create_logger


gcp_conf, _ = load_env()
config = load_yaml("config/main.yaml")
logger = create_logger("Vector database")

def create_vector_database(chunks, provider: str, path:str, emb_model:str, batch_size:int = 50):
    """Initializes a vector database (Chroma, Qdrant, FAISS) from text chunks in batches.

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

    embeddings = GoogleGenerativeAIEmbeddings(
        model=emb_model, 
        project=gcp_conf.project_id, 
        vertexai=True
    )

    vector_db = None
        
    for i in range(0, total_chunks, batch_size):
        time.sleep(1) 
        next_batch = chunks[i : i + batch_size]
        logger.info(f"Indexing chunks ({i}/{total_chunks})...")
        
        if vector_db is None:
            if provider == "chroma":
                vector_db = Chroma.from_documents(
                    documents=next_batch,
                    embedding=embeddings,
                    persist_directory=path,
                    collection_metadata={"hnsw:space": config["vector_db"]["similarity_metric"]} 
                )
            elif provider == "qdrant":
                collection_name = config["vector_db"].get("qdrant_collection_name", "my_collection")
                vector_db = QdrantVectorStore.from_documents(
                    embedding=embeddings,
                    collection_name=collection_name,
                    path=path,
                    documents=next_batch,
                    distance_strategy=config["vector_db"]["similarity_metric"].upper()
                )
            elif provider == "faiss":
                vector_db = FAISS.from_documents(
                    documents=next_batch,
                    embedding=embeddings,
                    distance_strategy=DistanceStrategy.COSINE
                )
        else:
            vector_db.add_documents(documents=next_batch)

    if provider == "faiss" and vector_db is not None:
        logger.info(f"Saving FAISS index locally to: {path}")
        vector_db.save_local(path)
        
    logger.info(f"Successfully populated vector database. Indexed {total_chunks} chunks.")

    return vector_db