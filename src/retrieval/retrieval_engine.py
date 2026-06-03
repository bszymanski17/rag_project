from langchain_chroma import Chroma
from langfuse import observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.utils.load_settings import load_env, create_logger
from src.retrieval.context_formatter import format_documents_to_context



gcp_config, _ = load_env()
logger = create_logger("Retrieval")

@observe()
def retrieval_content(user_query: str, emb_model: str, db_path: str, retriever_top_k: int, distance_threshold: float):
    """Performs semantic search retrieval from the vector database.

    Args:
        user_query: The search query or financial question from the user.
        emb_model: The deployment name of the Vertex AI embedding model.
        db_path: The local directory path where the Chroma DB is stored.
        retriever_top_k: The number of top relevant document chunks to retrieve.

    Returns:
        A tuple containing:
            - formatted_context (str): Combined text content of chunks for the LLM prompt.
            - raw_chunks (List[str]): List of raw page contents from the retrieved documents.
            - similarity_scores (List[float]): List of floats representing the distance scores.
    """

    logger.info(f"Starting retrieval process.")
    embedding_model = GoogleGenerativeAIEmbeddings(model=emb_model, project=gcp_config.project_id, vertexai=True)  
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    docs_with_scores = db.similarity_search_with_score(user_query, k=retriever_top_k)

    valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores if float(score) <= distance_threshold]
    logger.info(f"Found {len(valid_results)} chunks meeting the distance threshold (<= {distance_threshold}).")


    if not valid_results and docs_with_scores:
        valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores[:5]]
    
    valid_docs = [doc for doc, score in valid_results]
    raw_chunks = [doc.page_content for doc, score in valid_results]
    similarity_scores = [score for doc, score in valid_results]

    logger.info(f"Successfully post-processed {len(valid_results)} chunks. Best score: {similarity_scores[0] if similarity_scores else 'N/A'}")
    
    return format_documents_to_context(valid_docs, similarity_scores), raw_chunks, similarity_scores