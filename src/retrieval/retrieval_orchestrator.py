from langchain_chroma import Chroma
from langfuse import observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from src.utils.load_settings import load_env, create_logger, load_yaml
from src.retrieval.context_formatter import format_documents_to_context, format_documents_to_context_with_page_number
from schemas.llm_output_schemas import QueryFilterSchema
from src.retrieval.query_metada_extractor import extract_query_metadata_filters
from src.retrieval.reranker import rerank_chunks
from src.retrieval.retrieval_filter import filter_chunks_by_threshold


config = load_yaml("config/main.yaml")
prompts = load_yaml("prompts/main.yaml")
gcp_config, _ = load_env()
logger = create_logger("Retrieval")

logger.info("Initializing Cross-Encoder re-ranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

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
    
    chroma_filter = None

    # metada extraction (start/end page number)
    if config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
        user_query, chroma_filter = extract_query_metadata_filters(user_query=user_query, prompts=prompts, config=config)
    docs_with_scores = db.similarity_search_with_score(user_query, k=retriever_top_k, filter=chroma_filter)

    # re-ranker / filtering by threshold
    if config["reranker"]["use_reranker"]:
        valid_docs, raw_chunks, similarity_scores = rerank_chunks(user_query=user_query, docs_with_scores=docs_with_scores, config=config, reranker=reranker)
    else:
        valid_docs, raw_chunks, similarity_scores = filter_chunks_by_threshold(docs_with_scores=docs_with_scores, distance_threshold=config["retriever"]["distance_treshold"])
        
    if config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
        return format_documents_to_context_with_page_number(valid_docs, similarity_scores), raw_chunks, similarity_scores
    return format_documents_to_context(valid_docs, similarity_scores), raw_chunks, similarity_scores