from langchain_chroma import Chroma
from langfuse import observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from src.utils.load_settings import load_env, create_logger, load_yaml
from src.retrieval.context_formatter import format_documents_to_context, format_documents_to_context_with_page_number
from schemas.llm_output_schemas import QueryFilterSchema


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

    if config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
        logger.info(f"Parsing user query for metadata constraints...")
        llm = ChatGoogleGenerativeAI(
            model=config["models"]["desc_model"],
            location="global",
            vertexai=True,
            project=gcp_config.project_id,
            timeout=90,
            max_retries=2,
            temperature=0
            )
        
        structured_llm = llm.with_structured_output(QueryFilterSchema)
  
        messages = [
            ("system", prompts["extract_metadata"]["system_prompt"]),
            ("human", prompts["extract_metadata"]["user_prompt"].format(user_query=user_query))
        ]
        parsed_filters = structured_llm.invoke(messages)

        user_query = parsed_filters.clean_query

        if parsed_filters.start_page and parsed_filters.end_page:
            chroma_filter = {
                "$and": [
                    {"page_number": {"$gte": int(parsed_filters.start_page)}},
                    {"page_number": {"$lte": int(parsed_filters.end_page)}}
                ]
            }
            logger.info(f"Applying page range filter: pages {parsed_filters.start_page} to {parsed_filters.end_page}")
        elif parsed_filters.start_page:
            chroma_filter = {"page_number": {"$eq": int(parsed_filters.start_page)}}
            logger.info(f"Applying exact page filter: page {parsed_filters.start_page}")
        else:
            logger.info("No page constraints detected. Performing full database search.")

    docs_with_scores = db.similarity_search_with_score(user_query, k=retriever_top_k, filter=chroma_filter)

    
    if not config["reranker"]["use_reranker"]:
        valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores if float(score) <= distance_threshold]
        logger.info(f"Found {len(valid_results)} chunks meeting the distance threshold (<= {distance_threshold}).")

        if not valid_results and docs_with_scores:
            valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores[:5]]
        valid_docs = [doc for doc, score in valid_results]
        raw_chunks = [doc.page_content for doc, score in valid_results]
        similarity_scores = [score for doc, score in valid_results]

        logger.info(f"Successfully post-processed {len(valid_results)} chunks. Best score: {similarity_scores[0] if similarity_scores else 'N/A'}")

    else:
        logger.info(f"Executing Cross-Encoder re-ranking on {len(docs_with_scores)} candidate chunks...")
    
        pairs = [[user_query, doc.page_content] for doc, _ in docs_with_scores]
        rerank_scores = reranker.predict(pairs)
        reranked_docs = list(zip([doc for doc, _ in docs_with_scores], rerank_scores))
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        final_results = reranked_docs[:config["reranker"]["top_n"]]
        valid_docs = [doc for doc, score in final_results]
        raw_chunks = [doc.page_content for doc, score in final_results]
        similarity_scores = [round(float(score), 4) for doc, score in final_results]

        logger.info(f"Successfully re-ranked results. Best Cross-Encoder score: {similarity_scores[0] if similarity_scores else 'N/A'}")
        
    if config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
        return format_documents_to_context_with_page_number(valid_docs, similarity_scores), raw_chunks, similarity_scores
    return format_documents_to_context(valid_docs, similarity_scores), raw_chunks, similarity_scores