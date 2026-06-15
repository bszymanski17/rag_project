from langchain_chroma import Chroma
from langfuse import observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Qdrant, FAISS
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from src.utils.load_settings import load_env, create_logger, load_yaml
from src.retrieval.context_formatter import format_documents_to_context, format_documents_to_context_with_page_number
from src.retrieval.query_metada_extractor import extract_query_metadata_filters
from src.retrieval.chunk_filtering.reranker import rerank_chunks
from src.retrieval.chunk_filtering.retrieval_filter import filter_chunks_by_threshold
from src.retrieval.visual_retrieval import retrieve_visual_content


config = load_yaml("config/main.yaml")
prompts = load_yaml("prompts/main.yaml")
gcp_config, _ = load_env()
logger = create_logger("Retrieval")

logger.info("Initializing Cross-Encoder re-ranker...")
# reranker = CrossEncoder("BAAI/bge-reranker-base")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=8192, device="cpu")

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

    if config["vector_db"]["chunk_approach"] == "visual_multimodal":
        return retrieve_visual_content(user_query=user_query, retriever_top_k=retriever_top_k)
    
    elif config["vector_db"]["chunk_approach"] == "visual_database":
        from src.retrieval.visual_database_retrieval import retrieve_maxsim
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch
        import streamlit as st

        logger.info("Retrieving content using Strict MaxSim logic...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0").to(device).eval()
        
        strict_db_path = config["vector_db"]["db_path"]
        raw_chunks, best_patches_dict = retrieve_maxsim(
            query=user_query, 
            db_path=strict_db_path, 
            model=model, 
            processor=processor, 
            device=device, 
            top_k=config["retriever"]["top_k"]
        )
        import streamlit as st
        st.session_state["strict_best_patches"] = best_patches_dict
        context = "\n".join(raw_chunks)
        score = [1.0] * len(raw_chunks)
        
        return context, raw_chunks, score

    provider = config["vector_db"]["provider"]
    embedding_model = GoogleGenerativeAIEmbeddings(model=emb_model, project=gcp_config.project_id, vertexai=True)
    if provider == "chroma": 
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    elif provider == "qdrant":
        collection_name = config["vector_db"].get("qdrant_collection_name", "my_collection")
        db = QdrantVectorStore.from_existing_collection(embedding=embedding_model, collection_name=collection_name,path=db_path)
    elif provider == "faiss":
        db = FAISS.load_local(
            folder_path=db_path, 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
        )
    else:
        raise ValueError(f"Unknown vector DB provider: {provider}")
    
    start_page = None
    end_page = None
    db_filter = None

    # metada extraction (start/end page number)
    if config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
        user_query, start_page, end_page = extract_query_metadata_filters(user_query=user_query, prompts=prompts, config=config)

    if start_page is not None and end_page is not None:
        if provider == "chroma":
            db_filter = {
                "$and": [
                    {"page_number": {"$gte": start_page}},
                    {"page_number": {"$lte": end_page}}
                ]
            }
        elif provider == "qdrant":
            from qdrant_client.models import Filter, FieldCondition, Range
            db_filter = Filter(
                must=[
                    FieldCondition(key="metadata.page_number", range=Range(gte=start_page, lte=end_page))
                ]
            )
        elif provider == "faiss":
            db_filter = lambda doc: start_page <= int(doc.metadata.get("page_number", -1)) <= end_page

    docs_with_scores = db.similarity_search_with_score(user_query, k=retriever_top_k, filter=db_filter)
    if config["vector_db"]["provider"] != "chroma":
        docs_with_scores = [(doc, float(1.0 - score)) for doc, score in docs_with_scores]

    # re-ranker / filtering by threshold
    if config["reranker"]["use_reranker"]:
        valid_docs, raw_chunks, similarity_scores = rerank_chunks(user_query=user_query, docs_with_scores=docs_with_scores, config=config, reranker=reranker)
    else:
        valid_docs, raw_chunks, similarity_scores = filter_chunks_by_threshold(docs_with_scores=docs_with_scores, distance_threshold=distance_threshold)
        
    if config["vector_db"]["chunk_approach"] == "multimodal_inline_metadata":
        return format_documents_to_context_with_page_number(valid_docs, similarity_scores), raw_chunks, similarity_scores
    return format_documents_to_context(valid_docs, similarity_scores), raw_chunks, similarity_scores