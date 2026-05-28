from src.utils.tools import load_yaml, load_env
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from src.retrieval.format_docs import format_docs
from langfuse import observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings


gcp_config, _ = load_env()

@observe()
def run_retrieval(user_query: str, emb_model: str, db_path: str, retriever_top_k: int, distance_threshold: float) -> str:
    """
    Performs semantic search retrieval from the vector database.

    Args:
        user_query: The search query or financial question from the user.
        emb_model: The deployment name of the Vertex AI embedding model.
        db_path: The local directory path where the Chroma DB is stored.
        retriever_top_k: The number of top relevant document chunks to retrieve.

    Returns:
        str: A single formatted string containing the combined text content of 
             the retrieved document chunks ready to be used as LLM context.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model=emb_model, project=gcp_config.project_id, vertexai=True)  
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    docs_with_scores = db.similarity_search_with_score(user_query, k=retriever_top_k)

    valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores if float(score) <= distance_threshold]

    if not valid_results and docs_with_scores:
        valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores[:5]]
    
    valid_docs = [doc for doc, score in valid_results]
    raw_chunks = [doc.page_content for doc, score in valid_results]
    similarity_scores = [score for doc, score in valid_results]

    
    return format_docs(valid_docs, similarity_scores), raw_chunks, similarity_scores