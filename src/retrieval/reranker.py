from src.utils.logger_config import create_logger

logger = create_logger("Reranker")

def rerank_chunks(user_query: str, docs_with_scores, config: dict, reranker):
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
    
    return valid_docs, raw_chunks, similarity_scores