from src.utils.logger_config import create_logger

logger = create_logger("Chunks filtering")

def filter_chunks_by_threshold(docs_with_scores, distance_threshold):
    """Filters retrieved database chunks based on a maximum distance threshold,

    Args:
        docs_with_scores: A list of tuples containing the retrieved LangChain Document objects and their 
            corresponding vector distance scores.
        distance_threshold: The maximum allowable distance score for a
          chunk to be included in the results.

    Returns:
        A tuple containing three elements:
            - Filtered LangChain Document objects that passed the threshold (or the fallback top 5).
            - The raw text content (page_content) extracted from the validated documents.
            - Rounded distance scores corresponding to the finalselected documents.
    """
    valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores if float(score) <= distance_threshold]
    logger.info(f"Found {len(valid_results)} chunks meeting the distance threshold (<= {distance_threshold}).")

    if not valid_results and docs_with_scores:
        valid_results = [(doc, round(float(score), 4)) for doc, score in docs_with_scores[:5]]
    valid_docs = [doc for doc, score in valid_results]
    raw_chunks = [doc.page_content for doc, score in valid_results]
    similarity_scores = [score for doc, score in valid_results]

    logger.info(f"Successfully post-processed {len(valid_results)} chunks. Best score: {similarity_scores[0] if similarity_scores else 'N/A'}")

    return valid_docs, raw_chunks, similarity_scores