def format_documents_to_context(docs: list, scores: list = None) -> str:
    """Formats a list of LangChain Document objects into a single context string.

    Args:
        docs: A list of LangChain `Document` objects retrieved from the vector database.

    Returns:
        str: A single string containing all document text chunks separated 
            by double newlines.
    """
    formatted_chunks = []
    
    for idx, doc in enumerate(docs):
        if scores and idx < len(scores):
            chunk_header = f"[Chunk {idx + 1} | Score: {scores[idx]}]\n"
        else:
            chunk_header = f"[Chunk {idx + 1}]\n"
        chunk_content = doc.page_content
            
        formatted_chunks.append(f"{chunk_header}{chunk_content}")
        
    return "\n\n".join(formatted_chunks)