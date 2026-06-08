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




def format_documents_to_context_with_page_number(docs: list, scores: list = None) -> str:
    """Formats a list of LangChain Document objects into a metadata-enriched context string,
    explicitly injecting source page numbers for the LLM.
    
    Args:
        docs: A list of LangChain `Document` objects with 'page_number' present in their metadata.
        scores: Optional list of similarity scores corresponding to each document.

    Returns:
        A single string containing all document text chunks prefixed with chunk index, 
            extracted page number, and scores (if provided), separated by double newlines.
    """
    formatted_chunks = []
    
    for idx, doc in enumerate(docs):
        page_num = doc.metadata.get("page_number", "Unknown")
        
        if scores and idx < len(scores):
            chunk_header = f"[Chunk {idx + 1} | Page: {page_num} | Score: {scores[idx]}]\n"
        else:
            chunk_header = f"[Chunk {idx + 1} | Page: {page_num}]\n"
            
        chunk_content = doc.page_content
        formatted_chunks.append(f"{chunk_header}{chunk_content}")
        
    return "\n\n".join(formatted_chunks)