from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.logger_config import create_logger
from schemas.llm_output_schemas import QueryFilterSchema
from src.utils.load_settings import load_env

logger = create_logger("Extracting metadata from query")
gcp_config, _ = load_env()

def extract_query_metadata_filters(user_query: str, prompts: dict, config: dict):
    """Parses a natural language user query to extract page-level metadata

    Args:
        user_query: The raw input query from the user
        prompts: Dictionary containing prompts templates
        config: Global configuration dictionary containing model profiles and parameters.

    Returns:
        A tuple containing:
            - The metadata-free query for vector search.
            - Chroma DB-compliant metadata filter dictionary or None if no constraints are found.
    """

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

    clean_query = parsed_filters.clean_query

    if parsed_filters.start_page and parsed_filters.end_page:
        logger.info(f"Applying page range filter: pages {parsed_filters.start_page} to {parsed_filters.end_page}")
        return clean_query, {
            "$and": [
                {"page_number": {"$gte": int(parsed_filters.start_page)}},
                {"page_number": {"$lte": int(parsed_filters.end_page)}}
            ]
        }
    else:
        logger.info("No page constraints detected. Performing full database search.")
        return user_query, None
