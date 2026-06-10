from src.utils.load_settings import load_yaml, create_logger
from src.indexing.database_orchestrator import initialize_database
from src.retrieval.retrieval_orchestrator import retrieval_content
from src.generation.answer_orchestrator import route_generation_stream

config = load_yaml("config/main.yaml")
logger = create_logger("Orchestrator")

def main(user_prompt: str, evaluation_mode: bool = False):
    """
    Executes the complete RAG pipeline.

    Args:
        user_prompt (str): The financial question or query entered by the user.

    Returns:
        Iterator[str]: A stream generator yielding partial text chunks of the
    """
    if not initialize_database():
        logger.error("Database initialization failed. The RAG pipeline cannot proceed.")
        raise RuntimeError("Database initialization failed. The RAG pipeline cannot proceed.")

    context, raw_chunks, score = retrieval_content(user_prompt, config["models"]["embedding_model"], config["vector_db"]["db_path"], config["retriever"]["top_k"], config["retriever"]["distance_treshold"])
   
    if evaluation_mode:
        return route_generation_stream(context, user_prompt, config['models']['main_model']), raw_chunks
    
    return route_generation_stream(context, user_prompt, config['models']['main_model'])


