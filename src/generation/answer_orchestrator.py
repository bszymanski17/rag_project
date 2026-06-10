from typing import Iterator
from src.utils.load_settings import load_yaml
from src.utils.logger_config import create_logger

# Importujemy oba niezależne komponenty generujące
from src.generation.answer_generator import generate_streamed_response
from src.generation.answer_generator_multimodal import generate_multimodal_streamed_response

config = load_yaml("config/main.yaml")
logger = create_logger("Answer Orchestrator")

def route_generation_stream(context: str, question: str, model: str) -> Iterator[str]:
    """Routes the generation request to the proper engine based on configuration.

    Args:
        context: Text chunks or newline-separated image paths.
        question: The user's query.
        model: Deployment name of the Gemini LLM.

    Returns:
        Iterator[str]: Token stream from the selected model chain.
    """
    chunk_approach = config["vector_db"]["chunk_approach"]

    if chunk_approach == "visual_multimodal":
        logger.info("Routing stream generation to Multimodal (Visual) engine.")
        return generate_multimodal_streamed_response(context=context, question=question, model=model)
    
    else:
        logger.info("Routing stream generation to Traditional Text engine.")
        return generate_streamed_response(context=context, question=question, model=model)