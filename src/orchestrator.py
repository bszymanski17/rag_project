from src.generation.generator import generate_data
from src.retrieval.retrieval_engine import run_retrieval
from src.utils.tools import load_yaml
from src.db_handler.db_orchestrator import initialize_database
from typing import Iterator

config = load_yaml("config/config.yaml")

def main(user_prompt: str, evaluation_mode: bool = False):
    """
    Executes the complete RAG pipeline.

    Args:
        user_prompt (str): The financial question or query entered by the user.

    Returns:
        Iterator[str]: A stream generator yielding partial text chunks of the
    """
    initialize_database()

    context, raw_chunks, score = run_retrieval(user_prompt, config["models"]["embedding_model"], config["vector_db"]["db_path"], config["retriever"]["top_k"], config["retriever"]["distance_treshold"])
   
    if evaluation_mode:
        return generate_data(context, user_prompt, config['models']['main_model']), raw_chunks
    
    return generate_data(context, user_prompt, config['models']['main_model'])

import time
if __name__ == "__main__":
    generator = main("What was the predominant currency in IFC's disbursed loxan portfolio as of June 30, 2024, and what was its value??")
    
    print("Rozpoczynam strumieniowanie:")
    for chunk in generator:
        print(chunk, end="", flush=True)
        time.sleep(0.05) # minimalne opóźnienie, byś lepiej widział czy leci po kawałku
    print("\nKoniec.")
