import os
import base64
import pandas as pd
import vertexai
from ragas import evaluate
from ragas.run_config import RunConfig

# Stabilne importy wrapperów i silnika Ragasa
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Importy metryk bezpośrednio ze stabilnego modułu
from ragas.metrics import (
    Faithfulness, 
    AnswerCorrectness, 
    ContextRecall,
    MultiModalFaithfulness, 
    MultiModalRelevance
)

from src.utils.load_settings import load_yaml, load_env
from src.utils.logger_config import create_logger
from src.evaluation.eval_utils import stream_to_string, save_final_report, prepare_ragas_dataset
from src.orchestrator import main

config = load_yaml("config/main.yaml")
config_eval = load_yaml("config/evaluation.yaml")
gcp_config, _ = load_env()
logger = create_logger("Evaluation")


def run_evaluation_pipeline(output_path: str, metrics: list = None):
    """Execute the complete RAGAS evaluation pipeline."""
    
    # 1. INICJALIZACJA GOOGLE VERTEX AI
    vertexai.init(project=gcp_config.project_id, location="us-central1")

    # 2. EMBEDDINGI DLA SĘDZIEGO
    embeddings_judge = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=config["models"]["embedding_model"],
        project=gcp_config.project_id,
        vertexai=True,
        timeout=90
    ))

    # 3. BEZPIECZNE OPAKOWANIE SĘDZIEGO PRZEZ LLM_FACTORY
    # Tworzymy stabilny model LangChain...
    langchain_llm = ChatGoogleGenerativeAI(
        model=config_eval["models"]["judge_model"],
        project=gcp_config.project_id,
        vertexai=True,
        timeout=90,
        max_retries=2
    )
    # ...i rejestrujemy go jako natywny sędzia Ragas przez llm_factory, podając go jako klient!
    model_name = config_eval["models"]["judge_model"]
    llm_judge = llm_factory(model=model_name, client=langchain_llm)

    # 4. USTAWIENIE METRYK
    if metrics is None:
        if config["vector_db"]["chunk_approach"] in ["visual_multimodal", "visual_database", "strict_multimodal"]:
            logger.info("Visual approach detected. Using MultiModal Ragas metrics.")
            metrics = [
                MultiModalFaithfulness(llm=llm_judge), 
                AnswerCorrectness(llm=llm_judge, embeddings=embeddings_judge), 
                MultiModalRelevance(llm=llm_judge)
            ]
        else:
            logger.info("Text approach detected. Using standard Ragas metrics.")
            metrics = [
                Faithfulness(llm=llm_judge), 
                AnswerCorrectness(llm=llm_judge, embeddings=embeddings_judge), 
                ContextRecall(llm=llm_judge)
            ]

    logger.info(f"Loading evaluation golden dataset.")
    try:
        eval_df = pd.read_csv(config_eval["path"]["dataset_path"])
    except FileNotFoundError:
        raise RuntimeError("Error: Evaluation dataset missing")

    logger.info(f"Loaded {len(eval_df)} test cases for evaluation.")

    # =========================================================================
    # MECHANIZM CACHE: Zabezpieczenie przed ponownym generowaniem odpowiedzi
    # =========================================================================
    temp_cache_file = "temp_generated_answers.csv"
    
    if os.path.exists(temp_cache_file):
        logger.info(f"ZNALEZIONO TEMPORALNY CACHE! Pomijam generowanie i ładuję gotowe odpowiedzi z '{temp_cache_file}'...")
        cache_df = pd.read_csv(temp_cache_file)
        system_answers = cache_df["system_answers"].tolist()
        system_contexts = [eval(ctx) for ctx in cache_df["system_contexts"].tolist()] 
    else:
        logger.info(f"Brak cache'u. Uruchamiam proces generacji odpowiedzi dla {len(eval_df)} pytań...")
        system_contexts = []
        system_answers = []

        for idx, row in eval_df.iterrows():
            logger.info(f"[{idx+1}/{len(eval_df)}] Processing question...")

            system_answer_stream, raw_chunks = main(
                row["Question"], evaluation_mode=True)

            full_answer_text = stream_to_string(system_answer_stream)

            if config["vector_db"]["chunk_approach"] in ["visual_multimodal", "visual_database", "strict_multimodal"]:
                loaded_contexts = []
                for chunk in raw_chunks:
                    if isinstance(chunk, str) and chunk.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(chunk):
                        try:
                            with open(chunk, "rb") as img_file:
                                b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                                mime_type = "image/png" if chunk.lower().endswith('.png') else "image/jpeg"
                                data_uri = f"data:{mime_type};base64,{b64_data}"
                                loaded_contexts.append(data_uri)
                        except Exception as e:
                            logger.warning(f"Failed to load image at {chunk}: {e}")
                            loaded_contexts.append(chunk) 
                    else:
                        loaded_contexts.append(chunk)
                system_contexts.append(loaded_contexts)
            else:
                system_contexts.append(raw_chunks)
                
            system_answers.append(full_answer_text.strip())

        # ZAPISUJEMY CACHE NA DYSK PRZED URUCHOMIENIEM RAGASA
        logger.info(f"Generacja zakończona sukcesem. Zapisuję cache do '{temp_cache_file}'...")
        pd.DataFrame({
            "system_answers": system_answers,
            "system_contexts": [str(ctx) for ctx in system_contexts]
        }).to_csv(temp_cache_file, index=False)
    # =========================================================================

    logger.info("Preparing data for Ragas...")
    ragas_dataset, report_df = prepare_ragas_dataset(
        eval_df, system_answers, system_contexts)

    logger.info("Launching Ragas evaluation framework.")

    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=180, max_workers=1)
    )

    logger.info("Ragas evaluation completed successfully.")

    logger.info(f"Exporting evaluation scores to final report: '{output_path}'")
    report_df = save_final_report(
        results, report_df, output_path=output_path, metrics=metrics)

    # Czyszczenie cache po udanej ewaluacji
    if os.path.exists(temp_cache_file):
        os.remove(temp_cache_file)

    logger.info("Evaluation pipeline execution finished successfully.")
    return report_df


if __name__ == "__main__":
    run_evaluation_pipeline(config_eval["path"]["output_path"])