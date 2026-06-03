import pandas as pd
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.metrics import Faithfulness, AnswerCorrectness, ContextRecall

from src.utils.load_settings import load_yaml, load_env
from src.utils.logger_config import create_logger
from src.evaluation.eval_utils import stream_to_string, save_final_report, prepare_ragas_dataset
from src.orchestrator import main

config = load_yaml("config/main.yaml")
config_eval = load_yaml("config/evaluation.yaml")
gcp_config, _ = load_env()
logger = create_logger("Evaluation")


def run_evaluation_pipeline(output_path: str, metrics: list = [Faithfulness(), AnswerCorrectness(), ContextRecall()]):
    """Execute the complete RAGAS evaluation pipeline.

    Args:
        output_path: Filepath where the final CSV report will be saved.
        metrics: A list of Ragas metric objects to evaluate.

    Returns:
        The final report DataFrame containing questions, ground truths, 
        generated answers, retrieved contexts, and calculated metric scores.
    """

    logger.info(f"Loading evaluation golden dataset.")
    try:
        eval_df = pd.read_csv(config_eval["path"]["dataset_path"])
    except FileNotFoundError as e:
        logger.error(f"Evalutaion dataset not found at {config_eval["path"]["dataset_path"]}.")
        raise RuntimeError(f"Error: Evaluation dataset missing at {config_eval["path"]["dataset_path"]}")

    logger.info(f"Loaded {len(eval_df)} test cases for evaluation.")

    system_contexts = []
    system_answers = []

    for idx, row in eval_df.iterrows():

        logger.info(f"[{idx+1}/{len(eval_df)}] Processing question...")

        system_answer_stream, raw_chunks = main(
            row["Question"], evaluation_mode=True)

        full_answer_text = stream_to_string(system_answer_stream)

        system_contexts.append(raw_chunks)
        system_answers.append(full_answer_text.strip())

    logger.info("Answer generating completed. Preparing data for Ragas...")
    ragas_dataset, report_df = prepare_ragas_dataset(
        eval_df, system_answers, system_contexts)

    embeddings_judge = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=config["models"]["embedding_model"],
        project=gcp_config.project_id,
        vertexai=True,
        timeout=90
    ))

    llm_judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config_eval["models"]["judge_model"],
        project=gcp_config.project_id,
        vertexai=True,
        timeout=90,
        max_retries=2
    ))

    logger.info("Launching Ragas evaluation framework.")

    results = evaluate(dataset=ragas_dataset,
                       metrics=metrics,
                       llm=llm_judge,
                       embeddings=embeddings_judge,
                       run_config=RunConfig(timeout=180, max_workers=1)
                       )

    logger.info("Ragas evaluation completed successfully.")

    logger.info(
        f"Exporting evaluation scores to final report: '{output_path}'")
    report_df = save_final_report(
        results, report_df, output_path=output_path, metrics=metrics)

    logger.info("Evaluation pipeline execution finished successfully.")

    return report_df


if __name__ == "__main__":
    run_evaluation_pipeline(config_eval["path"]["output_path"])
