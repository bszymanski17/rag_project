import pandas as pd
from src.utils.tools import load_yaml, load_env
from src.evaluation.data_processing import stream_to_string, save_final_report, prepare_ragas_dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from datasets import Dataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.orchestrator import main
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


config = load_yaml("config/config.yaml")
gcp_config, _ = load_env()

def run_evaluation_pipeline(output_path: str, rag_function, metrics: list = [faithfulness, answer_correctness, context_recall, context_precision]):
    eval_df = pd.read_csv(config["evaluation"]["dataset_path"])

    system_contexts = []
    system_answers = []

    for idx, row in eval_df.iterrows():

        system_answer_stream, raw_chunks = rag_function(row["Question"], evaluation_mode=True)
        
        full_answer_text = stream_to_string(system_answer_stream)

        system_contexts.append(raw_chunks)
        system_answers.append(full_answer_text.strip())

    ragas_dataset, report_df = prepare_ragas_dataset(eval_df, system_answers, system_contexts)

    embeddings_judge = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=config["models"]["embedding_model"],
            project=gcp_config.project_id,
            vertexai=True # <-- Wymusza bezpieczny backend Vertex przez HTTP REST
        )
    )
    
    llm_judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=config["models"]["judge_model"],
            project=gcp_config.project_id,
            vertexai=True # <-- Wymusza bezpieczny backend Vertex przez HTTP REST
        )
    )
    results = evaluate(dataset=ragas_dataset,
        metrics=metrics,
        llm=llm_judge,
        embeddings=embeddings_judge,
        run_config=RunConfig(timeout=180, max_workers=1)
    )

    report_df = save_final_report(results, report_df, output_path=output_path, metrics=metrics)

    return report_df

if __name__ == "__main__":
    run_evaluation_pipeline("evaluation/results/baseline_results.csv", main)
        
        