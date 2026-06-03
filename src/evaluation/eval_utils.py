from datasets import Dataset

def stream_to_string(stream) -> str:
    """Consumes a token stream from the LLM and aggregates it into a single string.

    Args:
        stream: An iterator text chunks from the LLM.

    Returns:
        str: The fully concatenated text response.
    """

    full_answer_text = ""
    for chunk in stream:
        if isinstance(chunk, str):
            full_answer_text += chunk
        else:
            full_answer_text += getattr(chunk, "content", "")
    return full_answer_text

def prepare_ragas_dataset(eval_df, system_answers, system_contexts):
    """Prepares and formats a pandas DataFrame into a Hugging Face Dataset compatible with the Ragas evaluation schema.

    Args:
        eval_df: The baseline evaluation DataFrame containing input questions.
        system_answers: List of generated answers from the RAG pipeline.
        system_contexts: List of retrieved source chunks for each question.

    Returns:
        Tuple[Dataset, pd.DataFrame]: A tuple containing:
            - Dataset: Hugging Face Dataset formatted with Ragas-required column names.
            - pd.DataFrame: A copy of the updated report DataFrame.
    """

    report_df = eval_df.copy()
    report_df["response"] = system_answers
    report_df["retrieved_contexts"] = system_contexts
    
    report_df = report_df.rename(columns={
        "Question": "user_input",
        "Ground_Truth_Answer": "reference"
    })
    
    return Dataset.from_pandas(report_df), report_df

def save_final_report(results_df, report_df, output_path: str, metrics: list):
    """Merges computed Ragas evaluation metrics back into the main report DataFrame and saves the combined result to a local CSV file.

    Args:
        results_df: The Ragas EvaluationResult object containing computed scores.
        report_df: The working DataFrame with questions, answers, and contexts.
        output_path: Local filesystem path where the final CSV report will be written.
        metrics: List of Ragas metric objects used during the evaluation run.

    Returns:
        pd.DataFrame: The final updated DataFrame containing all data and evaluation scores.
    """
    results_df = results_df.to_pandas()
    list_of_metrics = [metric.name for metric in metrics]    
    
    for metric in list_of_metrics:
        if metric in results_df.columns:
            report_df[metric] = results_df[metric].values
    
    report_df.to_csv(output_path, index=False)
    
    return report_df
