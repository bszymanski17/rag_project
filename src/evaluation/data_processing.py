from datasets import Dataset

def stream_to_string(stream):
    full_answer_text = ""
    for chunk in stream:
        if isinstance(chunk, str):
            full_answer_text += chunk
        else:
            full_answer_text += getattr(chunk, "content", "")
    return full_answer_text

def prepare_ragas_dataset(eval_df, system_answers, system_contexts):
    report_df = eval_df.copy()
    report_df["response"] = system_answers
    report_df["retrieved_contexts"] = system_contexts
    
    report_df = report_df.rename(columns={
        "Question": "user_input",
        "Ground_Truth_Answer": "reference"
    })
    
    return Dataset.from_pandas(report_df), report_df

def save_final_report(results_df, report_df, output_path,metrics):
    results_df = results_df.to_pandas()
    list_of_metrics = [metric.name for metric in metrics]    
    
    for metric in list_of_metrics:
        if metric in results_df.columns:
            report_df[metric] = results_df[metric].values
    
    report_df.to_csv(output_path, index=False)
    
    return report_df