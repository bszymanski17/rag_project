import torch
import chromadb
from typing import List, Tuple, Dict
from colpali_engine.models import ColQwen2, ColQwen2Processor
from src.utils.load_settings import create_logger

logger = create_logger("Strict Retrieval")


def retrieve_maxsim(query: str,db_path: str,model,processor,device: str,top_k: int):
    """Retrieves and ranks document pages using MaxSim Late Interaction scoring.

    Args:
        query: The user query string.
        db_path: Path to the ChromaDB persistent database.
        model: Loaded ColQwen2 model for query embedding generation.
        processor: Loaded ColQwen2Processor for query tokenization.
        device: Device to run inference on ('mps' or 'cpu').
        top_k: Number of top ranked pages to return.

    Returns:
        A tuple containing:
            - List of image paths for the top-k ranked pages.
            - Dict mapping page number to list of winning patch indices
              (patches that contributed most to the MaxSim score).
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("strict_patches")

    batch_queries = processor.process_queries([query]).to(device)
    with torch.no_grad():
        query_emb = model(**batch_queries)[0].cpu() 

    candidate_pages = set()
    for q_token in query_emb:
        res = collection.query(query_embeddings=[q_token.tolist()], n_results=10)
        for meta in res["metadatas"][0]:
            candidate_pages.add(meta["page_num"])

    if not candidate_pages:
        logger.warning("No candidate pages found.")
        return [], {}

    logger.info(f"Candidate pages: {sorted(candidate_pages)}")

    page_scores = {}
    best_patches_per_page = {}

    for page in candidate_pages:
        res = collection.get(where={"page_num": page}, include=["embeddings", "metadatas"])
        if res["embeddings"] is None or len(res["embeddings"]) == 0:
            continue

        page_emb = torch.tensor(res["embeddings"]).float() 
        patch_indices = [m["patch_index"] for m in res["metadatas"]]

        sims = torch.einsum("pd,qd->pq", page_emb, query_emb)  
        max_sims, max_patch_idx = sims.max(dim=0)           

        score = max_sims.sum().item()
        page_scores[page] = score
        best_patches_per_page[page] = [patch_indices[idx] for idx in max_patch_idx.tolist()]

    ranked_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    logger.info(f"Ranked pages: {[(p, round(s, 3)) for p, s in ranked_pages]}")

    final_paths = []
    final_patches = {}

    for page_num, _ in ranked_pages:
        res = collection.get(where={"page_num": page_num}, include=["metadatas"])
        img_path = res["metadatas"][0]["image_path"]
        final_paths.append(img_path)
        final_patches[page_num] = best_patches_per_page[page_num]

    return final_paths, final_patches


def rerank_patches_by_answer(answer: str, raw_chunks: list, db_path: str, model, processor, device: str) -> dict:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("strict_patches")

    batch_queries = processor.process_queries([answer]).to(device)
    with torch.no_grad():
        answer_emb = model(**batch_queries)[0].cpu()

    best_patches = {}
    for path in raw_chunks:
        page_num = int(path.split("page_")[-1].replace(".jpg", ""))
        res = collection.get(where={"page_num": page_num}, include=["embeddings", "metadatas"])
        if res["embeddings"] is None or len(res["embeddings"]) == 0:
            continue

        page_emb = torch.tensor(res["embeddings"]).float()
        patch_indices = [m["patch_index"] for m in res["metadatas"]]

        sims = torch.einsum("pd,qd->pq", page_emb, answer_emb)
        max_patch_idx = sims.max(dim=0).indices
        best_patches[page_num] = [patch_indices[idx] for idx in max_patch_idx.tolist()]

    return best_patches