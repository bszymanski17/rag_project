import os
import torch
import chromadb
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2, ColQwen2Processor

from src.utils.load_settings import load_yaml
from src.utils.logger_config import create_logger

logger = create_logger("Strict Indexer")
config = load_yaml("config/main.yaml")


def build_strict_vector_db(images_dir: str, db_path: str):
    """Indexes image patches into database with explicit location metadata.

    Args:
        images_dir: Path to the directory containing page images.
        db_path: Path where the database will be created.
    """
    logger.info("Initializing database...")
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection("strict_patches")
    except Exception:
        pass

    collection = client.create_collection("strict_patches", metadata={"hnsw:space": config["vector_db"]["similarity_metric"]})

    logger.info("Loading ColQwen2 model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = ColQwen2Processor.from_pretrained(config["models"]["visual_model"])
    model = ColQwen2.from_pretrained(
        config["models"]["visual_model"],
        torch_dtype=torch.float32
    ).to(device).eval()

    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for img_file in tqdm(image_files, desc="Indexing pages"):
        page_num = int(img_file.split("_")[1].split(".")[0])
        img_path = os.path.join(images_dir, img_file)

        image = Image.open(img_path).convert("RGB")
        batch_images = processor.process_images([image]).to(device)

        with torch.no_grad():
            image_embeddings = model(**batch_images)

        image_mask = processor.get_image_mask(batch_images)
        patches = image_embeddings[0][image_mask[0]].cpu().numpy()  

        spatial_merge_size = getattr(processor.image_processor, "merge_size", 2)
        n_patches = processor.get_n_patches(
            image_size=(image.size[0], image.size[1]),
            spatial_merge_size=spatial_merge_size,
        )
        n_x, n_y = n_patches

        ids = []
        embeddings = []
        metadatas = []

        for patch_idx, patch_vec in enumerate(patches):
            row = patch_idx // n_x
            col = patch_idx % n_x
            ids.append(f"page_{page_num}_patch_{patch_idx}")
            embeddings.append(patch_vec.tolist())
            metadatas.append({
                "page_num": page_num,
                "patch_index": patch_idx,
                "patch_row": row,
                "patch_col": col,
                "n_patches_x": n_x,
                "n_patches_y": n_y,
                "image_path": img_path,
            })

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

    logger.info(f"Indexing complete. Saved to {db_path}")