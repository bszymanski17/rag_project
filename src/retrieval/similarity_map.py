import torch
from PIL import Image, ImageDraw
from colpali_engine.models import ColQwen2, ColQwen2Processor
import streamlit as st
import logging

from src.utils.logger_config import create_logger

logger = create_logger("Similarity map")


@st.cache_resource
def load_colqwen2_model(model_name: str = "vidore/colqwen2-v1.0", device: str = "mps"):
    """Loads and caches the ColQwen2 model and processor.

    Args:
        model_name (str): Hugging Face model identifier.
        device (str): Device to load the model onto (e.g., "mps", "cuda").

    Returns:
        A tuple containing the loaded model and its processor.
    """
    logger.info(f"Loading ColQwen2 model on {device}...")
    processor = ColQwen2Processor.from_pretrained(model_name)
    model = ColQwen2.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    logger.info("ColQwen2 loaded successfully.")
    return model, processor


def get_highlighted_image(image_path: str, query: str, model, processor, device: str = "mps") -> Image.Image:
    """Generates a similarity heatmap overlay on an image based on a text query.

    Args:
        image_path (str): Path to the source image file.
        query (str): Text query to visually locate on the image.
        model: The loaded ColQwen2 model instance.
        processor: The associated model processor.
        device (str): Device to run the inference on.

    Returns:
        The original RGB image overlaid with the similarity heatmap.
    """
    logger.info(f"Generating similarity map for: {image_path}")
    image = Image.open(image_path).convert("RGB")

    batch_images = processor.process_images([image]).to(device)
    batch_queries = processor.process_queries([query]).to(device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    spatial_merge_size = getattr(processor.image_processor, "merge_size", 2)
    n_patches = processor.get_n_patches(
        image_size=(image.size[0], image.size[1]),
        spatial_merge_size=spatial_merge_size,
    )
    n_x, n_y = n_patches
    logger.info(f"n_patches: {n_patches}")

    image_mask = processor.get_image_mask(batch_images)
    logger.info(f"image_mask True count: {image_mask[0].sum().item()}, expected: {n_x * n_y}")

    masked_image_emb = image_embeddings[0][image_mask[0]]  
    query_emb = query_embeddings[0]                       

    sim = torch.einsum("pd,qd->qp", masked_image_emb, query_emb) 
    sim_map_flat = sim.max(dim=0).values.cpu().float().numpy()  

    logger.info(f"sim_map_flat shape: {sim_map_flat.shape}, min={sim_map_flat.min():.4f}, max={sim_map_flat.max():.4f}, mean={sim_map_flat.mean():.4f}")

    sim_map = sim_map_flat[:n_x * n_y].reshape(n_y, n_x)  

    s_min, s_max = sim_map.min(), sim_map.max()
    if s_max > s_min:
        sim_map = (sim_map - s_min) / (s_max - s_min)

    logger.debug(f"After normalization — min={sim_map.min():.4f}, max={sim_map.max():.4f}, mean={sim_map.mean():.4f}")
    logger.debug(f"Patches above 0.3: {(sim_map >= 0.3).sum()}, above 0.5: {(sim_map >= 0.5).sum()}, above 0.7: {(sim_map >= 0.7).sum()}")

    img_w, img_h = image.size
    n_h, n_w = sim_map.shape
    patch_h = img_h / n_h
    patch_w = img_w / n_w

    heatmap = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap)

    for row in range(n_h):
        for col in range(n_w):
            score = float(sim_map[row, col])
            x0, y0 = int(col * patch_w), int(row * patch_h)
            x1, y1 = int((col + 1) * patch_w), int((row + 1) * patch_h)
            if score >= 0.5:
                r = min(255, 180 + int(75 * score))
                g = min(255, int(120 * score))
                a = int(0.6 * 255 * score)
                draw.rectangle([x0, y0, x1, y1], fill=(r, g, 0, a))
            else:
                a = int(0.5 * 255 * (1 - score))
                draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, a))

    result = Image.alpha_composite(image.convert("RGBA"), heatmap)
    logger.info(f"Heatmap complete for {image_path}")
    return result.convert("RGB")
