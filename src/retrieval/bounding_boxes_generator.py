import chromadb
from PIL import Image, ImageDraw
from src.utils.load_settings import create_logger

logger = create_logger("Bounding Boxes")


def draw_bounding_boxes_from_metadata(image_path: str,winning_patch_indices: list,db_path: str,page_num: int,outline_color: tuple = (255, 0, 0, 255),fill_color: tuple = (255, 255, 0, 60),line_width: int = 5,):
    """
    Draws bounding boxes around patches that contributed most to the answer (MaxSim winners).

    Args:
        image_path: Path to the source page image.
        winning_patch_indices: List of flat patch indices to highlight.
        db_path: Path to ChromaDB persistent database.
        page_num: Page number used to fetch grid dimensions from metadata.
        outline_color: RGBA color for box outline.
        fill_color: RGBA color for box fill.
        line_width: Border thickness in pixels.

    Returns:
        PIL Image with bounding boxes drawn over relevant patches.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("strict_patches")

    res = collection.get(where={"page_num": page_num}, include=["metadatas"], limit=1)
    if not res["metadatas"]:
        logger.error(f"No metadata found for page {page_num}")
        return Image.open(image_path).convert("RGB")

    n_x = res["metadatas"][0]["n_patches_x"]
    n_y = res["metadatas"][0]["n_patches_y"]

    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    patch_w = W / n_x
    patch_h = H / n_y

    logger.info(f"Drawing {len(set(winning_patch_indices))} bounding boxes on {image_path} (grid: {n_x}x{n_y})")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for patch_idx in set(winning_patch_indices):
        row = patch_idx // n_x
        col = patch_idx % n_x
        x0 = col * patch_w
        y0 = row * patch_h
        x1 = x0 + patch_w
        y1 = y0 + patch_h
        overlay_draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=line_width)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")