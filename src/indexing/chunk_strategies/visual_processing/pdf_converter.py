import os
from pdf2image import convert_from_path
from PIL import Image

from src.utils.load_settings import create_logger, load_yaml

logger = create_logger("Visual Chunking")
config = load_yaml("config/main.yaml")

def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list[dict]:
    """
    Load a PDF and convert each page into a PIL Image.
    """
    logger.info(f"Loading document and rendering pages to images: {pdf_path}")
    
    try:
        pages_images = convert_from_path(pdf_path, dpi=250)
        logger.info(f"Successfully rendered {len(pages_images)} pages.")
    except Exception as e:
        logger.error(f"Error during document rendering: {str(e)}")
        raise e

    visual_chunks = []
    
    for i, page_img in enumerate(pages_images):
        page_num = i + 1
        
        image_path = os.path.join(output_dir, f"page_{page_num}.jpg")
        page_img.save(image_path, "JPEG")

        
        visual_chunks.append({
            "page_number": page_num,
            "image": page_img,        
            "source": pdf_path
        })

    logger.info(f"Successfully generated {len(visual_chunks)} visual page chunks.")
    
    return visual_chunks