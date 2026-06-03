import base64

def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file into a Base64 string for multimodal LLM processing.

    Args:
        image_path: Path to the image file on the disk.

    Returns:
        UTF-8 decoded Base64 string representing the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")