from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.load_settings import create_logger

logger = create_logger("Chunking document")


def chunk_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int) -> list:
    """Loads a PDF file and splits its text into smaller chunks.

    Args:
        pdf_path: Path to the PDF file on disk.
        chunk_size: Maximum number of characters in a single text chunk.
        chunk_overlap: Number of characters that adjacent chunks 

    Returns:
        A list of LangChain `Document` objects, each containing 
            a text chunk and associated metadata.
    """

    logger.info("Loading document...")

    try:
        loader = PyMuPDFLoader(pdf_path)
        raw_document = loader.load()
        logger.info("Document loaded successfully.")
    except Exception as e:
        logger.error(f"Error during document loading: {str(e)}")
        raise RuntimeError(f"Error during document loading: {str(e)}")

    logger.info("Splitting document...")

    text_spliter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_doc = text_spliter.split_documents(raw_document)

    logger.info("Document splited.")

    return chunked_doc
