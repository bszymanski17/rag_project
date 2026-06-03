from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import base64
import os
from langfuse.langchain import CallbackHandler


from src.utils.load_settings import create_logger, load_yaml
from src.indexing.element_describer import generate_desc

logging.getLogger("transformers").setLevel(logging.ERROR)
logger = create_logger("Chunking document")
config = load_yaml("config/config.yaml")


def chunk_pdf(pdf_path: str, blacklist: list):
    """Load a PDF, filter out blacklisted categories, and chunk content into documents.

    Args:
        pdf_path: Filepath to the target PDF document.
        blacklist: List of element categories to exclude from processing (e.g. ['Image', 'Table']).

    Returns:
        A list of sanitized and split LangChain Document objects ready for vector database indexing.
    """
    logger.info("Loading document...")

    from langchain_community.document_loaders import UnstructuredPDFLoader
    image_dir = ("images/extracted_images" if "Image" not in blacklist else None)
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)

    try:
        loader_kwargs = {
            "file_path": pdf_path,
            "strategy": "hi_res",
            "mode": "elements",
            "pdf_infer_table_structure": ("Table" not in blacklist),
            "languages": ["eng"]
        }

        if image_dir:
            loader_kwargs["extract_image_block_types"] = ["Image"]
            loader_kwargs["extract_image_block_output_dir"] = image_dir

        loader = UnstructuredPDFLoader(**loader_kwargs)
        raw_elements = loader.load()
        
        logger.info("Document loaded successfully!")
        
        logger.info("Document loaded successfully!")
    except Exception as e:
        logger.error(f"Error during document loading: {str(e)}")
        raise e

    filtered_elements = [doc for doc in raw_elements if doc.metadata.get(
        "category") not in blacklist]
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=config["vector_db"]["chunk_size"],
        chunk_overlap=config["vector_db"]["chunk_overlap"])

    final_documents = []

    logger.info("Splitting document...")

    for doc in filtered_elements:
        category = doc.metadata.get("category")
        if category == "Table":
            desc = generate_desc(element_type="Table", element_content=doc.page_content)
            table_document = Document(page_content=desc, metadata=doc.metadata)
            final_documents.append(table_document)
        elif category == "Image":
            extracted_image_path = doc.metadata.get("image_path")
            desc = generate_desc(element_type="Image", element_content=extracted_image_path)
            image_document = Document(page_content=desc, metadata=doc.metadata)
            final_documents.append(image_document)
        else:
            chunks = text_splitter.split_documents([doc])
            final_documents.extend(chunks)

    final_documents = filter_complex_metadata(final_documents)

    logger.info("Document splitted successfully.")
    

    return final_documents
