import os
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.indexing.element_describer import generate_description
from src.utils.load_settings import create_logger, load_yaml

logger = create_logger("Chunking document")
config = load_yaml("config/main.yaml")


def chunk_pdf(pdf_path: str, blacklist: list) -> list:
    """Load a PDF, enrich tables/images in-line, and chunk the continuous document stream."""
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
    except Exception as e:
        logger.error(f"Error during document loading: {str(e)}")
        raise e

    filtered_elements = [doc for doc in raw_elements if doc.metadata.get(
        "category") not in blacklist]

    logger.info("Stitching and enriching the document into a single stream...")
    
    full_enriched_text = ""
    base_metadata = {}
    
    if filtered_elements:
        base_metadata = {k: v for k, v in filtered_elements[0].metadata.items() if k != "category"}

    for doc in filtered_elements:
        category = doc.metadata.get("category", "Text")
        page_num = doc.metadata.get("page_number", "Unknown")
        
        if category == "Table":
            desc = generate_description(element_type="Table", element_content=doc.page_content)
            full_enriched_text += (
                f"\n\n[START OF TABLE - LOCATION: Page {page_num}]\n"
                f"{desc}\n"
                f"[END OF TABLE]\n\n"
            )
            
        elif category == "Image":
            extracted_image_path = doc.metadata.get("image_path")
            desc = generate_description(element_type="Image", element_content=extracted_image_path)
            full_enriched_text += (
                f"\n\n[START OF IMAGE/CHART - LOCATION: Page {page_num}]\n"
                f"{desc}\n"
                f"[END OF IMAGE/CHART]\n\n"
            )
            
        else:
            full_enriched_text += f"\n{doc.page_content}"

    logger.info("Executing final text splitting on the enriched stream...")
    text_splitter = RecursiveCharacterTextSplitter(
        #separators=["\n\n", "\n", " ", ""],
        separators = ["\n\n\n","\n\n",". ",".\n","? ","! "," ",""],
        chunk_size=config["vector_db"]["chunk_size"],
        chunk_overlap=config["vector_db"]["chunk_overlap"]
    )

    unified_document = Document(page_content=full_enriched_text, metadata=base_metadata)
    final_documents = text_splitter.split_documents([unified_document])

    final_documents = filter_complex_metadata(final_documents)
    logger.info(f"Successfully generated {len(final_documents)} contextualized chunks.")

    return final_documents