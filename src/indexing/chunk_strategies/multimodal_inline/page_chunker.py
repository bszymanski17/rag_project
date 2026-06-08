import os
import re
from collections import defaultdict
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.indexing.element_describer import generate_description
from src.utils.load_settings import create_logger, load_yaml

logger = create_logger("Chunking document")
config = load_yaml("config/main.yaml")


def chunk_pdf(pdf_path: str, blacklist: list) -> list:
    """Load a PDF, enrich tables/images in-line per page, and chunk keeping strict page metadata."""
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

    filtered_elements = [doc for doc in raw_elements if doc.metadata.get("category") not in blacklist]

    logger.info("Grouping elements by page to preserve metadata...")
    
    # Groupping by page
    pages_dict = defaultdict(list)
    for doc in filtered_elements:
        page_num = doc.metadata.get("page_number", 1)
        pages_dict[page_num].append(doc)

    page_documents = []

    for page_num in sorted(pages_dict.keys()):
        page_text = ""
        elements = pages_dict[page_num]
        
        page_metadata = {k: v for k, v in elements[0].metadata.items() if k != "category"}
        page_metadata["page_number"] = int(page_num)  
        

        for doc in elements:
            category = doc.metadata.get("category", "Text")
            
            if category == "Table":
                desc = generate_description(element_type="Table", element_content=doc.page_content)
                page_text += (
                    f"\n\n[START OF TABLE - LOCATION: Page {page_num}]\n"
                    f"{desc}\n"
                    f"[END OF TABLE]\n\n"
                )
                
            elif category == "Image":
                extracted_image_path = doc.metadata.get("image_path")
                desc = generate_description(element_type="Image", element_content=extracted_image_path)
                page_text += (
                    f"\n\n[START OF IMAGE/CHART - LOCATION: Page {page_num}]\n"
                    f"{desc}\n"
                    f"[END OF IMAGE/CHART]\n\n"
                )
                
            else:
                cleaned_text = doc.page_content.replace("\n", " ")
                cleaned_text = re.sub(r' +', ' ', cleaned_text).strip()
                if cleaned_text:
                    page_text += f"\n\n{cleaned_text}"

        page_doc = Document(page_content=page_text, metadata=page_metadata)
        page_documents.append(page_doc)

    logger.info("Executing final text splitting on per-page streams...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", ". ", ".\n", "? ", "! ", " ", ""],
        chunk_size=config["vector_db"]["chunk_size"],
        chunk_overlap=config["vector_db"]["chunk_overlap"]
    )

    final_documents = text_splitter.split_documents(page_documents)
    final_documents = filter_complex_metadata(final_documents)
    
    logger.info(f"Successfully generated {len(final_documents)} chunks with valid page metadata.")
    return final_documents