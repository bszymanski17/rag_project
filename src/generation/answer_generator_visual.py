import base64
from typing import Iterator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler

from src.utils.load_settings import load_yaml, load_env
from src.utils.logger_config import create_logger

config = load_yaml("config/main.yaml")
prompts = load_yaml("prompts/main.yaml")
gcp_config, _ = load_env()
logger = create_logger("Multimodal Generation")

langfuse_handler = CallbackHandler()

def generate_multimodal_streamed_response(context: str, question: str, model: str) -> Iterator[str]:
    """Streams the generated response from Gemini using visual context (images) encoded in base64.

    Args:
        context: Newline-separated paths to the retrieved document images.
        question: The financial question asked by the user.
        model: The deployment name of the main Gemini LLM.

    Returns:
        A stream generator yielding real-time text chunks.
    """
    logger.info("Initializing multimodal Gemini model for visual RAG...")

    llm = ChatGoogleGenerativeAI(
        model=model,                    
        project=gcp_config.project_id,
        location="global",
        streaming=True,               
        vertexai=True                   
    )

    image_paths = context.split("\n")
    
    system_text = prompts['main_llm']['system_prompt']
    user_text = prompts["main_llm"]["user_prompt"].format(
        context="[Document pages are provided below as images]", 
        question=question
    )
    
    content = [{"type": "text", "text": user_text}]
    
    for path in image_paths:
        if path.strip():
            try:
                with open(path, "rb") as img_file:
                    b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                })
            except FileNotFoundError:
                logger.error(f"Context image not found at path: {path}")
                continue

    messages = [
        ("system", system_text),
        HumanMessage(content=content)
    ]
    
    rag_chain = (llm | StrOutputParser())
    logger.info("Multimodal visual RAG chain constructed successfully. Starting stream...")
    
    return rag_chain.stream(messages, config={"callbacks": [langfuse_handler]})