from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.tools import load_yaml, load_env
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from typing import Iterator
from langchain_google_vertexai import ChatVertexAI


prompts = load_yaml("prompts/prompts.yaml")
gcp_config, langfuse_config = load_env()

Langfuse(
    public_key=langfuse_config.langfuse_public_key,
    secret_key=langfuse_config.langfuse_secret_key,
    host=langfuse_config.langfuse_base_url
)

langfuse_handler = CallbackHandler()

def generate_data(context: str, question: str, model: str) -> Iterator[str]:
    """
    Streams the generated response from the LLM based on the provided context.

    Args:
        context: The background knowledge from the vector store.
        question: The specific financial question asked by the user.
        model: The exact deployment name of the Vertex AI main LLM.

    Returns:
        Iterator[str]: A stream generator yielding real-time text chunks 
                       of the generated response.
    """

    llm = ChatGoogleGenerativeAI(
    model=model,                    
    project=gcp_config.project_id,
    location="global",           
    streaming=True,               
    vertexai=True                   
)


    prompt = ChatPromptTemplate.from_messages([("system", prompts['main_llm']['system_prompt']), ("user", prompts["main_llm"]["user_prompt"])])

    rag_chain = (prompt | llm | StrOutputParser())

    return rag_chain.stream({"context": context, "question": question}, config={"callbacks": [langfuse_handler]})