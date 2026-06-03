from typing import Dict, Any, Tuple
import yaml
import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import ValidationError

from src.utils.logger_config import create_logger
from src.schemas import GCPConfig, LFConfig


logger = create_logger("Loading yaml config")

@st.cache_data
def load_yaml(path: str) -> Dict[str, Any]:
    """Load applictaion settings from a YAML file.

    Args:
        config_paths: Path to the YAML configuration file.

    Returns:
        A dictionary containging the configuration parameters. Return an empty dict if loading fails.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"YAML config loaded successfuly from {path}")
        return config
    except FileNotFoundError:
        logger.error(f"Error: File not found at: {path}")
    except Exception as e:
        logger.error(f"Error during YAML confih loading: {str(e)}")


@st.cache_data
def load_env() -> Tuple[GCPConfig, LFConfig]:
    """Loads and validates environment variables for DB and GCP.

    Returns:
        A tuple containing tthree dictionaries:
            - GCP_CONFIG: Google Cloud settings (project_id).
            - LANGFUSE_CONFIG: Lanfuse settings.
    """

    logger.info("Loading environment variables...")
    load_dotenv()

    try:

        gcp_conf = GCPConfig(
            project_id= os.getenv("GCP_PROJECT_ID")
        )

        langfuse_conf = LFConfig(
            langfuse_secret_key= os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_public_key= os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_base_url =os.getenv("LANGFUSE_BASE_URL")
        )
 
    except ValidationError as v:
        logger.error(f"Error during validation enviroment variables: {str(v)}")
        raise RuntimeError(f"Error during validation enviroment variables. {str(v)}")
    except Exception as e:
        logger.error(f"Error during loading enviroment variables: {str(e)}")
        raise RuntimeError("Error during loading enviroment variables.")
    
    return gcp_conf, langfuse_conf




