from pydantic import BaseModel

class GCPConfig(BaseModel):
    """
    Configuration model for the Google Cloud Platform (GCP) environment.
    """
    project_id: str

class LFConfig(BaseModel):
    """
    Configuration model for the Google Cloud Platform (GCP) environment.
    """
    langfuse_secret_key: str
    langfuse_public_key: str 
    langfuse_base_url: str