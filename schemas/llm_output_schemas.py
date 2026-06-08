from typing import Optional
from pydantic import BaseModel, Field

class QueryFilterSchema(BaseModel):
    clean_query: str = Field(
        description="The core search query stripped of any page range or filtering instructions."
    )
    start_page: Optional[int] = Field(
        None, description="The starting page number if explicitly requested by the user (e.g., from 'pages 10-20' -> 10)."
    )
    end_page: Optional[int] = Field(
        None, description="The ending page number if explicitly requested by the user (e.g., from 'pages 10-20' -> 20)."
    )