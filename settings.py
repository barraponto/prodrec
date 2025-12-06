from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    data_path: Path = Field(default=Path("./data/flipkart_product_review.csv"))
    embeddings_model: str = Field(default="BAAI/bge-base-en-v1.5")
    rag_model: str = Field(default="groq:llama-3.1-8b-instant")
    vectorstore_collection: str = Field(default="productrec")

    astradb_api_endpoint: str = Field(default="")
    astradb_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")
