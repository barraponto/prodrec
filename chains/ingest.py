import pandas as pd
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from settings import Settings


class IngestChain:
    required_columns: set[str] = {"product_title", "review"}

    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=self.settings.embeddings_model,
            huggingfacehub_api_token=self.settings.huggingface_api_key,
        )
        self.vectorstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name=self.settings.vectorstore_collection,
            api_endpoint=self.settings.astradb_api_endpoint,
            token=self.settings.astradb_api_key,
        )

    def _ensure_data(self, data: pd.DataFrame | None = None):
        if data is None:
            data = pd.read_csv(self.settings.data_path, on_bad_lines="skip")
        return data

    def _normalize_data(self, data: pd.DataFrame):
        data.columns = data.columns.str.lower()
        if missing_columns := self.required_columns - set(data.columns):
            raise ValueError(f"Missing required columns: {missing_columns!r}")
        return data.dropna()[list(self.required_columns)]

    def _get_document(self, row: pd.Series) -> Document:
        return Document(
            page_content=str(row["review"]),
            metadata={"product_name": row["product_title"]},
        )

    def ingest(self, data: pd.DataFrame | None = None):
        data = self._ensure_data(data)
        data = self._normalize_data(data)

        documents = [self._get_document(row) for _, row in data.iterrows()]

        self.vectorstore.add_documents(documents)
