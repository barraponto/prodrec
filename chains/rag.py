from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import InMemorySaver

from settings import Settings


class RAGChain:
    def __init__(self, settings: Settings, vectorstore: VectorStore):
        self.settings: Settings = settings
        self.vectorstore: VectorStore = vectorstore
        self.model = init_chat_model(
            self.settings.rag_model,
            api_key=self.settings.groq_api_key,
            temperature=0.5,
        )
        rag_tool = create_retriever_tool(
            self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            name="retrieve_product_recommendations",
            response_format="content_and_artifact",
            description="""
            A tool to retrieve product recommendations based on the user's query.
            Will return info about the top 3 products matching description.
            """,
        )
        self.agent = create_agent(
            model=self.model,
            checkpointer=InMemorySaver(),
            tools=[rag_tool],
            middleware=[SummarizationMiddleware(self.model, trigger=("messages", 8))],
            system_prompt="""
            You are a helpful assistant that can help with product recommendations.
            You will be given a user's query and you will need to retrieve the products matching the query.
            Use the retrieve_product_recommendations tool to get product recommendations.
            DO NOT use any other tools.
            """,
        )
