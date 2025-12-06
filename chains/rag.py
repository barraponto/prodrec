from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import InMemorySaver

from settings import Settings


class RAGChain:
    def __init__(self, settings: Settings, vectorstore: VectorStore):
        self.settings: Settings = settings
        self.vectorstore: VectorStore = vectorstore
        self.agent = create_agent(
            model=self.settings.rag_model,
            temperature=0.5,
            checkpointer=InMemorySaver(),
            tools=[
                create_retriever_tool(
                    self.vectorstore.as_retriever({"k": 3}),
                    name="rag_tool",
                    description="A tool to retrieve product recommendations with reviews",
                )
            ],
            middleware=[
                SummarizationMiddleware(
                    llm=self.settings.rag_model,
                    trigger={"messages": 3},
                    keep={"messages": 1},
                )
            ],
        )
