import os
import uuid

from flask import Flask, render_template, request, session
from prometheus_client import Counter, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from chains.ingest import IngestChain
from chains.rag import RAGChain
from settings import Settings

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})

LLM_CALLS = Counter("llm_calls", "Number of LLM calls")

settings = Settings()
ingest = IngestChain(settings)
rag = RAGChain(settings, ingest.vectorstore)


@app.route("/", methods=["GET", "POST"])
def index():
    if session.get("thread_id") is None:
        session["thread_id"] = str(uuid.uuid4())

    if request.method == "POST":
        question = request.form["question"]
        LLM_CALLS.inc()
        response = rag.agent.invoke(
            {"messages": [question]},
            config={"configurable": {"thread_id": session.get("thread_id")}},
        )
        message = response["messages"][-1]
        return render_template("index.html", message=message.content)
    return render_template("index.html")
