import json
import pickle
import typing
import datetime
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse

app = FastAPI(
    description="Personnal knowledge graph.",
    title="FactGPT",
    version="0.0.1",
)

origins = [
    "https://ohmygaugh-crypto.fly.dev",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Knowledge:
    """This class is a wrapper around the pipeline."""

    def __init__(self) -> None:
        self.pipeline = None

    def start(self):
        """Load the pipeline."""
        with open("database/pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        print("Pipeline loaded successfully")
        return self

    def search(
        self,
        q: str,
        tags: str,
    ) -> typing.Dict:
        """Returns the documents."""
        return self.pipeline.search(q=q, tags=tags)

    def plot(
        self,
        q: str,
        k_tags: int,
        k_yens: int = 1,
        k_walk: int = 3,
    ) -> typing.Dict:
        """Returns the graph."""
        nodes, links = self.pipeline.plot(
            q=q,
            k_tags=k_tags,
            k_yens=k_yens,
            k_walk=k_walk,
        )
        return {"nodes": nodes, "links": links}


knowledge = Knowledge()


async def async_chat(query: str, content: str):
    """Re-rank the documents using ChatGPT."""
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""
                You are knowledgable personnal assitant, based on the input query {query}, you will recommend the best resources from the set of retrieved documents. You will write for the top 3 recommended resources their title, a comprehensive and short description and their url. Rely on the set of documents provided and on your knowledge.
                """,
            },
            {"role": "user", "content": content},
        ],
        temperature=0.3,
        max_tokens=300,
        stream=True,
        top_p=1,
    )

    answer = ""
    async for token in response:
        token = token["choices"][0]["delta"]
        if "content" in token:
            answer += token["content"]

            while "\n\n" in answer:
                answer = answer.replace("\n\n", "\n")

            for replacement in [
                ("1. ", "\n1. "),
                ("2. ", "\n2. "),
                ("3. ", "\n3. "),
                ("Title:", ""),
                ("Summary:", ""),
                ("Tags:", ""),
                ("URL:", ""),
                ("Description:", ""),
                ("  ", " "),
            ]:
                answer = answer.replace(*replacement)

            yield answer.strip()


@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}


@app.get("/search/{sort}/{tags}/{k_tags}/{q}")
def search(k_tags: int, tags: str, sort: bool, q: str):
    """Search for documents."""
    tags = tags != "null"
    documents = knowledge.search(q=q, tags=tags)
    print(f"Search query: {q}, Tags: {tags}, Documents found: {len(documents)}")
    if bool(sort):
        documents = [
            document
            for _, document in sorted(
                [(document["date"], document) for document in documents],
                key=lambda document: datetime.datetime.strptime(
                    document[0], "%Y-%m-%d"
                ),
                reverse=True,
            )
        ]
    return {"documents": documents}


@app.get("/plot/{k_tags}/{q}", response_class=ORJSONResponse)
def plot(k_tags: int, q: str):
    """Plot tags."""
    result = knowledge.plot(q=q, k_tags=k_tags)
    print(f"Plot query: {q}, Result: {result}")
    return result


@app.on_event("startup")
def start():
    """Intialiaze the pipeline."""
    return knowledge.start()


@app.get("/chat/{k_tags}/{q}")
async def chat(k_tags: int, q: str):
    """LLM recommendation."""
    documents = knowledge.search(q=q, tags=False)
    content = ""
    for document in documents:
        content += "title: " + document["title"] + "\n"
        content += "summary: " + document["summary"][:30] + "\n"
        content += "tags: " + (
            ", ".join(document["tags"] + document["extra-tags"]) + "\n"
        )
        content += "url: " + document["url"] + "\n\n"
    content = "title: ".join(content[:3000].split("title:")[:-1])
    print(f"Chat query: {q}, Content length: {len(content)}")
    return StreamingResponse(async_chat(query=q, content=content), media_type="text/plain")