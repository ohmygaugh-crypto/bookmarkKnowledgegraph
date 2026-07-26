import json
import os
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

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class Knowledge:
    """This class is a wrapper around the pipeline."""

    def __init__(self) -> None:
        self.pipeline = None

    def start(self):
        """Load the pipeline."""
        try:
            from knowledge_database.pipeline import Pipeline

            with open("database/pipeline.pkl", "rb") as f:
                pipeline_data = pickle.load(f)

            if isinstance(pipeline_data, Pipeline):
                self.pipeline = pipeline_data
            else:
                self.pipeline = Pipeline(
                    documents=pipeline_data.get("documents", []),
                    triples=pipeline_data.get("triples", []),
                    excluded_tags=pipeline_data.get("excluded_tags", {}),
                )
            print("Pipeline loaded successfully")
        except Exception as e:
            print(f"Error loading pipeline: {e}")
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


class LLMConfig(typing.NamedTuple):
    provider: str
    base_url: str
    model: str
    api_key: str
    max_tokens: int
    default_headers: typing.Dict[str, str]


def get_llm_config() -> LLMConfig:
    """Build the active OpenAI-compatible provider configuration."""
    provider = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter."
            )

        default_headers = {}
        if referer := os.environ.get("OPENROUTER_HTTP_REFERER", "").strip():
            default_headers["HTTP-Referer"] = referer
        if title := os.environ.get("OPENROUTER_APP_NAME", "").strip():
            default_headers["X-OpenRouter-Title"] = title

        return LLMConfig(
            provider=provider,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            model=os.environ.get(
                "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"
            ),
            api_key=api_key,
            max_tokens=max_tokens,
            default_headers=default_headers,
        )

    if provider == "lmstudio":
        return LLMConfig(
            provider=provider,
            base_url=os.environ.get(
                "LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1"
            ).rstrip("/"),
            model=os.environ.get("LMSTUDIO_MODEL", "google/gemma-4-e4b"),
            api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            max_tokens=max_tokens,
            default_headers={},
        )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER={provider!r}; use 'openrouter' or 'lmstudio'."
    )


async def async_chat(query: str, content: str, config: LLMConfig):
    """Re-rank documents with the configured OpenAI-compatible provider."""
    client = openai.AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=config.default_headers,
    )
    response = await client.chat.completions.create(
        model=config.model,
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
        max_tokens=config.max_tokens,
        stream=True,
        top_p=1,
    )

    answer = ""
    async for chunk in response:
        content_delta = chunk.choices[0].delta.content
        if content_delta:
            answer += content_delta

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
    config = get_llm_config()
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
    return StreamingResponse(
        async_chat(query=q, content=content, config=config),
        media_type="text/plain",
    )
