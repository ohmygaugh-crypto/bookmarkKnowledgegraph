from ..graph import Graph
from ..retriever import Retriever
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle

__all__ = ["Pipeline"]


class Pipeline:
    """Knowledge Pipeline.

    Example:
    --------

    >>> import json
    >>> from knowledge_database import pipeline

    >>> with open("database/database.json", "r") as f:
    ...     documents = json.load(f)

    >>> with open("database/triples.json", "r") as f:
    ...     triples = json.load(f)

    >>> knowledge_pipeline = pipeline.Pipeline(documents=documents, triples=triples)

    >>> documents, nodes, links = knowledge_pipeline("knowledge graph embeddings")
    >>> nodes, links = knowledge_pipeline.plot("knowledge graph embeddings")
    >>> documents = knowledge_pipeline.search("knowledge graph embeddings")
    >>> documents = knowledge_pipeline.search_documents_tags("knowledge graph embeddings")

    """

    def __init__(self, documents, triples, excluded_tags=None):
        self.retriever = Retriever(documents=documents)
        self.graph = Graph(triples=triples)
        self.excluded_tags = {} if excluded_tags is None else excluded_tags
        
        # Initialize TF-IDF components but don't save them
        # The vectorizer and transformer will be used internally by the retriever
        # but we won't pickle them to avoid binary compatibility issues
        self.vectorizer = TfidfVectorizer()
        self.transformer = TfidfTransformer()
        
        # Save only the necessary data for the pipeline
        with open('database/pipeline.pkl', 'wb') as f:
            pickle.dump({
                'documents': documents,
                'triples': triples,
                'excluded_tags': self.excluded_tags
            }, f)

    def search(self, q: str, tags: bool = False):
        """Search for documents.

        Parameters
        ----------
        q
            Query.
        tags
            If tags is set to True, documents returned will have tags and extra-tags that match the
            query.
        """
        if tags:
            return self.retriever.documents_tags(q)
        return self.retriever.documents(q)

    def __call__(
        self,
        q: str,
        k_tags: int = 20,
        k_yens: int = 3,
        k_walk: int = 3,
    ):
        """Search for documents and tags."""
        documents = self.retriever.documents(q)
        retrieved_tags = self.retriever.tags(q)

        tags = {}
        for document in documents:
            for tag in document["tags"] + document["extra-tags"]:
                if tag not in self.excluded_tags:
                    tags[tag] = True

        nodes, links = self.graph(
            tags=list(tags)[:k_tags],
            retrieved_tags=retrieved_tags,
            k_yens=k_yens,
            k_walk=k_walk,
        )
        return documents, nodes, links

    def plot(self, q: str, k_tags: int = 20, k_yens: int = 3, k_walk: int = 3):
        """Search for tags."""
        _, nodes, links = self(q=q, k_tags=k_tags, k_yens=k_yens, k_walk=k_walk)
        return nodes, links
