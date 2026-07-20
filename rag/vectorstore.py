"""Base vectorielle locale (ChromaDB, en mémoire).

Les embeddings sont calculés par la fonction d'embedding par défaut de
ChromaDB (all-MiniLM-L6-v2 via ONNX, exécutée localement).
"""

# Streamlit Community Cloud : remplace le sqlite3 système (trop ancien
# pour ChromaDB) par pysqlite3-binary quand il est disponible
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb


class VectorStore:
    def __init__(self, collection_name: str = "corpus"):
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)

    def index(self, chunks: list[str]) -> None:
        """Indexe les segments (calcul des embeddings + stockage)."""
        self._collection.add(
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            documents=chunks,
        )

    def search(self, question: str, top_k: int) -> list[dict]:
        """Retourne les `top_k` segments les plus proches de la question."""
        results = self._collection.query(query_texts=[question], n_results=top_k)
        documents = results["documents"][0]
        distances = results["distances"][0]
        return [
            {"text": doc, "distance": round(dist, 4)}
            for doc, dist in zip(documents, distances)
        ]

    @property
    def count(self) -> int:
        return self._collection.count()
