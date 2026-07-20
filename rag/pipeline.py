"""Pipeline Lite RAG : indexation du corpus puis réponse aux questions."""

import time

from rag import config, llm, monitoring
from rag.loader import load_and_split
from rag.vectorstore import VectorStore


class RagPipeline:
    def __init__(
        self,
        corpus_path: str = config.CORPUS_PATH,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        model: str = config.OLLAMA_MODEL,
    ):
        self.corpus_path = corpus_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.store = VectorStore(collection_name=f"corpus-{chunk_size}")
        self._build_index()

    def _build_index(self) -> None:
        chunks = load_and_split(self.corpus_path, self.chunk_size, self.chunk_overlap)
        self.n_chunks = len(chunks)
        if self.store.count == 0:
            self.store.index(chunks)

    def ask(self, question: str, top_k: int = config.TOP_K) -> dict:
        """Répond à une question et retourne réponse, passages, métriques."""
        start = time.perf_counter()
        status = "ok"
        answer = ""
        passages: list[dict] = []
        try:
            passages = self.store.search(question, top_k)
            answer = llm.generate(
                question,
                [p["text"] for p in passages],
                model=self.model,
                url=config.OLLAMA_URL,
                timeout=config.LLM_TIMEOUT,
            )
            low = answer.lower()
            if "je ne trouve" in low and "dans le corpus" in low:
                status = "hors_corpus"
        except llm.OllamaUnavailableError:
            status = "ollama_indisponible"
            raise
        except Exception:
            status = "erreur"
            raise
        finally:
            elapsed = time.perf_counter() - start
            monitoring.log_query(
                question=question,
                model=self.model,
                elapsed_seconds=elapsed,
                n_segments=len(passages),
                status=status,
            )
        return {
            "answer": answer,
            "passages": passages,
            "n_segments": len(passages),
            "elapsed_seconds": round(elapsed, 2),
            "status": status,
        }
