import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from chunker import Chunk


class SimpleRetriever:
    """Dense retriever using sentence_transformers+ FAISS flat index"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.records: list[Chunk] | None = None

    def fit(self, records: list[Chunk]) -> None:
        """Encode all chunks and build the FAISS index."""
        self.records = records
        texts = [r.text for r in records]

        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Index built: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks for a query"""
        if self.index is None or self.records is None:
            raise RuntimeError("Call fit() before search()")

        query_embed = self.model.encode([query], convert_to_numpy=True).astype(
            "float32"
        )
        faiss.normalize_L2(query_embed)

        scores, indices = self.index.search(query_embed, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            record = self.records[idx]
            results.append(
                {
                    "chunk_id": record.chunk_id,
                    "text": record.text,
                    "source": record.source,
                    "length": record.length,
                    "score": round(float(score), 4),
                }
            )
        return results
