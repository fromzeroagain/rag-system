# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from chunker import Chunk


# class SimpleRetriever:
#     """Dense retriever using sentence_transformers+ FAISS flat index"""

#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.index: faiss.IndexFlatIP | None = None
#         self.records: list[Chunk] | None = None

#     def fit(self, records: list[Chunk]) -> None:
#         """Encode all chunks and build the FAISS index."""
#         self.records = records
#         texts = [r.text for r in records]

#         embeddings = self.model.encode(
#             texts, show_progress_bar=True, convert_to_numpy=True
#         ).astype("float32")
#         faiss.normalize_L2(embeddings)
#         self.index = faiss.IndexFlatIP(embeddings.shape[1])
#         self.index.add(embeddings)
#         print(f"Index built: {self.index.ntotal} vectors")

#     def search(self, query: str, top_k: int = 5) -> list[dict]:
#         """Return top_k chunks for a query"""
#         if self.index is None or self.records is None:
#             raise RuntimeError("Call fit() before search()")

#         query_embed = self.model.encode([query], convert_to_numpy=True).astype(
#             "float32"
#         )
#         faiss.normalize_L2(query_embed)

#         scores, indices = self.index.search(query_embed, top_k)
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             record = self.records[idx]
#             results.append(
#                 {
#                     "chunk_id": record.chunk_id,
#                     "text": record.text,
#                     "source": record.source,
#                     "length": record.length,
#                     "score": round(float(score), 4),
#                 }
#             )
#         return

import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chunker import Chunk


def make_chunk_id(source: str, chunk_id: int, text: str) -> str:
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}__chunk_{chunk_id}__{content_hash}"


class ChromaRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir="./chroma_db",
        collection_name: str = "rag_chunks",
    ):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def fit(self, records: list[Chunk]) -> None:
        """1.Fresh index, all ids are new->embed everything->upsert all.
        2. Same document, no content changes: same text->same hash->same ids->all ids are already in DB->skipping embedding entirely.
        3. Documents are updated: changed text->different hash->different ids. Old ids no longer in new set->detected as stale->they get deleted.
        New ids not in DB->embedded and upserted."""

        if not records:
            return

        source = records[0].source

        new_ids = [make_chunk_id(r.source, r.chunk_id, r.text) for r in records]
        new_ids_set = set(new_ids)

        # 1. HERE WE CHECK IF ANY IDS ARE GONNA EXIST FOR THE SAME SOURCE
        exisitng_id_for_source = self.collection.get(
            where={"source": source}, include=[]
        )
        existing_id_for_source_set = set(exisitng_id_for_source["ids"])

        # Stale IDs: exist in chromdb for this source but are not in the new chunk set

        stale_ids = list(existing_id_for_source_set - new_ids_set)

        if stale_ids:
            self.collection.delete(ids=stale_ids)
            print(f"Deleted {len(stale_ids)} stale chunks from previous version")

        # STEP 2 : FIND WHICH CHUNKS ARE GENUINELY NEW

        truly_new_ids = list(new_ids_set - existing_id_for_source_set)
        new_records = [r for r, id_ in zip(records, new_ids) if id_ in truly_new_ids]

        if not new_records:
            print(f"All {len(records)} chunks unchanged skipping embedding.")
            return

        print(f"Embedding {len(new_records)}new/changed chunks")

        # 3. EMBED NEW CHUNKS

        new_texts = [r.text for r in new_records]
        embed_ids = [make_chunk_id(r.source, r.chunk_id, r.text) for r in new_records]
        embeddings = self.model.encode(
            new_texts, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)

        metadata = [
            {"source": r.source, "chunk_id": r.chunk_id, "length": r.length}
            for r in new_records
        ]

        # 4. UPSERT INTO CHROMADB
        self.collection.upsert(
            ids=embed_ids,
            documents=new_texts,
            embeddings=embeddings,
            metadatas=metadata,
        )
        print(f"Chromadb collection has {self.collection.count()} Chunks")

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None):
        query_embed = self.model.encode(
            [query], show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        query_norms = np.linalg.norm(query_embed, axis=1, keepdims=True)
        query_embed = query_embed / np.clip(query_norms, 1e-10, None)

        where = {"source": source_filter} if source_filter else None
        result = self.collection.query(
            query_embeddings=query_embed.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []

        for dist, doc, meta in zip(
            result["distances"][0], result["documents"][0], result["metadatas"][0]
        ):
            score = round(1 - dist, 4)
            output.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "text": doc,
                    "source": meta["source"],
                    "length": meta["length"],
                    "score": score,
                }
            )
        return output

    # UTILITY FUNCTIONS

    def count(self) -> int:
        return self.collection.count()

    def delete_source(self, source: str) -> int:
        existing = self.collection.get(where={"source": source}, include=[])
        ids_to_delete = existing["ids"]
        if not ids_to_delete:
            print(f"No chunks found for source: {source}")
            return 0
        self.collection.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)}chunks for '{source}'")
        return len(ids_to_delete)

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)
