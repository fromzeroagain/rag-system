# from pathlib import Path
# from pypdf import PdfReader
# from chunker import clean_text, chunk_text, build_chunk_records

# from retriever import SimpleRetriever


# def load_pdf(path: Path) -> str:
#     reader = PdfReader(path)
#     p = []

#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             p.append(text)
#         else:
#             print(f"page {i + 1} returned no text")
#     return "\n".join(p)


# def ingest(path: Path, chunk_size: int = 500, overlap: int = 100) -> SimpleRetriever:
#     """PDF->CLEAN->CHUNK->EMBED->INDEX"""
#     print(f"Loading {path}...")
#     raw = load_pdf(path)
#     print(f"Cleaning text...")
#     clean = clean_text(raw)
#     print(f"{len(clean)} charracters after cleaning")
#     print(f"Chunking...")
#     chunk = chunk_text(clean, chunk_size=chunk_size, overlap=overlap)
#     print(f"{len(chunk)} chunks created")
#     records = build_chunk_records(chunk, source=path.name)
#     print("Embedding and indexing")
#     retriver = SimpleRetriever()
#     retriver.fit(records)
#     return retriver


# if __name__ == "__main__":
#     pdf_path = Path("../data/samplerag.pdf")
#     if not pdf_path.exists():
#         print(f"ERROR: {pdf_path} not found. Put your PDF in the data/ folder")
#     retriever = ingest(pdf_path)
#     test_queries = [
#         "What is RAG?",
#         "What is a vector database?",
#         "What are limitation of RAG?",
#     ]

#     for i in test_queries:
#         print(f"\nQuery:{i}")
#         results = retriever.search(i, 3)
#         for r in results:
#             print(f"[{r['score']}] chunk {r['chunk_id']}- {r['text']}")


#  DAY 3

from pathlib import Path
from pypdf import PdfReader

from chunker import clean_text, chunk_text, build_chunk_records

from retriever import HybridRetriever, ChromaRetriever


def load_path(path: Path) -> str:
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
        else:
            print(f"[WARN] page {i + 1} has no text")

    return "\n".join(pages)


def ingest(
    pdf_path: Path, chunk_size: int = 500, overlap: int = 100
) -> HybridRetriever:
    print(f"\nLoading {pdf_path.name}")
    raw = load_path(pdf_path)
    print("Cleaning...")
    cleaned = clean_text(raw)
    print(f"{len(cleaned):} characters")
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(chunks)} chunks")
    records = build_chunk_records(chunks, source=pdf_path.name)
    print("\n Fitting HybridRetriever")
    retriever = HybridRetriever()
    retriever.fit(records)
    return retriever


if __name__ == "__main__":
    pdf_path = Path("data/samplerag.pdf")

    if not pdf_path.exists():
        print(f"ERROR {pdf_path} not found")

    hybrid = ingest(pdf_path)
    test_cases = [
        {
            "query": "What is RAG?",
            "note": "Semantic query : both methods should do well",
        },
        {
            "query": "What is Top k retrieval",
            "note": "Exact keyword query : Top k should contribute strongly",
        },
        {
            "query": "Differences between RAG and Fine Tuning",
            "note": "Conceptual query — dense should contribute strongly",
        },
    ]

    dense_only = ChromaRetriever()

    for case in test_cases:
        query = case["query"]
        note = case["note"]

        print(f"{query}\n")
        print(f"{note}")

        print("\nDense only (ChromaDB)")
        dense_results = dense_only.search(query, top_k=3)
        for r in dense_results:
            print(f"score={r['score']},chunk {r['chunk_id']} {r['text']}...")

        print("\nHybrid (BM25 + ChromaDB + RRF)")
        hybrid_results = hybrid.search(query, top_k=3)
        for r in hybrid_results:
            print(f"rrf={r['rrf_score']},chunk {r['chunk_id']}{r['text']}...")
        dense_ids = {r["chunk_id"] for r in dense_results}
        hybrid_ids = {r["chunk_id"] for r in hybrid_results}
        overlap_ids = dense_ids & hybrid_ids
        unique_to_hybrid = hybrid_ids - dense_ids

        print(f"\n  Overlap with dense: {len(overlap_ids)}/3 chunks")
        if unique_to_hybrid:
            print(
                f"  Unique to hybrid (BM25 contribution):"
                f"chunks {sorted(unique_to_hybrid)}"
            )
