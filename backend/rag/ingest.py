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


from pathlib import Path
from pypdf import PdfReader
from chunker import clean_text, chunk_text, build_chunk_records, Chunk
from retriever import ChromaRetriever


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
        else:
            print(f"page {i + 1} returned no text")
    return "\n".join(pages)


def ingest(
    pdf_path: Path, chunk_size: int = 500, overlap: int = 100
) -> tuple[list[Chunk], ChromaRetriever]:
    """Full ingestion: PDF->clean->chunk->embed->chromaDB"""
    print(f"\nLoading {pdf_path.name}...")
    raw = load_pdf(pdf_path)

    print("Cleaning text...")
    cleaned = clean_text(raw)
    print(f"{len(cleaned):,} characters")

    print("Chunking...")
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(chunks)} chunks")

    records = build_chunk_records(chunks, source=pdf_path.name)

    print("Fitting ChromaReriever...")
    retriever = ChromaRetriever()
    retriever.fit(records)

    return records, retriever


if __name__ == "__main__":
    pdf_path = Path("data/samplerag.pdf")

    if not pdf_path.exists():
        print(f"{pdf_path} not found. Put your PDF in the data folder.")

    records, retriever = ingest(pdf_path)

    # test 1
    test_queries = [
        "What is RAG?",
        "What is a vector database?",
        "What are the limitations of RAG?",
    ]

    print("----------------------------------")
    print("basic retrieval")
    print("----------------------------------")

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"[{r['score']}] chunk {r['chunk_id']} → {r['text']}")

    # Test 2: persistence test
    print("----------------------------------")
    print("Persistence:second instance should skip embedding")
    print("----------------------------------")

    retriever2 = ChromaRetriever()
    retriever2.fit(records)
    print(f"retriever2 chunk count:{retriever2.count()}")
    r1 = retriever.search("What is RAG?", top_k=3)
    r2 = retriever2.search("What is RAG?", top_k=3)
    ids_match = [x["chunk_id"] for x in r1] == [x["chunk_id"] for x in r2]
    print(f"Results match between instances:{ids_match}")

    # Test 3: update simulation
    print("----------------------------------")
    print("Update: simulate document content change")
    print("----------------------------------")

    count_before = retriever.count()
    print(f"Chunks before update: {count_before}")
    from chunker import Chunk
    import dataclasses

    modified_records = []
    for i, record in enumerate(records):
        if i < 2:
            modified = dataclasses.replace(
                record, text=f"[UPDATED CONTENT] {record.text}"
            )
            modified_records.append(modified)
        else:
            modified_records.append(record)

    print("Re-fitting with 2 modified chunks")
    retriever.fit(modified_records)

    count_after = retriever.count()
    print(f"Chunks after update: {count_after}")
    print(f"Count unchanged (same number of chunks):{count_before == count_after}")

    updated_results = retriever.search("[UPDATED CONTENT]", top_k=2)
    found_updated = any("[UPDATED CONTENT]" in r["text"] for r in updated_results)
    print(f"Updated content is now searchable: {found_updated}")

    # Test 4: metadata filter test
    print("----------------------------------")
    print("Metadata filter: source_filter restricts results")
    print("----------------------------------")

    # Searching with correct source filter
    results_filtered = retriever.search(
        "What is RAG?",
        top_k=3,
        source_filter=pdf_path.name,
    )
    print(f"  Results with source_filter='{pdf_path.name}': {len(results_filtered)}")
    all_correct_source = all(r["source"] == pdf_path.name for r in results_filtered)
    print(f"All results from correct source: {all_correct_source}")

    # Searching with a non-existent source
    results_wrong = retriever.search(
        "What is RAG?",
        top_k=3,
        source_filter="nonexistent.pdf",
    )
    print(f"Results with wrong source_filter: {len(results_wrong)}")
