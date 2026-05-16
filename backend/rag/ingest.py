from pathlib import Path
from pypdf import PdfReader
from chunker import clean_text, chunk_text, build_chunk_records

from retriever import SimpleRetriever


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)
    p = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            p.append(text)
        else:
            print(f"page {i + 1} returned no text")
    return "\n".join(p)


def ingest(path: Path, chunk_size: int = 500, overlap: int = 100) -> SimpleRetriever:
    """PDF->CLEAN->CHUNK->EMBED->INDEX"""
    print(f"Loading {path}...")
    raw = load_pdf(path)
    print(f"Cleaning text...")
    clean = clean_text(raw)
    print(f"{len(clean)} charracters after cleaning")
    print(f"Chunking...")
    chunk = chunk_text(clean, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(chunk)} chunks created")
    records = build_chunk_records(chunk, source=path.name)
    print("Embedding and indexing")
    retriver = SimpleRetriever()
    retriver.fit(records)
    return retriver


if __name__ == "__main__":
    pdf_path = Path("../data/samplerag.pdf")
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found. Put your PDF in the data/ folder")
    retriever = ingest(pdf_path)
    test_queries = [
        "What is RAG?",
        "What is a vector database?",
        "What are limitation of RAG?",
    ]

    for i in test_queries:
        print(f"\nQuery:{i}")
        results = retriever.search(i, 3)
        for r in results:
            print(f"[{r['score']}] chunk {r['chunk_id']}- {r['text']}")
