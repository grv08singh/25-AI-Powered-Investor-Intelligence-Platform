"""
Example usage of Qdrant vectorstore for the investor intelligence platform.
"""

from dotenv import load_dotenv
from pathlib import Path
from ingestion.semantic_chunker import chunk_markdown, get_bge_embeddings
from vectorstore.qdrant_store import QdrantVectorStore, Retriever

load_dotenv()


def main():
    # Initialize embeddings (BGE-M3)
    embeddings = get_bge_embeddings(
        model_name="BAAI/bge-m3",
        device="cpu"
    )

    # Initialize Qdrant vector store
    vector_store = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name="investor_intelligence"
    )

    # Create collection (only needed once)
    vector_store.create_collection(embedding_dimensions=1024)

    # Process a markdown file
    markdown_file = "data/markdown/2024_Apple.md"
    chunks = chunk_markdown(markdown_file=markdown_file, embeddings=embeddings)

    # Upload chunks to Qdrant
    vector_store.upload_chunks(
        chunks=chunks,
        embeddings=embeddings,
        company="Apple",
        year="2024",
        source_file=markdown_file
    )

    # Use retriever to search
    retriever = vector_store.get_retriever()

    query = "What is Apple's revenue growth?"
    results = retriever.invoke(
        query=query,
        embeddings=embeddings,
        company="Apple",
        year="2024",
        top_k=5
    )

    print(f"\nQuery: {query}\n")
    print(f"Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(doc.page_content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
