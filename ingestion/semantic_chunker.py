from pathlib import Path
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.
    Args:
        markdown_file: Markdown file path.
    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")

def chunk_markdown(markdown_file: str, embeddings) -> list[Document]:
    """
    Generate semantic chunks from markdown.
    Args:
        markdown_file: Markdown file path.
        embeddings: Embedding model (e.g., HuggingFaceEmbeddings).
    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    return splitter.create_documents([markdown_content])

def get_bge_embeddings(model_name: str = "BAAI/bge-m3", device: str = "cpu"):
    """
    Get free local BGE embeddings via sentence-transformers.
    Args:
        model_name: Hugging Face model ID (default: BGE-M3, multilingual, 1024-dim)
        device: "cpu" or "cuda"
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}  # Required for cosine similarity
    )

if __name__ == "__main__":
    # Free local BGE-M3 embeddings (no API key needed)
    # First run will download ~2GB model weights
    embeddings = get_bge_embeddings(
        model_name="BAAI/bge-m3",  # Best free multilingual model
        device="cpu"  # Change to "cuda" if you have GPU
    )

    markdown_file = "data/markdown/2024_Apple.md"
    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )
    print(f"Generated {len(chunks)} chunks\n")
    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()
