"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter
# Vì sao chọn RecursiveCharacterTextSplitter?
#   - Phù hợp với tài liệu pháp luật Việt Nam không có heading cấu trúc rõ ràng
#   - Tách text từ từ (paragraph → sentence → words) để giữ ngữ cảnh
#   - Ổn định, phù hợp cho production, không yêu cầu model embedding lúc chunking
CHUNK_SIZE = 500        # Vì sao chọn 500? Phù hợp với hầu hết LLM context length
                        # Không quá dài (sẽ mất chi tiết), không quá ngắn (context yếu)
CHUNK_OVERLAP = 50      # Vì sao chọn 50? 10% overlap để tránh mất thông tin giữa chunks

CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: BAAI/bge-m3
# Vì sao chọn BAAI/bge-m3?
#   - Multilingual, hỗ trợ tốt tiếng Việt
#   - Dimension 1024, balance giữa dimension cao và speed
#   - Mô hình mã nguồn mở, miễn phí
#   - Hỗ trợ dense + sparse (lexical) search
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Vector store: Weaviate
# Vì sao chọn Weaviate?
#   - Hỗ trợ hybrid search (dense + BM25) built-in
#   - Hỗ trợ full-text search, filtering
#   - Có version cloud miễn phí, hoặc docker local
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict, 'embedding': list} — mỗi item là 1 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            # Generate embedding
            embedding = embedding_model.encode(chunk_text).tolist()
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
                "embedding": embedding
            })
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    
    Sử dụng in-memory store và lưu xuống file .pkl cho persistence.
    (Trong production, nên dùng Weaviate hoặc database thực)
    """
    import pickle
    
    # Tạo thư mục index
    index_dir = STANDARDIZED_DIR.parent / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / f"chunks_{EMBEDDING_MODEL.replace('/', '_')}.pkl"
    
    # Lưu chunks vào file
    with open(index_file, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"✓ Indexed {len(chunks)} chunks to {index_file}")
    
    # Lưu metadata
    metadata = {
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "num_chunks": len(chunks),
    }
    metadata_file = index_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    # Note: chunks already have embeddings from chunk_documents()
    # (vì embedding được tạo trong loop đó)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


def load_indexed_chunks() -> list[dict]:
    """Load chunks từ index file."""
    import pickle
    
    index_dir = STANDARDIZED_DIR.parent / "index"
    index_file = index_dir / f"chunks_{EMBEDDING_MODEL.replace('/', '_')}.pkl"
    
    if not index_file.exists():
        print(f"⚠ Index file not found at {index_file}")
        print("  Run run_pipeline() to create index first.")
        return []
    
    with open(index_file, "rb") as f:
        chunks = pickle.load(f)
    
    return chunks


if __name__ == "__main__":
    run_pipeline()
