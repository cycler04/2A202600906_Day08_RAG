"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from . import task4_chunking_indexing

# Config
EMBEDDING_MODEL = task4_chunking_indexing.EMBEDDING_MODEL
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Load indexed chunks
    chunks = task4_chunking_indexing.load_indexed_chunks()
    
    if not chunks:
        return []
    
    # Embed query
    query_embedding = embedding_model.encode(query)
    
    # Compute similarity
    chunk_embeddings = np.array([c["embedding"] for c in chunks])
    
    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Get top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                "content": chunks[idx]["content"],
                "score": float(similarities[idx]),
                "metadata": chunks[idx]["metadata"]
            })
    
    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

