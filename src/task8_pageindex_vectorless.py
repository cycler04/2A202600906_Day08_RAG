"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    (Mock implementation - can be replaced with actual PageIndex client)
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY not set. Skipping upload.")
        print("  Set PAGEINDEX_API_KEY in .env to use PageIndex.")
        return
    
    try:
        from pageindex import PageIndex
        
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        
        count = 0
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name}
            )
            count += 1
            print(f"  ✓ Uploaded: {md_file.name}")
        
        print(f"✓ Successfully uploaded {count} documents to PageIndex")
    
    except ImportError:
        print("⚠ PageIndex library not installed. Install with: pip install pageindex")
    except Exception as e:
        print(f"⚠ Error uploading to PageIndex: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY not set. Returning mock results.")
        # Return mock fallback
        return _get_fallback_results(query, top_k)
    
    try:
        from pageindex import PageIndex
        
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)
        
        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata,
                "source": "pageindex"
            }
            for r in results
        ]
    
    except ImportError:
        print("⚠ PageIndex library not installed. Using fallback results.")
        return _get_fallback_results(query, top_k)
    except Exception as e:
        print(f"⚠ PageIndex search failed: {e}")
        return _get_fallback_results(query, top_k)


def _get_fallback_results(query: str, top_k: int = 5) -> list[dict]:
    """
    Mock fallback results when PageIndex is not available.
    Làm thêm bước này để hệ thống không break nếu chưa có PageIndex API key.
    """
    from . import task4_chunking_indexing
    
    # Fallback to simple keyword matching from indexed chunks
    chunks = task4_chunking_indexing.load_indexed_chunks()
    
    if not chunks:
        return []
    
    # Simple keyword matching
    query_lower = query.lower()
    query_words = query_lower.split()
    
    scored_chunks = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        # Count matching keywords
        matches = sum(1 for word in query_words if word in content_lower)
        
        if matches > 0:
            score = matches / len(query_words) if query_words else 0
            scored_chunks.append({
                "content": chunk["content"],
                "score": score,
                "metadata": chunk["metadata"],
                "source": "pageindex_fallback"
            })
    
    # Sort by score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_chunks[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")

