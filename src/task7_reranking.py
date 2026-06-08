"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - RRF (Reciprocal Rank Fusion): Kết hợp scores từ nhiều searchers
    - MMR (Maximal Marginal Relevance): tự implement

Nếu dùng RRF, phương pháp này kết hợp ranking từ semantic + lexical search.
"""

from typing import Optional, Callable
import numpy as np


def rerank_rrf(
    result_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,  # Constant cho RRF formula
) -> list[dict]:
    """
    Rerank bằng Reciprocal Rank Fusion (RRF).
    
    RRF kết hợp ranking từ multiple retrievers (semantic + lexical).
    
    Formula: score(d) = Σ (1 / (k + rank(d)))
    
    Args:
        result_lists: List of result lists từ khác nhau:
                      [[semantic results], [lexical results]]
        top_k: Số lượng kết quả cuối cùng
        k: Constant (mặc định 60)
    
    Returns:
        List of top_k merged results, sorted by RRF score descending
    """
    # Tạo dict để track scores
    rrf_scores = {}
    
    # Tính RRF score cho mỗi document
    for result_list in result_lists:
        for rank, result in enumerate(result_list, 1):
            doc_content = result["content"]
            
            if doc_content not in rrf_scores:
                rrf_scores[doc_content] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "scores": []
                }
            
            # RRF: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            rrf_scores[doc_content]["scores"].append(rrf_score)
    
    # Tính tổng RRF scores
    final_results = []
    for doc_key, item in rrf_scores.items():
        final_score = sum(item["scores"])
        final_results.append({
            "content": item["content"],
            "score": final_score,
            "metadata": item["metadata"]
        })
    
    # Sort by score descending
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    return final_results[:top_k]


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Lưu ý: Yêu cầu model có sẵn locally hoặc Internet kết nối.
    
    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank
        model_name: Cross-encoder model name
    
    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    try:
        from sentence_transformers import CrossEncoder
        
        model = CrossEncoder(model_name)
        
        # Prepare pairs: [(query, candidate_text), ...]
        pairs = [[query, cand["content"]] for cand in candidates]
        
        # Score
        scores = model.predict(pairs)
        
        # Add scores to candidates
        reranked = []
        for cand, score in zip(candidates, scores):
            reranked.append({
                **cand,
                "score": float(score),
                "rerank_method": "cross-encoder"
            })
        
        # Sort and return top_k
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    
    except Exception as e:
        print(f"⚠ Cross-encoder reranking failed: {e}")
        print("  Falling back to original scores")
        return candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    selected = []
    remaining = list(range(len(candidates)))
    
    query_emb = np.array(query_embedding).reshape(1, -1)
    
    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')
        
        for i, idx in enumerate(remaining):
            # Relevance to query
            cand_emb = np.array(candidates[idx]["embedding"]).reshape(1, -1)
            relevance = cosine_similarity(query_emb, cand_emb)[0][0]
            
            # Max similarity to already selected
            max_sim_to_selected = 0
            if selected:
                for sel_idx in selected:
                    sel_emb = np.array(candidates[sel_idx]["embedding"]).reshape(1, -1)
                    sim = cosine_similarity(cand_emb, sel_emb)[0][0]
                    max_sim_to_selected = max(max_sim_to_selected, sim)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx is not None:
            actual_idx = remaining.pop(best_idx)
            selected.append(actual_idx)
    
    # Build result
    result = []
    for idx in selected:
        result.append(candidates[idx])
    
    return result


if __name__ == "__main__":
    # Test RRF
    results1 = [
        {"content": "text1", "score": 0.9, "metadata": {"source": "semantic"}},
        {"content": "text2", "score": 0.8, "metadata": {"source": "semantic"}},
    ]
    results2 = [
        {"content": "text2", "score": 0.7, "metadata": {"source": "lexical"}},
        {"content": "text3", "score": 0.6, "metadata": {"source": "lexical"}},
    ]
    
    merged = rerank_rrf([results1, results2], top_k=3)
    for r in merged:
        print(f"[{r['score']:.3f}] {r['content'][:50]}...")

    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    # TODO: Implement RRF
    #
    # rrf_scores = {}  # content -> score
    # content_map = {}  # content -> full dict
    #
    # for ranked_list in ranked_lists:
    #     for rank, item in enumerate(ranked_list, 1):
    #         key = item["content"]
    #         rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
    #         content_map[key] = item
    #
    # # Sort by RRF score
    # sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    #
    # results = []
    # for content, score in sorted_items[:top_k]:
    #     item = content_map[content].copy()
    #     item["score"] = score
    #     results.append(item)
    #
    # return results
    raise NotImplementedError("Implement rerank_rrf")


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
