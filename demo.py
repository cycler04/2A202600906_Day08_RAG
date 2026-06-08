#!/usr/bin/env python
"""
Demo Script - RAG Pipeline for Drug Law Q&A

Chạy test trên toàn bộ pipeline RAG.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.task3_convert_markdown import convert_all
from src.task4_chunking_indexing import run_pipeline as task4_pipeline
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("RAG Pipeline Demo - Drug Law Q&A Chatbot")
    print("="*70 + "\n")
    
    # Step 1: Convert markdown
    print("[Step 1/4] Converting documents to markdown...")
    try:
        convert_all()
        print("✓ Document conversion completed\n")
    except Exception as e:
        print(f"⚠ Warning: {e}\n")
    
    # Step 2: Chunking & Indexing
    print("[Step 2/4] Creating index (chunking & embedding)...")
    try:
        task4_pipeline()
        print("✓ Indexing completed\n")
    except Exception as e:
        print(f"⚠ Warning: {e}\n")
    
    # Step 3: Test Retrieval
    print("[Step 3/4] Testing retrieval pipeline...")
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý từ 5-100 gam?",
        "Những chất nào thuộc danh mục ma tuý loại I?",
        "Ca sĩ nổi tiếng bị bắt vì bao nhiêu gam ma tuý?",
    ]
    
    for q in test_queries:
        print(f"\n  Q: {q}")
        results = retrieve(q, top_k=3)
        print(f"  A: Tìm thấy {len(results)} kết quả liên quan")
        if results:
            print(f"     Best match score: {results[0]['score']:.3f}")
    
    print("\n✓ Retrieval pipeline tested\n")
    
    # Step 4: Test Generation
    print("[Step 4/4] Testing generation with citations...")
    print("\nGenerating RAG responses...\n")
    
    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("="*70)
        
        try:
            result = generate_with_citation(q, top_k=3)
            print(f"\nA: {result['answer'][:200]}...\n")
            print(f"Sources: {len(result['sources'])} documents")
            print(f"Retrieval source: {result['retrieval_source']}")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n" + "="*70)
    print("✓ Demo completed successfully!")
    print("="*70)
    print("\n💡 Next steps:")
    print("  1. Run Streamlit app: streamlit run app.py")
    print("  2. Run evaluation: python group_project/evaluation/eval_pipeline.py")
    print("  3. Customize prompts and parameters in the code\n")


if __name__ == "__main__":
    main()
