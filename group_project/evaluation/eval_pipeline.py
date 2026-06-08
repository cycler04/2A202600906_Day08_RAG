"""
RAG Evaluation Pipeline sử dụng DeepEval.

Đánh giá hiệu suất của RAG pipeline trên golden dataset.
Metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision
"""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.task10_generation import generate_with_citation

try:
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase
except ImportError:
    print("⚠ DeepEval not installed. Install with: pip install deepeval")
    DEEPEVAL_AVAILABLE = False
else:
    DEEPEVAL_AVAILABLE = True

# Paths
EVAL_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"


def create_golden_dataset() -> list[dict]:
    """
    Tạo golden dataset với 15+ cặp Q&A để evaluation.
    """
    golden_dataset = [
        {
            "question": "Hình phạt cho tội tàng trữ trái phép chất ma tuý từ 5-100 gam là gì?",
            "expected_answer": "Phạt từ 1-3 năm tù, phạt bổ sung từ 20-50 triệu đồng",
            "context": "Luật Phòng chống ma tuý 2021"
        },
        {
            "question": "Luật Phòng chống ma tuý 2021 quy định gì về sử dụng ma tuý lần đầu?",
            "expected_answer": "Phạt tiền từ 5-10 triệu đồng hoặc phạt 3-6 tháng tù hoặc cơ sở cải tạo 6-12 tháng",
            "context": "Luật Phòng chống ma tuý 2021, Điều 37"
        },
        {
            "question": "Danh mục chất ma tuý loại I bao gồm những chất nào?",
            "expected_answer": "Heroin, Cocaine, LSD, MDMA, Methamphetamine, Fentanyl",
            "context": "Luật Phòng chống ma tuý 2021, Điều 3"
        },
        {
            "question": "Hình phạt mua bán chất ma tuý dưới 10 gam loại I là bao nhiêu?",
            "expected_answer": "Phạt từ 3-7 năm tù, phạt bổ sung 30-50 triệu đồng",
            "context": "Bộ luật Hình sự, Điều 247"
        },
        {
            "question": "Cơ sở sản xuất, kinh doanh chất ma tuý phải cách xa khu dân cư bao nhiêu mét?",
            "expected_answer": "Tối thiểu 100 mét",
            "context": "Nghị định 105/2021/NĐ-CP"
        },
        {
            "question": "Tiền chất nào dùng để sản xuất Heroin?",
            "expected_answer": "Acetic anhydride",
            "context": "Luật Phòng chống ma tuý, Danh sách tiền chất"
        },
        {
            "question": "Sử dụng lần thứ hai trong vòng 5 năm bị phạt như thế nào?",
            "expected_answer": "Phạt tiền 10-20 triệu hoặc tù 6-12 tháng hoặc cơ sở cải tạo 12-24 tháng",
            "context": "Luật Phòng chống ma tuý 2021"
        },
        {
            "question": "Danh mục chất ma tuý loại II bao gồm những gì?",
            "expected_answer": "Cannabis, Morphine, Codeine, Methadone",
            "context": "Luật Phòng chống ma tuý 2021, Điều 3"
        },
        {
            "question": "Hình phạt cho người tàng trữ trái phép 0.5-5 gam ma tuý loại I?",
            "expected_answer": "Phạt từ 6-12 tháng tù, phạt bổ sung 5-10 triệu đồng",
            "context": "Bộ luật Hình sự, Điều 248"
        },
        {
            "question": "Xét nghiệm ma tuý được thực hiện bằng những phương pháp nào?",
            "expected_answer": "Nước tiểu (thường), máu, tóc, thủy tinh thể",
            "context": "Nghị định 105/2021/NĐ-CP"
        },
        {
            "question": "Sản xuất, chế biến ma tuý từ 100-500 gam bị phạt bao nhiêu?",
            "expected_answer": "Phạt từ 15-20 năm tù, phạt bổ sung 100-300 triệu đồng",
            "context": "Bộ luật Hình sự, Điều 252"
        },
        {
            "question": "Mua bán ma tuý trên 100 gam loại I bị xử lý như thế nào?",
            "expected_answer": "Phạt từ 20 năm tù đến chung thân, phạt bổ sung 300-500 triệu đồng",
            "context": "Bộ luật Hình sự, Điều 247"
        },
        {
            "question": "Vận chuyển 500 gam ma tuý loại I bị phạt bao nhiêu?",
            "expected_answer": "Phạt từ 20 năm tù đến chung thân",
            "context": "Bộ luật Hình sự, Điều 253"
        },
        {
            "question": "Lôi kéo, xúi giục thanh thiếu niên sử dụng ma tuý bị xử phạt như thế nào?",
            "expected_answer": "Phạt từ 5-10 năm tù, phạt bổ sung 50-100 triệu đồng",
            "context": "Bộ luật Hình sự, Điều 250"
        },
        {
            "question": "Che giấu tội phạm ma tuý bị xử phạt bao nhiêu?",
            "expected_answer": "Phạt từ 2-5 năm tù",
            "context": "Bộ luật Hình sự, Điều 256"
        },
        {
            "question": "Ca sĩ nổi tiếng bị bắt vì tàng trữ bao nhiêu gam crystal?",
            "expected_answer": "2.5 gam crystal",
            "context": "VnExpress, bài báo về ca sĩ"
        }
    ]
    
    return golden_dataset


def load_or_create_golden_dataset() -> list[dict]:
    """Load golden dataset từ file hoặc tạo mới."""
    if GOLDEN_DATASET_PATH.exists():
        with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        dataset = create_golden_dataset()
        GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        return dataset


def run_evaluation():
    """
    Chạy evaluation trên golden dataset sử dụng DeepEval.
    """
    if not DEEPEVAL_AVAILABLE:
        print("⚠ DeepEval not available. Skipping evaluation.")
        print("Install with: pip install deepeval")
        return
    
    # Load golden dataset
    print("Loading golden dataset...")
    golden_dataset = load_or_create_golden_dataset()
    print(f"✓ Loaded {len(golden_dataset)} test cases\n")
    
    # Create test cases
    print("="*70)
    print("Generating RAG responses...")
    print("="*70 + "\n")
    
    test_cases = []
    for i, item in enumerate(golden_dataset):
        print(f"[{i+1}/{len(golden_dataset)}] {item['question'][:50]}...")
        
        try:
            result = generate_with_citation(item["question"])
            
            # Prepare retrieval context
            retrieval_context = [c["content"] for c in result["sources"]]
            
            test_case = LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=retrieval_context,
            )
            test_cases.append(test_case)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            continue
    
    if not test_cases:
        print("\n❌ No test cases generated.")
        return
    
    print(f"\n✓ Generated {len(test_cases)} test cases\n")
    
    # Define metrics
    print("="*70)
    print("Defining evaluation metrics...")
    print("="*70 + "\n")
    
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]
    
    print("Metrics:")
    for metric in metrics:
        print(f"  - {metric.__class__.__name__}")
    
    # Run evaluation
    print("\n" + "="*70)
    print("Running evaluation...")
    print("="*70 + "\n")
    
    try:
        results = evaluate(test_cases, metrics)
        
        # Save results
        save_results(results, test_cases)
        
    except Exception as e:
        print(f"⚠ Error during evaluation: {e}")
        print("This might be due to missing OpenAI API key.")
        print("\nPlease set OPENAI_API_KEY in your environment:")
        print("  export OPENAI_API_KEY='your-key-here'")


def save_results(results, test_cases):
    """
    Lưu kết quả evaluation vào file markdown.
    """
    import datetime
    
    content = "# RAG Evaluation Results\n\n"
    content += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Summary
    content += "## Summary\n\n"
    content += f"- Total Test Cases: {len(test_cases)}\n\n"
    
    # Detailed Results
    content += "## Detailed Results\n\n"
    content += "```\n"
    content += str(results)
    content += "\n```\n\n"
    
    # Recommendations
    content += "## Recommendations for Improvement\n\n"
    content += "1. **Retrieve Quality:**\n"
    content += "   - Adjust chunk size (current: 500 tokens)\n"
    content += "   - Try different embedding models\n"
    content += "   - Experiment with overlap percentage\n\n"
    
    content += "2. **Reranking:**\n"
    content += "   - Use cross-encoder models\n"
    content += "   - Tune RRF parameters (k value)\n"
    content += "   - Try different ranking algorithms (MMR, etc.)\n\n"
    
    content += "3. **Generation:**\n"
    content += "   - Fine-tune system prompt\n"
    content += "   - Adjust temperature and top_p\n"
    content += "   - Use different LLM models\n\n"
    
    content += "4. **Data:**\n"
    content += "   - Expand golden dataset\n"
    content += "   - Add more diverse test queries\n"
    content += "   - Include edge cases and tricky questions\n\n"
    
    # Save to file
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✓ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()

    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    # TODO: Implement
    #
    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )
    # from datasets import Dataset
    #
    # eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    #
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     eval_data["question"].append(item["question"])
    #     eval_data["answer"].append(result["answer"])
    #     eval_data["contexts"].append([c["content"] for c in result["sources"]])
    #     eval_data["ground_truth"].append(item["expected_answer"])
    #
    # dataset = Dataset.from_dict(eval_data)
    # result = evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    # )
    # return result.to_pandas()
    raise NotImplementedError("Implement evaluate_with_ragas")


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="DrugLaw_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    # TODO: Implement A/B comparison
    #
    # configs = {
    #     "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
    #     "dense_only": {"use_reranking": False, "alpha": 1.0},
    # }
    #
    # results = {}
    # for config_name, params in configs.items():
    #     # Run eval with this config
    #     ...
    #     results[config_name] = scores
    #
    # return results
    raise NotImplementedError("Implement compare_configs")


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    # TODO: Format and write results
    #
    # content = "# RAG Evaluation Results\n\n"
    # content += "## Overall Scores\n\n"
    # content += "| Metric | Score |\n|--------|-------|\n"
    # ...
    # content += "\n## A/B Comparison\n\n"
    # ...
    # content += "\n## Worst Performers\n\n"
    # ...
    # content += "\n## Recommendations\n\n"
    # ...
    #
    # RESULTS_PATH.write_text(content, encoding="utf-8")
    raise NotImplementedError("Implement export_results")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # TODO: Import your RAG pipeline
    # from src.task10_generation import generate_with_citation
    #
    # Chọn 1 framework:
    # results = evaluate_with_deepeval(pipeline, golden_dataset)
    # results = evaluate_with_ragas(pipeline, golden_dataset)
    # results = evaluate_with_trulens(pipeline, golden_dataset)
    #
    # comparison = compare_configs(pipeline, golden_dataset)
    # export_results(results, comparison)
    print("⚠ Implement evaluation logic and run again!")
