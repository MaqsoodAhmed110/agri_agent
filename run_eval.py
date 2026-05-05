import sys
import os
import json
from evaluation.run_evaluation import load_dataset, run_agent_query, score_with_ragas
from src.multi_agent_graph import _build_graph

# ─── Configuration ─────────────────────────────────────────────
THRESHOLDS_PATH = "eval_thresholds.json"
CI_RESULTS_PATH = "ci_eval_results.json"

def load_thresholds():
    with open(THRESHOLDS_PATH, "r") as f:
        return json.load(f)

def main():
    print("\nSTARTING CI QUALITY GATE EVALUATION")
    
    # 1. Load thresholds
    thresholds = load_thresholds()
    
    # 2. Initialize Agent Graph
    # We use a non-persistent checkpointer for clean CI runs
    graph = _build_graph(use_checkpointer=False)
    
    # 3. Load Dataset (subset for CI speed)
    dataset = load_dataset()[:5] # Use first 5 test cases for fast CI feedback
    
    ragas_samples = []
    
    # 4. Run Inference
    print(f"Running inference on {len(dataset)} test cases...")
    for item in dataset:
        print(f"  Query: {item['query'][:50]}...")
        result = run_agent_query(graph, item['query'])
        ragas_samples.append({
            "question": item['query'],
            "answer": result['answer'],
            "contexts": result['contexts'],
            "ground_truth": item['ground_truth'],
        })

    # 5. Run Scoring
    scores = score_with_ragas(ragas_samples)
    
    # 6. Quality Gate Check
    passed = True
    summary = []
    
    for metric, config in thresholds.items():
        actual = scores.get(metric, 0.0)
        required = config["threshold"]
        status = "PASS" if actual >= required else "FAIL"
        if status == "FAIL":
            passed = False
        
        summary.append({
            "metric": metric,
            "actual": actual,
            "threshold": required,
            "status": status
        })
        print(f"Metric: {metric:16} | Actual: {actual:.4f} | Required: {required:.4f} | [{status}]")

    # 7. Write machine-readable results
    with open(CI_RESULTS_PATH, "w") as f:
        json.dump({
            "status": "PASS" if passed else "FAIL",
            "metrics": summary
        }, f, indent=2)

    if passed:
        print("\nQUALITY GATE PASSED")
        sys.exit(0)
    else:
        print("\nQUALITY GATE FAILED: Performance below threshold")
        sys.exit(1)

if __name__ == "__main__":
    main()
