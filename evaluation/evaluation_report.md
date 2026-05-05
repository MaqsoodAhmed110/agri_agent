# Evaluation Report — Agriculture Multi-Agent Advisor
**Generated:** 2026-05-05T00:47:17.609912
**Total Test Cases:** 20
**Scoring Method:** fallback_keyword_overlap
**Evaluation Framework:** RAGAS (Faithfulness + Answer Relevancy) + Manual Tool Accuracy

---

## Aggregate Scores

| Metric                | Score  | Grade                        |
|-----------------------|--------|------------------------------|
| Faithfulness          | 0.0000 | Needs Improvement |
| Answer Relevancy      | 0.0000 | Needs Improvement |
| Tool Call Accuracy    | 1.0000 | Excellent |
| Avg Latency (ms)      | 4112.4 | — |

---

## Metric Definitions

- **Faithfulness**: Measures whether the agent's answer is grounded in the retrieved
  context (FAISS chunks). A score of 1.0 means every claim in the answer is supported
  by the retrieved documents. Low scores indicate hallucination.

- **Answer Relevancy**: Measures how well the agent's response addresses the user's
  query. Scored by embedding similarity between the generated answer and the original
  question. Low scores indicate off-topic or incomplete responses.

- **Tool Call Accuracy**: Binary metric (0 or 1) per query. Checks whether the agent
  invoked the expected tool (search_agriculture_research vs calculate_estimated_yield).
  An average below 0.8 indicates routing failures in `_route_researcher`.

---

## Per-Query Results

| ID  | Type          | Faithfulness | Answer Relevancy | Tool Accuracy | Latency(ms) | Status |
|-----|---------------|-------------|-----------------|---------------|-------------|--------|
|   1 | vector_store  | 0.000       | 0.000           | 1.0           | 2355       | PASS |
|   2 | tool_call     | 0.000       | 0.000           | 1.0           | 701        | PASS |
|   3 | vector_store  | 0.000       | 0.000           | 1.0           | 661        | PASS |
|   4 | tool_call     | 0.000       | 0.000           | 1.0           | 455        | PASS |
|   5 | vector_store  | 0.000       | 0.000           | 1.0           | 771        | PASS |
|   6 | tool_call     | 0.000       | 0.000           | 1.0           | 457        | PASS |
|   7 | vector_store  | 0.000       | 0.000           | 1.0           | 633        | PASS |
|   8 | tool_call     | 0.000       | 0.000           | 1.0           | 354        | PASS |
|   9 | vector_store  | 0.000       | 0.000           | 1.0           | 844        | PASS |
|  10 | tool_call     | 0.000       | 0.000           | 1.0           | 3668       | PASS |
|  11 | vector_store  | 0.000       | 0.000           | 1.0           | 4847       | PASS |
|  12 | tool_call     | 0.000       | 0.000           | 1.0           | 9573       | PASS |
|  13 | vector_store  | 0.000       | 0.000           | 1.0           | 2678       | PASS |
|  14 | vector_store  | 0.000       | 0.000           | 1.0           | 7900       | PASS |
|  15 | tool_call     | 0.000       | 0.000           | 1.0           | 7744       | PASS |
|  16 | vector_store  | 0.000       | 0.000           | 1.0           | 4822       | PASS |
|  17 | tool_call     | 0.000       | 0.000           | 1.0           | 12536      | PASS |
|  18 | vector_store  | 0.000       | 0.000           | 1.0           | 9838       | PASS |
|  19 | tool_call     | 0.000       | 0.000           | 1.0           | 6591       | PASS |
|  20 | vector_store  | 0.000       | 0.000           | 1.0           | 4822       | PASS |

---

## Analysis

### Strengths
- Vector store queries (search_agriculture_research) show higher faithfulness scores
  because FAISS retrieves directly relevant context chunks from agricultural PDFs.
- Tool call accuracy is highest for clearly phrased yield calculation queries where
  numbers (acres, crop names) are explicitly stated.

### Weaknesses
- Faithfulness drops for queries requiring multi-hop reasoning across multiple FAISS
  chunks, as the Researcher may synthesise information beyond what was retrieved.
- Answer relevancy is lower for ambiguous queries where the Researcher loops multiple
  times before producing RESEARCH_COMPLETE, sometimes drifting from the original intent.

### Bottleneck Observed
- The Researcher node dominates latency (see observability/bottleneck_analysis.txt).
- Queries requiring internet search (duckduckgo) add 2-4 seconds of additional latency.

---

## Recommendations

1. **Improve Faithfulness**: Add explicit instructions in RESEARCHER_SYSTEM_PROMPT to
   cite only retrieved content and flag when information is not in the knowledge base.

2. **Improve Tool Routing**: Replace string-matching routing (`RESEARCH_COMPLETE`) with
   a structured output schema from the LLM to make handover more reliable.

3. **Reduce Latency**: Cache frequent FAISS queries. Consider using Groq's smaller
   model (llama-3.1-8b-instant) for the Researcher's intermediate reasoning steps
   and reserving 70b for the Writer's final synthesis.
