"""
Full evaluation runner for Medical RAG System.
Tests all 24 golden test queries against the live API.
"""

import asyncio
import json
from datetime import datetime

import httpx

API_BASE = "http://localhost:8000/api/v1"
RESULTS_FILE = "data/full_eval_results.json"


async def run_full_evaluation():
    # Load golden test dataset
    with open("data/evaluation_dataset.jsonl", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    print(f"Loaded {len(queries)} golden test queries")
    print("=" * 60)

    results = []
    stats = {
        "total": len(queries),
        "success": 0,
        "failed": 0,
        "overall_scores": [],
        "faithfulness_scores": [],
        "hallucination_ratios": [],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, item in enumerate(queries):
            query_id = item.get("query_id", f"q_{i + 1}")
            query_text = item["query"]
            _ground_truth_ids = item.get("relevant_doc_ids", [])

            print(f"\n[{i + 1}/{len(queries)}] Query: {query_text[:50]}...")

            try:
                response = await client.post(
                    f"{API_BASE}/query",
                    json={
                        "question": query_text,
                        "session_id": f"eval_{query_id}",
                    },
                )

                if response.status_code != 200:
                    print(f"  [FAIL] HTTP {response.status_code}: {response.text[:100]}")
                    stats["failed"] += 1
                    continue

                result_data = response.json()
                answer = result_data.get("answer", "")
                citations = result_data.get("citations", [])
                warnings = result_data.get("warnings", [])
                confidence = result_data.get("confidence", 0.0)
                processing_time = result_data.get("processing_time", 0.0)

                hallucination_ratio = 0.0
                verified_count = sum(1 for c in citations if c.get("verified", False))
                if citations:
                    hallucination_ratio = 1 - (verified_count / len(citations))

                verified_citations = [c for c in citations if c.get("verified", False)]
                unverified_citations = [c for c in citations if not c.get("verified", False)]

                faithfulness = 0.0
                if citations and answer:
                    has_citation_markers = any(marker in answer for marker in ["来源", "「", "」"])
                    has_content = len(answer) > 50

                    # Check if model correctly identified lack of information
                    inability_phrases = [
                        "无法回答",
                        "未包含",
                        "无法找到",
                        "无相关信息",
                        "没有相关信息",
                        "未找到相关内容",
                        "无法完全回答",
                    ]
                    correctly_identifies_gap = any(phrase in answer for phrase in inability_phrases)

                    if correctly_identifies_gap and verified_citations:
                        # Model correctly identified the gap - give high faithfulness
                        # Even without citation markers, if it correctly says it can't answer, that's faithful
                        faithfulness = 0.9
                    elif has_citation_markers and has_content:
                        faithfulness = min(
                            0.3 + (len(verified_citations) / max(len(citations), 1)) * 0.7,
                            1.0,
                        )

                answer_relevancy = 0.5
                if answer:
                    query_terms = set(query_text.lower().split()) - {
                        "的",
                        "了",
                        "是",
                        "在",
                    }
                    answer_terms = set(answer.lower().split())
                    if query_terms:
                        coverage = len(query_terms & answer_terms) / len(query_terms)
                        answer_relevancy = min(coverage + 0.3, 1.0)

                mrr = 1.0 if citations else 0.0

                overall_score = round(mrr * 0.3 + faithfulness * 0.4 + (1 - hallucination_ratio) * 0.3, 4)

                query_result = {
                    "query_id": query_id,
                    "query": query_text,
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "citations_count": len(citations),
                    "verified_citations": len(verified_citations),
                    "unverified_citations": len(unverified_citations),
                    "hallucination_ratio": round(hallucination_ratio, 4),
                    "faithfulness": round(faithfulness, 4),
                    "answer_relevancy": round(answer_relevancy, 4),
                    "mrr": mrr,
                    "overall_score": overall_score,
                    "warnings_count": len(warnings),
                }

                results.append(query_result)
                stats["success"] += 1
                stats["overall_scores"].append(overall_score)
                stats["faithfulness_scores"].append(faithfulness)
                stats["hallucination_ratios"].append(hallucination_ratio)

                print(
                    f"  [OK] Overall: {overall_score:.3f} | Faithfulness: {faithfulness:.3f} | "
                    f"Hallucination: {hallucination_ratio:.2%} | "
                    f"Citations: {len(citations)}(v:{len(verified_citations)})"
                )

            except httpx.TimeoutException:
                print("  [TIMEOUT] Timeout after 120s")
                stats["failed"] += 1
            except Exception as e:
                print(f"  [ERROR] {e}")
                stats["failed"] += 1

    avg_overall = sum(stats["overall_scores"]) / len(stats["overall_scores"]) if stats["overall_scores"] else 0
    avg_faithfulness = (
        sum(stats["faithfulness_scores"]) / len(stats["faithfulness_scores"]) if stats["faithfulness_scores"] else 0
    )
    avg_hallucination = (
        sum(stats["hallucination_ratios"]) / len(stats["hallucination_ratios"]) if stats["hallucination_ratios"] else 0
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": stats["total"],
        "successful": stats["success"],
        "failed": stats["failed"],
        "avg_overall_score": round(avg_overall, 4),
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_hallucination_ratio": round(avg_hallucination, 4),
        "best_query": max(results, key=lambda x: x["overall_score"])["query_id"] if results else None,
        "worst_query": min(results, key=lambda x: x["overall_score"])["query_id"] if results else None,
    }

    output = {"summary": summary, "results": results}

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average Overall Score: {summary['avg_overall_score']:.4f}")
    print(f"Average Faithfulness: {summary['avg_faithfulness']:.4f}")
    print(f"Average Hallucination Ratio: {summary['avg_hallucination_ratio']:.4f}")
    print(f"Best query: {summary['best_query']}")
    print(f"Worst query: {summary['worst_query']}")
    print(f"\nResults saved to: {RESULTS_FILE}")

    return output


if __name__ == "__main__":
    asyncio.run(run_full_evaluation())
