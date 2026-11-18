"""CPU-only checks of evidence fusion, stage boundaries, and failure reporting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyper import pipeline
from hyper.prompts import HYPOTHESIS_PROMPT, QUERY_PROMPT, ANSWER_PROMPT


class SyntheticQuestions:
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "question": f"Synthetic question {index}",
            "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
            "answer_idx": "A",
        }


class SyntheticRetriever:
    def retrieve(self, query, k):
        if k != 5:
            raise AssertionError("Each evidence query must request five passages")
        # One shared passage and four distinct passages per query.
        ids = ["shared"] + [f"{query}-{i}" for i in range(4)]
        return [dict(id=i, title=i, content=f"Evidence {i}") for i in ids], [
            5,
            4,
            3,
            2,
            1,
        ]


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, retriever=None, malformed=False):
        calls = []

        def generate(model, messages, *args):
            calls.append(messages)
            self.assertEqual(len(messages), 2)
            if len(calls) == 1:
                response = (
                    "best_guess: A"
                    if malformed
                    else json.dumps(
                        {
                            "best_guess": "A",
                            "best_guess_text": "Alpha",
                            "reasoning": "PRIVATE_HYPOTHESIS_REASON",
                            "confirming_evidence": ["support alpha"],
                            "discriminating_features": ["alpha feature"],
                        }
                    )
                )
            elif len(calls) == 2:
                response = (
                    "x"
                    if malformed
                    else "Query 1: support alpha\nQuery 2: distinguish beta\nQuery 3: feature gamma"
                )
            else:
                response = '{"answer_choice":"A"}'
            return [response] * 2

        with tempfile.TemporaryDirectory() as directory, patch.object(
            pipeline, "BenchmarkDataset", SyntheticQuestions
        ), patch.object(pipeline, "set_global_seed"), patch.object(
            pipeline, "_run_phase_in_subprocess", generate
        ), patch(
            "hyper.retrieval.create_retriever",
            return_value=retriever or SyntheticRetriever(),
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            result = pipeline.evaluate(
                "synthetic-model", max_questions=2, output_dir=directory
            )
            payload = json.loads(Path(result["file"]).read_text())
        return calls, result, payload

    def test_fusion_and_stage_boundaries(self):
        calls, result, payload = self.run_pipeline()
        self.assertEqual(result["correct"], 2)
        self.assertEqual(len(calls), 3)
        for row in payload["results"]:
            evidence = row["modes"]["hyper"]
            self.assertEqual(evidence["num_docs"], 13)
            self.assertEqual(evidence["retrieved_docs"][0]["id"], "shared")
            self.assertEqual(evidence["retrieved_docs"][0]["fused_score"], 15)
            self.assertEqual(len(evidence["retrieved_docs"][0]["query_trace"]), 3)
        self.assertEqual(calls[0][0][0]["content"], HYPOTHESIS_PROMPT["system"])
        self.assertEqual(calls[1][0][0]["content"], QUERY_PROMPT["system"])
        self.assertEqual(calls[2][0][0]["content"], ANSWER_PROMPT["system"])
        self.assertIn("PRIVATE_HYPOTHESIS_REASON", calls[1][0][1]["content"])
        self.assertNotIn("PRIVATE_HYPOTHESIS_REASON", calls[2][0][1]["content"])
        self.assertNotIn("Document [14]", calls[2][0][1]["content"])

    def test_malformed_generation_uses_parser_fallback(self):
        calls, result, payload = self.run_pipeline(malformed=True)
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(calls), 3)
        self.assertEqual(payload["results"][0]["modes"]["hyper"]["num_queries"], 3)

    def test_empty_retrieval_stops_evaluation(self):
        class EmptyRetriever:
            def retrieve(self, query, k):
                return [], []

        with self.assertRaisesRegex(RuntimeError, "Retrieval failed"):
            self.run_pipeline(retriever=EmptyRetriever())

    def test_initialization_error_stops_evaluation(self):
        class BrokenRetriever:
            def _lazy_init(self):
                raise OSError("missing corpus")

        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.run_pipeline(retriever=BrokenRetriever())


if __name__ == "__main__":
    unittest.main()
