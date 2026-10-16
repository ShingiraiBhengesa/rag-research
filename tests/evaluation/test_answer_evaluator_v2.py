import inspect
import unittest

from prematurestop.evaluation.answer_scoring import (
    ExtractionResult,
    evaluate_answer_v2,
    extract_short_answer,
    official_scores,
    percentage_point_delta,
    summarize_predictions,
)


class AnswerEvaluatorV2Tests(unittest.TestCase):
    def test_1_exact_answer(self):
        result = evaluate_answer_v2(
            question="Who finished ahead of Libby Mitchell?",
            raw_final_answer="Answer: Eliot Cutler",
            gold_answer="Eliot Cutler",
        )

        self.assertTrue(result.semantic.semantic_correct)
        self.assertEqual(result.official_scores.exact_match, 1.0)

    def test_2_correct_answer_inside_longer_sentence(self):
        result = evaluate_answer_v2(
            question="What programming language was the game created after?",
            raw_final_answer="Answer: The game was created after the Java programming language.",
            gold_answer="Java",
        )

        self.assertTrue(result.semantic.semantic_correct)

    def test_3_alias_match(self):
        result = evaluate_answer_v2(
            question="Which insurance company sponsored the 2014 Curling Skins Game?",
            raw_final_answer="Answer: Travelers Companies",
            gold_answer="Travelers",
            gold_aliases=["Travelers Companies"],
        )

        self.assertTrue(result.semantic.semantic_correct)
        self.assertEqual(result.official_scores.exact_match, 1.0)

    def test_4_semantic_equivalent(self):
        result = evaluate_answer_v2(
            question="What nationality are the band and the artist?",
            raw_final_answer="Answer: Both are from the United States.",
            gold_answer="American",
        )

        self.assertTrue(result.semantic.semantic_correct)
        self.assertEqual(result.semantic.reason, "semantic_equivalence")

    def test_5_yes_no_paraphrase(self):
        result = evaluate_answer_v2(
            question="Are Dodgy and Massive Attack from the same country?",
            raw_final_answer="Answer: Dodgy originated in England, while Massive Attack originated in England as well.",
            gold_answer="yes",
        )

        self.assertTrue(result.semantic.semantic_correct)

    def test_6_correctly_formatted_list(self):
        answer = "Boston, Providence, Hartford, New York City, Philadelphia, Wilmington, Baltimore, and Washington, D.C."
        result = evaluate_answer_v2(
            question="Which major cities are part of the Northeast megalopolis?",
            raw_final_answer=f"Answer: {answer}",
            gold_answer=answer,
        )

        self.assertTrue(result.semantic.semantic_correct)

    def test_7_partial_list(self):
        result = evaluate_answer_v2(
            question="Which major cities are part of the Northeast megalopolis?",
            raw_final_answer="Answer: Boston and Providence.",
            gold_answer="Boston, Providence, Hartford",
        )

        self.assertFalse(result.semantic.semantic_correct)
        self.assertEqual(result.semantic.reason, "partial_list_answer")

    def test_8_gold_in_negated_statement(self):
        result = evaluate_answer_v2(
            question="What programming language was the game named after?",
            raw_final_answer="Answer: It was not named after Java; it was named after Python.",
            gold_answer="Java",
        )

        self.assertFalse(result.semantic.semantic_correct)

    def test_9_gold_mentioned_only_as_alternative(self):
        result = evaluate_answer_v2(
            question="What programming language was the game named after?",
            raw_final_answer="Answer: It could be Java or Python.",
            gold_answer="Java",
        )

        self.assertEqual(result.semantic.semantic_label, "needs_adjudication")

    def test_10_conflicting_answers(self):
        extracted = extract_short_answer("Who won?", "Answer: Either Alice or Bob.")

        self.assertFalse(extracted.extractable)
        self.assertEqual(extracted.reason, "conflicting_answers")

    def test_11_missing_answer(self):
        extracted = extract_short_answer("Who won?", "")

        self.assertFalse(extracted.extractable)
        self.assertEqual(extracted.reason, "no_answer")

    def test_12_extraction_failure(self):
        extracted = extract_short_answer("Who won?", "Answer:")

        self.assertFalse(extracted.extractable)
        self.assertEqual(extracted.reason, "malformed_output")

    def test_13_extractor_cannot_access_gold(self):
        parameters = tuple(inspect.signature(extract_short_answer).parameters)

        self.assertEqual(parameters, ("question", "raw_final_answer"))

    def test_14_clean_and_attack_labels_cannot_affect_scoring(self):
        kwargs = {
            "question": "Who wrote The Devil's Feather?",
            "raw_final_answer": "Answer: Minette Walters wrote The Devil's Feather.",
            "gold_answer": "Minette Walters",
        }

        self.assertNotIn("condition", inspect.signature(evaluate_answer_v2).parameters)
        self.assertEqual(evaluate_answer_v2(**kwargs), evaluate_answer_v2(**kwargs))

    def test_15_correct_alias_maximization(self):
        scores = official_scores("Travelers", "The Travelers Companies", ["Travelers"])

        self.assertEqual(scores.exact_match, 1.0)
        self.assertEqual(scores.token_f1, 1.0)

    def test_16_correct_percentage_calculations(self):
        summary = summarize_predictions(
            [
                _prediction(True, 1.0, 1.0),
                _prediction(False, 0.0, 0.0),
            ]
        )

        self.assertEqual(summary["answer_accuracy"], 0.5)
        self.assertEqual(summary["exact_match"], 0.5)

    def test_17_correct_macro_f1(self):
        summary = summarize_predictions(
            [
                _prediction(False, 0.0, 1.0 / 3.0),
                _prediction(False, 0.0, 2.0 / 3.0),
            ]
        )

        self.assertAlmostEqual(summary["token_f1"], 0.5)

    def test_18_percentage_point_differences(self):
        self.assertAlmostEqual(percentage_point_delta(0.45, 0.60), -15.0)

    def test_19_no_rounding_before_aggregation(self):
        summary = summarize_predictions(
            [
                _prediction(False, 0.0, 1.0 / 3.0),
                _prediction(False, 0.0, 1.0 / 3.0),
                _prediction(False, 0.0, 1.0),
            ]
        )

        self.assertAlmostEqual(summary["token_f1"], 5.0 / 9.0)

    def test_22_single_answer_multiple_candidates_goes_to_adjudication(self):
        result = evaluate_answer_v2(
            question="Who is the grandchild of Stefan Zamoyski?",
            raw_final_answer="Answer: The grandchildren include Elzbieta Zamoyska, Stefan Adam Zamoyski, and Andrzej Zamoyski.",
            gold_answer="Stefan Adam Zamoyski",
        )

        self.assertEqual(result.semantic.semantic_label, "needs_adjudication")


def _prediction(correct, em, f1):
    return {
        "valid_output": True,
        "extractable": True,
        "semantic_correct": correct,
        "semantic_label": "correct" if correct else "incorrect",
        "exact_match": em,
        "token_f1": f1,
        "gold_containment": correct,
    }


if __name__ == "__main__":
    unittest.main()
