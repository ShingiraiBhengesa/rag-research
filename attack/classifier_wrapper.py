"""
TextAttack-compatible wrapper for the Adaptive-RAG T5-Large classifier.
Translates the seq2seq model into a 3-class classifier (A/B/C) interface.
"""
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from textattack.models.wrappers import ModelWrapper


class AdaptiveRAGClassifierWrapper(ModelWrapper):
    """Wraps Adaptive-RAG's T5-Large complexity classifier for TextAttack."""

    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        print(f"Loading tokenizer from {model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f"Loading model from {model_path}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.model.eval()

        # Label mapping: A=0, B=1, C=2
        self.labels = ["A", "B", "C"]

        # Token IDs for A, B, C
        self.label_token_ids = {
            lab: self.tokenizer.encode(lab, add_special_tokens=False)[0]
            for lab in self.labels
        }
        print(f"Label token IDs: {self.label_token_ids}")

    def __call__(self, text_input_list):
        """
        TextAttack calls this with a list of strings.
        Returns numpy array of shape (batch_size, 3) with softmax probs for A/B/C.
        """
        results = []
        for text in text_input_list:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=384,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                # Logits for the first generated token
                logits = outputs.scores[0][0]

                # Extract only A/B/C logits
                abc_logits = torch.tensor([
                    logits[self.label_token_ids["A"]].item(),
                    logits[self.label_token_ids["B"]].item(),
                    logits[self.label_token_ids["C"]].item(),
                ])

                probs = torch.softmax(abc_logits, dim=0).numpy()
                results.append(probs)

        return np.array(results)


if __name__ == "__main__":
    MODEL_PATH = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
                  "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
                  "flan_t5_xxl/epoch/30")

    print("=" * 60)
    print("Loading classifier wrapper...")
    print("=" * 60)
    wrapper = AdaptiveRAGClassifierWrapper(MODEL_PATH, device="cuda:0")

    # Test with sample queries of varying complexity
    test_queries = [
        # Expected Route C (multi-hop)
        "What does the acronym of the organization Danish Football Union is part of stand for?",
        "In which city is the headquarters of the company that produces the MacBook located?",
        # Expected Route B (single-hop)
        "What is the capital of France?",
        "Who directed the movie Inception?",
        # Expected Route A (no retrieval needed, general knowledge)
        "What is 2 + 2?",
        "What color is the sky?",
    ]

    print("\n" + "=" * 60)
    print("Testing wrapper on sample queries")
    print("=" * 60)
    probs_batch = wrapper(test_queries)

    for q, p in zip(test_queries, probs_batch):
        pred_idx = int(np.argmax(p))
        pred = wrapper.labels[pred_idx]
        print(f"\nQuery: {q[:70]}")
        print(f"  Probs: A={p[0]:.3f}, B={p[1]:.3f}, C={p[2]:.3f}")
        print(f"  Prediction: Route {pred}")

    print("\n" + "=" * 60)
    print("Wrapper test complete.")
    print("=" * 60)
