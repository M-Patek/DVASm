"""Automatic evaluation metrics for annotation quality."""

import re
from typing import Dict, List, Optional

import numpy as np

try:
    from rouge_score import rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    rouge_scorer = None


class MetricsCalculator:
    """Calculate automatic metrics for annotation quality."""

    def __init__(self):
        if _ROUGE_AVAILABLE and rouge_scorer is not None:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        else:
            self.rouge_scorer = None

    def bleu(self, reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
        """
        Calculate BLEU score (simplified implementation).

        Args:
            reference: Reference (ground truth) text
            hypothesis: Generated text to evaluate
            max_n: Maximum n-gram size (default 4)

        Returns:
            Dict with bleu_1, bleu_2, bleu_3, bleu_4 scores
        """
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)

        scores = {}
        for n in range(1, max_n + 1):
            score = self._bleu_n(ref_tokens, hyp_tokens, n)
            scores[f"bleu_{n}"] = score

        return scores

    def _bleu_n(self, ref_tokens: List[str], hyp_tokens: List[str], n: int) -> float:
        """Calculate BLEU-n score."""
        if len(hyp_tokens) < n:
            return 0.0

        # Get n-grams
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        hyp_ngrams = self._get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            return 0.0

        # Count matches
        matches = sum(
            min(hyp_ngrams.get(ngram, 0), ref_ngrams.get(ngram, 0)) for ngram in hyp_ngrams
        )

        # Precision
        precision = matches / len(hyp_tokens) if hyp_tokens else 0.0

        # Brevity penalty
        bp = (
            1.0
            if len(hyp_tokens) > len(ref_tokens)
            else np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
        )

        return bp * precision

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """Get n-gram counts from tokens."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            reference: Reference text
            hypothesis: Generated text

        Returns:
            Dict with rouge1_f, rouge2_f, rougeL_f scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }

    def cider(
        self, references: List[str], hypothesis: str, n: int = 4, sigma: float = 6.0
    ) -> float:
        """
        Calculate CIDEr score (Consensus-based Image Description Evaluation).

        Args:
            references: List of reference texts
            hypothesis: Generated text
            n: Maximum n-gram size
            sigma: Standard deviation for Gaussian penalty

        Returns:
            CIDEr score
        """
        # Tokenize
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens_list = [self._tokenize(ref) for ref in references]

        # Calculate TF-IDF weighted n-gram matching
        score = 0.0
        for i in range(1, n + 1):
            # Get n-grams
            hyp_ngrams = self._get_ngrams(hyp_tokens, i)
            ref_ngrams_list = [self._get_ngrams(ref, i) for ref in ref_tokens_list]

            # Document frequency
            doc_freq = {}
            for ref_ngrams in ref_ngrams_list:
                for ngram in set(ref_ngrams.keys()):
                    doc_freq[ngram] = doc_freq.get(ngram, 0) + 1

            # TF-IDF for hypothesis
            hyp_tfidf = {}
            total_hyp = sum(hyp_ngrams.values())
            for ngram, count in hyp_ngrams.items():
                tf = count / total_hyp if total_hyp > 0 else 0
                idf = np.log(len(references) / doc_freq.get(ngram, 1))
                hyp_tfidf[ngram] = tf * idf

            # TF-IDF for references (average)
            ref_tfidf = {}
            for ref_ngrams in ref_ngrams_list:
                total_ref = sum(ref_ngrams.values())
                for ngram, count in ref_ngrams.items():
                    tf = count / total_ref if total_ref > 0 else 0
                    idf = np.log(len(references) / doc_freq.get(ngram, 1))
                    ref_tfidf[ngram] = ref_tfidf.get(ngram, 0) + tf * idf / len(references)

            # Cosine similarity
            dot_product = sum(
                hyp_tfidf.get(ngram, 0) * ref_tfidf.get(ngram, 0)
                for ngram in set(hyp_tfidf.keys()) & set(ref_tfidf.keys())
            )
            hyp_norm = np.sqrt(sum(v**2 for v in hyp_tfidf.values()))
            ref_norm = np.sqrt(sum(v**2 for v in ref_tfidf.values()))

            if hyp_norm > 0 and ref_norm > 0:
                score += dot_product / (hyp_norm * ref_norm)

        # Average over n-gram sizes
        return score / n

    def meteor(self, reference: str, hypothesis: str, alpha: float = 0.9) -> float:
        """
        Calculate METEOR score (simplified implementation).

        Args:
            reference: Reference text
            hypothesis: Generated text
            alpha: Weight for precision vs recall

        Returns:
            METEOR score
        """
        ref_tokens = set(self._tokenize(reference))
        hyp_tokens = set(self._tokenize(hypothesis))

        # Matches
        matches = len(ref_tokens & hyp_tokens)

        if not hyp_tokens or not ref_tokens:
            return 0.0

        # Precision and recall
        precision = matches / len(hyp_tokens)
        recall = matches / len(ref_tokens)

        # F-mean
        if precision + recall == 0:
            return 0.0

        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        # Fragmentation penalty (simplified)
        # In full METEOR, this accounts for word order
        penalty = 0.5 * (1 - matches / max(len(hyp_tokens), len(ref_tokens)))

        return f_mean * (1 - penalty)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def compute_all(
        self, reference: str, hypothesis: str, cider_references: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.

        Args:
            reference: Primary reference text
            hypothesis: Generated text
            cider_references: Multiple references for CIDEr (optional)

        Returns:
            Dict with all metric scores
        """
        results = {}

        # BLEU
        bleu_scores = self.bleu(reference, hypothesis)
        results.update(bleu_scores)

        # ROUGE
        rouge_scores = self.rouge(reference, hypothesis)
        results.update(rouge_scores)

        # METEOR
        results["meteor"] = self.meteor(reference, hypothesis)

        # CIDEr (if multiple references provided)
        if cider_references:
            results["cider"] = self.cider(cider_references, hypothesis)

        return results


def compare_annotations(
    prediction: Dict, reference: Dict, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compare two annotations and compute metrics.

    Args:
        prediction: Predicted annotation dict
        reference: Reference (gold) annotation dict
        metrics: List of metrics to compute (default: all)

    Returns:
        Dict of metric scores
    """
    calc = MetricsCalculator()

    # Extract captions
    pred_caption = prediction.get("caption", "")
    ref_caption = reference.get("caption", "")

    # Compute metrics
    results = calc.compute_all(ref_caption, pred_caption)

    # Filter if specific metrics requested
    if metrics:
        results = {k: v for k, v in results.items() if k in metrics}

    return results
