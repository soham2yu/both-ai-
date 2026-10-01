"""
Matcher / diff engine — the core of this feature.

For each rubric point, decide: matched / partial / missing / contradicted,
find the evidence sentence in the student's OCR'd text, and verify that
evidence quote actually exists in the source (anti-hallucination guardrail).

Tiering (as specified): NLP first, fall back to embeddings, then LLM only
for what's still ambiguous.

  Tier 1 (NLP): TF-IDF cosine similarity -- classical lexical/n-gram
  overlap. Free, instant, no network. Catches confidently-matched and
  confidently-missing points (same wording or close synonyms/n-grams).

  Tier 2 (embeddings, fallback): only runs on points TF-IDF left ambiguous
  (mid-confidence band -- meaning lexical overlap couldn't decide). Dense
  embeddings (Gemini primary, Ollama fallback via llm_client) catch
  semantically-equivalent answers phrased in completely different words,
  which TF-IDF misses by design.

  Tier 3 (LLM, last resort): only for points STILL ambiguous after
  embeddings -- a genuine judgment call (e.g. "does this count as
  contradicting the rubric point, or just an incomplete version of it").
  Gemini primary, Ollama fallback, via llm_client.

If a tier is unavailable (no API key, no Ollama, no internet), the
pipeline falls back to the previous tier's verdict rather than guessing --
never blocks, never crashes.
"""

import re
import json
import difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import llm_client
import config
import segmenter

HIGH_THRESHOLD = config.MATCHER_TFIDF_HIGH_THRESHOLD   # TF-IDF >= this -> matched, no further tiers needed
LOW_THRESHOLD = config.MATCHER_TFIDF_LOW_THRESHOLD      # TF-IDF <  this -> missing, no further tiers needed
# between the two -> ambiguous -> try embeddings, then LLM

EMBED_HIGH_THRESHOLD = config.MATCHER_EMBED_HIGH_THRESHOLD   # embedding similarity >= this -> matched
EMBED_LOW_THRESHOLD = config.MATCHER_EMBED_LOW_THRESHOLD     # embedding similarity <  this -> missing
# still between -> try LLM tie-break


def _split_sentences(text: str) -> list:
    """Delegates to segmenter.py, which repairs OCR line-wraps before
    splitting. Kept under this name because extra_content.py imports it."""
    return segmenter.segment_sentences(text)


def _tfidf_similarity_matrix(rubric_points: list, student_sentences: list):
    corpus = rubric_points + student_sentences
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(corpus)
    r_vecs = tfidf[: len(rubric_points)]
    s_vecs = tfidf[len(rubric_points):]
    return cosine_similarity(r_vecs, s_vecs)


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _verify_quote_in_source(quote: str, source_text: str, min_ratio: float = None) -> bool:
    min_ratio = config.MATCHER_QUOTE_VERIFY_MIN_RATIO if min_ratio is None else min_ratio
    """Anti-hallucination guardrail: confirm the 'evidence' quote actually
    appears (allowing for OCR noise) in the student's text before trusting it."""
    if not quote:
        return False
    if quote.lower() in source_text.lower():
        return True
    best_ratio = 0.0
    for sent in _split_sentences(source_text):
        ratio = difflib.SequenceMatcher(None, quote.lower(), sent.lower()).ratio()
        best_ratio = max(best_ratio, ratio)
    return best_ratio >= min_ratio


def _embedding_tiebreak(rubric_point: str, candidate_sentence: str):
    """Tier 2. Returns (status, confidence) or None if no embedding backend
    is available."""
    vectors = llm_client.embed_texts([rubric_point, candidate_sentence])
    if vectors is None:
        return None
    sim = _cosine(vectors[0], vectors[1])
    if sim >= EMBED_HIGH_THRESHOLD:
        return "matched", sim
    if sim < EMBED_LOW_THRESHOLD:
        return "missing", 1 - sim
    return None  # still ambiguous -- let Tier 3 (LLM) decide


def _llm_tiebreak(rubric_point: str, candidate_sentence: str) -> dict:
    """Tier 3. Returns a dict with status/confidence, or None if unavailable
    (caller falls back to whatever the earlier tier decided)."""
    prompt = f"""You are grading one specific rubric point against one sentence from
a student's answer. Judge ONLY this pair.

Rubric point (what must be shown): "{rubric_point}"
Student sentence: "{candidate_sentence}"

Return ONLY a JSON object, nothing else:
{{"status": "matched" | "partial" | "contradicted" | "unrelated", "confidence": 0.0-1.0}}
"""
    parsed = llm_client.generate_json(prompt, temperature=0.0)
    if parsed and parsed.get("status") in {"matched", "partial", "contradicted", "unrelated"}:
        return parsed
    return None


def match_rubric_to_answer(rubric_points: list, student_text: str, use_embedding_tier: bool = True,
                            use_llm_tier: bool = True) -> list:
    """
    rubric_points: [{"id": "r1", "point": "...", "marks": 2.0}, ...]
    student_text: full OCR'd student answer for this question

    Returns per-point results:
        [{"id", "point", "marks_allocated", "status", "evidence_quote",
          "confidence", "marks_awarded", "quote_verified", "resolved_by"}]
    "resolved_by" records which tier made the final call -- useful for
    debugging/demoing which tier is doing the actual work.
    """
    sentences = _split_sentences(student_text)
    points = [rp["point"] for rp in rubric_points]

    if not sentences:
        return [{
            **rp, "marks_allocated": rp["marks"], "status": "missing",
            "evidence_quote": "", "confidence": 0.0, "marks_awarded": 0.0,
            "quote_verified": False, "resolved_by": "no_student_text",
        } for rp in rubric_points]

    sim_matrix = _tfidf_similarity_matrix(points, sentences)
    results = []

    for i, rp in enumerate(rubric_points):
        best_j = sim_matrix[i].argmax()
        best_score = float(sim_matrix[i][best_j])
        best_sentence = sentences[best_j]
        resolved_by = "tfidf"

        if best_score >= HIGH_THRESHOLD:
            status, confidence = "matched", best_score
        elif best_score < LOW_THRESHOLD:
            status, confidence = "missing", 1 - best_score
            best_sentence = ""
        else:
            # Tier 1 ambiguous -- try Tier 2 (embeddings)
            status, confidence = "partial", best_score  # default if nothing resolves it further

            embed_verdict = _embedding_tiebreak(rp["point"], best_sentence) if use_embedding_tier else None
            if embed_verdict:
                status, confidence = embed_verdict
                resolved_by = "embeddings"
                if status == "missing":
                    best_sentence = ""
            elif use_llm_tier:
                # Tier 2 also ambiguous (or unavailable) -- try Tier 3 (LLM)
                llm_verdict = _llm_tiebreak(rp["point"], best_sentence)
                if llm_verdict:
                    status = llm_verdict["status"] if llm_verdict["status"] != "unrelated" else "missing"
                    confidence = llm_verdict["confidence"]
                    resolved_by = "llm"
                    if status == "missing":
                        best_sentence = ""
                else:
                    resolved_by = "tfidf_unresolved"  # every fallback tier was unavailable

        quote_verified = _verify_quote_in_source(best_sentence, student_text) if best_sentence else False
        if best_sentence and not quote_verified:
            status, confidence, best_sentence = "partial", 0.3, ""
            resolved_by = "quote_guardrail"

        raw_marks = {
            "matched": rp["marks"], "partial": rp["marks"] * config.MATCHER_PARTIAL_CREDIT_FRACTION,
            "missing": 0.0, "contradicted": 0.0,
        }[status]
        # Restrict to 0, 0.5, 1, 1.5, 2
        allowed_marks = [0.0, 0.5, 1.0, 1.5, 2.0]
        marks_awarded = min(allowed_marks, key=lambda x: abs(x - raw_marks))

        results.append({
            **rp, "marks_allocated": rp["marks"], "status": status,
            "evidence_quote": best_sentence, "confidence": round(confidence, 3),
            "marks_awarded": marks_awarded, "quote_verified": quote_verified,
            "resolved_by": resolved_by,
        })

    return results


# ---------- Fast paths for MCQ / numeric / one-liner answers ----------

def grade_mcq(student_choice: str, correct_choice: str) -> dict:
    return {"status": "matched" if student_choice.strip().upper() == correct_choice.strip().upper() else "missing"}


def grade_numeric(student_value: float, correct_value: float, tolerance: float = None) -> dict:
    tolerance = config.NUMERIC_TOLERANCE_FRACTION if tolerance is None else tolerance
    if correct_value == 0:
        is_match = abs(student_value) < 1e-9
    else:
        is_match = abs(student_value - correct_value) / abs(correct_value) <= tolerance
    return {"status": "matched" if is_match else "missing"}


def grade_one_liner(student_text: str, correct_text: str, threshold: float = None) -> dict:
    threshold = config.ONE_LINER_MATCH_THRESHOLD if threshold is None else threshold
    fuzzy_ratio = difflib.SequenceMatcher(None, student_text.lower().strip(), correct_text.lower().strip()).ratio()
    tfidf_sim = _tfidf_similarity_matrix([correct_text], [student_text])[0][0]
    score = max(fuzzy_ratio, tfidf_sim)
    if score < threshold:
        embed_vectors = llm_client.embed_texts([correct_text, student_text])
        if embed_vectors is not None:
            score = max(score, _cosine(embed_vectors[0], embed_vectors[1]))
    return {"status": "matched" if score >= threshold else "missing", "confidence": round(float(score), 3)}


if __name__ == "__main__":
    rubric = [
        {"id": "r1", "marks": 3.0, "point": "Newton's third law states that for every action there is an equal and opposite reaction"},
        {"id": "r2", "marks": 3.0, "point": "Forces always occur in pairs acting on two different bodies"},
    ]
    student_answer = ("Newton's 3rd law says every action has an equal and opposite reaction. "
                       "I also like physics a lot because it is interesting.")

    results = match_rubric_to_answer(rubric, student_answer)
    print(json.dumps(results, indent=2))
    print(f"\nBackend status: {llm_client.backend_status()}")
