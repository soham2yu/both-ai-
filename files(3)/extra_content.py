from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import segmenter


def find_extra_content(rubric_points: list, student_text: str) -> list:
    """Return student sentences that do not map closely to any rubric point."""
    sentences = segmenter.segment_sentences(student_text)
    if not sentences:
        return []

    if not rubric_points:
        return [{"sentence": s, "similarity": 0.0, "reason": "no rubric points available"} for s in sentences if len(s.split()) >= config.EXTRA_CONTENT_MIN_WORDS]

    rubric_texts = [p.get("point", "") for p in rubric_points if p.get("point")]
    results = []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    for sentence in sentences:
        if len(sentence.split()) < config.EXTRA_CONTENT_MIN_WORDS:
            continue
        best_score = 0.0
        for rubric in rubric_texts:
            if not rubric:
                continue
            try:
                matrix = vectorizer.fit_transform([rubric, sentence])
                score = cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]
            except ValueError:
                score = 0.0
            best_score = max(best_score, float(score))

        if best_score < config.EXTRA_CONTENT_UNCOVERED_THRESHOLD:
            results.append({
                "sentence": sentence,
                "similarity": round(best_score, 3),
                "reason": "sentence does not match any rubric point",
            })
    return results
