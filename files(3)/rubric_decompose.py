"""
Rubric decomposition — turns a faculty's reference answer into a list of
discrete, gradeable rubric points, each with a mark weight.

Tiering (as specified): NLP first, fall back to embeddings.

  Tier 1 (NLP): spaCy dependency parse. Splits at genuine clause boundaries
  -- a coordinating conjunction ("and"/"but") joining two VERBS (i.e. two
  independent clauses) -- while correctly NOT splitting a conjunction that
  just joins two adjectives/nouns modifying the same thing.
  This fixes a real bug an earlier regex-based version of this module had:
  naively splitting on every " and " turned "equal and opposite reaction"
  into two nonsense fragments. Dependency parsing tells the two cases apart:
      "equal and opposite reaction"     -> and(equal, opposite), both ADJ -> don't split
      "X happens, and Y occurs"         -> and(happens, occurs), both VERB -> split
  Numbered lists and sentence/semicolon boundaries are always split (unambiguous).

  Tier 2 (embeddings, fallback): only runs on a chunk Tier 1 left intact
  but which still looks long/compound (heuristic: word count or clause
  markers). Splits the chunk at internal punctuation/relative-clause
  candidates, embeds each candidate half, and accepts the split only if the
  two halves are semantically distinct (low cosine similarity) -- i.e. they
  are actually about different things, not just a long single idea.
  Uses llm_client.embed_texts() (Gemini primary, Ollama fallback); if
  neither is available, this tier silently no-ops and the Tier-1 chunk is
  kept as-is rather than guessing.

Runs once per QUESTION (not per student), so latency doesn't matter here.
"""

import re
import json
import numpy as np
import spacy

import llm_client
import config

try:
    _NLP = spacy.load(config.SPACY_MODEL)
except OSError:
    _NLP = None  # graceful degrade -- Tier 1 falls back to simple sentence/semicolon splitting


SENTENCE_SPLIT_PATTERN = re.compile(
    r"""
    (?:\n\s*\d+[\.\)]\s*)          |  # numbered list markers: "1." "2)"
    (?:(?<=[a-z])\.\s+(?=[A-Z]))   |  # sentence boundaries
    (?:;\s*)                          # semicolons
    """,
    re.VERBOSE,
)

COMPOUND_MARKERS = re.compile(r"\b(also|which|because|so that|resulting in)\b", re.IGNORECASE)


def _spacy_clause_split(sentence: str) -> list:
    """Splits one sentence at coordinating conjunctions that join two verbs
    (independent clauses) -- not conjunctions joining adjectives/nouns."""
    if _NLP is None or not sentence.strip():
        return [sentence]

    doc = _NLP(sentence)
    split_points = []
    for token in doc:
        if token.dep_ == "cc":  # coordinating conjunction ("and", "but", "or")
            head = token.head
            conjuncts = [c for c in head.children if c.dep_ == "conj"]
            for conj in conjuncts:
                if head.pos_ == "VERB" and conj.pos_ == "VERB":
                    split_points.append(token.i)

    if not split_points:
        return [sentence]

    tokens = [t.text_with_ws for t in doc]
    pieces, current = [], ""
    for i, tok_text in enumerate(tokens):
        if i in split_points:
            pieces.append(current.strip())
            current = ""
            continue  # drop the conjunction itself
        current += tok_text
    pieces.append(current.strip())
    return [p for p in pieces if p]


def _rule_based_split(reference_answer: str) -> list:
    text = reference_answer.strip()
    raw_chunks = SENTENCE_SPLIT_PATTERN.split(text)
    chunks = [c.strip(" .;") for c in raw_chunks if c and c.strip(" .;")]

    final = []
    for chunk in chunks:
        final.extend(_spacy_clause_split(chunk))
    final = [c.strip(" .;") for c in final if c.strip(" .;")]
    # "Steps of photosynthesis:" style lead-ins introduce a list; they are not
    # a gradeable fact, so they must not consume marks. Only dropped when real
    # points remain, so a lone header can't empty the rubric.
    content = [c for c in final if not c.endswith(":")]
    return content if content else final


def _is_compound(chunk: str) -> bool:
    word_count = len(chunk.split())
    return word_count > config.RUBRIC_COMPOUND_WORD_COUNT or bool(COMPOUND_MARKERS.search(chunk))


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


CANDIDATE_SPLIT_PATTERN = re.compile(r",\s*(?:which|because|so that|resulting in|also)\s+|,\s+and\s+", re.IGNORECASE)


def _embedding_split_chunk(chunk: str) -> list:
    """Tier 2: only reached for chunks spaCy left intact but that still
    look compound. Tries a candidate split point; accepts it only if
    embeddings confirm the two halves are semantically distinct."""
    candidates = CANDIDATE_SPLIT_PATTERN.split(chunk)
    if len(candidates) < 2:
        return [chunk]  # no plausible split point found -- leave intact

    half_a, half_b = candidates[0].strip(), " ".join(candidates[1:]).strip()
    if not half_a or not half_b:
        return [chunk]

    vectors = llm_client.embed_texts([chunk, half_a, half_b])
    if vectors is None:
        return [chunk]  # no embedding backend available -- don't guess, keep as one point

    _, vec_a, vec_b = vectors
    similarity = _cosine(vec_a, vec_b)

    if similarity < config.RUBRIC_EMBED_SPLIT_ACCEPT_THRESHOLD:
        return [half_a, half_b]
    return [chunk]


def decompose_rubric(reference_answer: str, total_marks: float, use_embedding_fallback: bool = True) -> list:
    """
    Returns:
        [{"id": "r1", "point": "...", "marks": 2.0}, ...]
    Marks are split evenly across final points unless overridden afterwards.
    """
    base_chunks = _rule_based_split(reference_answer)

    final_points = []
    for chunk in base_chunks:
        if use_embedding_fallback and _is_compound(chunk):
            final_points.extend(_embedding_split_chunk(chunk))
        else:
            final_points.append(chunk)

    if not final_points:
        final_points = [reference_answer.strip()]

    n = len(final_points)
    even_mark = round(total_marks / n, 2)
    rubric = []
    running_total = 0
    for i, point in enumerate(final_points):
        mark = even_mark if i < n - 1 else round(total_marks - running_total, 2)
        running_total += mark
        rubric.append({"id": f"r{i+1}", "point": point, "marks": mark})

    return rubric


if __name__ == "__main__":
    ref = ("Newton's third law states that for every action there is an equal and opposite "
           "reaction; forces always occur in pairs acting on two different bodies, "
           "which is why a rocket can propel forward by expelling gas backward.")
    rubric = decompose_rubric(ref, total_marks=6)
    print(json.dumps(rubric, indent=2))
    print(f"\nspaCy loaded: {_NLP is not None}")
    print(f"Backend status: {llm_client.backend_status()}")
