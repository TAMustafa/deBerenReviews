import json
from typing import List, Dict
import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL


SYSTEM_INSTRUCTIONS = (
    "Je bent een operationeel analist voor een restaurantketen. Lees de aantallen per klachtcategorie en een steekproef van recente negatieve reviews. "
    "Formuleer 3-7 beknopte, uitvoerbare verbetersuggesties voor het bedrijf. Iedere suggestie is specifiek, haalbaar en gericht op operationele verbeteringen (bezetting, processen, kwaliteitscontrole, training, menu, prijsstelling, ambiance). "
    "Vermijd algemene adviezen. Gebruik bewijs uit de data. Geef het antwoord ALLEEN terug als een JSON-lijst met Nederlandstalige strings."
)


def _build_prompt(complaint_counts: Dict[str, int], negative_reviews: List[str], top_terms_per_sentiment: List[List[str]] | None) -> str:
    parts = [
        "Aantallen per klachtcategorie:",
    ]
    for k, v in sorted(complaint_counts.items(), key=lambda x: (-x[1], x[0])):
        parts.append(f"- {k}: {v}")
    parts.append("")
    if top_terms_per_sentiment:
        parts.append("Belangrijkste termen per sentiment (model-afgeleid):")
        for item in top_terms_per_sentiment:
            # item expected as (sentiment, [terms])
            try:
                sentiment, terms = item[0], item[1]
            except Exception:
                continue
            # limit to 15 terms
            terms = list(terms)[:15]
            parts.append(f"- {sentiment}: {', '.join(terms)}")
        parts.append("")
    parts.append("Voorbeeld negatieve reviews (ingekort):")
    for i, r in enumerate(negative_reviews[:50], start=1):
        # Keep each review modest length
        snippet = r.strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        parts.append(f"{i}. {snippet}")
    parts.append("")
    parts.append(
        "Genereer beknopte, kwalitatieve suggesties. Antwoord ALLEEN met een JSON-array van Nederlandstalige strings (geen extra tekst)."
    )
    return "\n".join(parts)


def generate_suggestions_llm(negative_reviews: List[str], complaint_counts: Dict[str, int], top_terms_per_sentiment: List[List[str]] | None = None) -> List[str]:
    """Call Ollama to generate business suggestions using gemma3:latest.

    Returns a list of suggestion strings. Falls back to empty list if the call fails.
    """
    prompt = _build_prompt(complaint_counts, negative_reviews, top_terms_per_sentiment)
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"System Instructions:\n{SYSTEM_INSTRUCTIONS}\n\nUser:\n{prompt}",
        "stream": False,
        # temperature left default for balance; could expose via config if needed
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        # Try parse JSON array
        try:
            suggestions = json.loads(text)
            if isinstance(suggestions, list):
                # Ensure all strings
                suggestions = [str(s).strip() for s in suggestions if str(s).strip()]
                # De-duplicate while preserving order
                seen = set()
                out = []
                for s in suggestions:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                return out
        except json.JSONDecodeError:
            pass
        # If not JSON, split lines heuristically
        lines = [ln.strip("- • ") for ln in text.splitlines() if ln.strip()]
        # keep 3-7
        lines = [ln for ln in lines if len(ln) > 6][:7]
        return lines
    except Exception:
        return []
