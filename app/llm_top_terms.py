import json
from typing import Dict, List, Tuple
import requests
import re
import logging

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

logging.basicConfig(level=logging.INFO)

REFINE_SYSTEM = (
    "You are an NLP assistant cleaning model-derived 'top terms per sentiment'. "
    "Remove tokens that are not meaningful sentiment-bearing terms for restaurant reviews: numbers, stopwords, stems without meaning, brand names, isolated short tokens, duplicates, and non-words. "
    "Return up to 15 concise, human-readable terms per sentiment that best indicate that sentiment. "
    "Output ONLY a compact JSON object mapping sentiment to a list of strings."
)

STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","at","by","from","as","it","is","are","was","were","be","been","being",
    "i","we","you","they","he","she","them","this","that","these","those","very","much","more","most","so","too","not","no","yes",
    # Dutch common stopwords
    "de","het","een","en","of","maar","want","dus","niet","geen","wel","ook","al","dan","als","bij","voor","na","van","met","op","aan",
    "ik","jij","je","u","hij","zij","wij","we","ze","dit","dat","die","deze","daar","hier","veel","weinig","meer","minder",
    # Domain-generic words to drop
    "restaurant","beren","deberen","menukaart","menu","eten","drinken","zaak","bedrijf","filiaal","best","top","lekker",  # keep 'lekker'? we drop generic flavor words; sentiment stays via others
}


def _prefilter_terms(terms: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in terms:
        s = str(t).strip()
        if not s:
            continue
        s_norm = s.lower()
        # Drop short tokens, non-alpha-heavy tokens, and stopwords
        if len(s_norm) < 3:
            continue
        if s_norm in STOPWORDS:
            continue
        # Must be mostly alphabetic and have at least 2 letters, drop tokens containing digits
        if any(ch.isdigit() for ch in s_norm):
            continue
        if sum(1 for ch in s_norm if ch.isalpha()) < 2:
            continue
        if s_norm in seen:
            continue
        seen.add(s_norm)
        out.append(s)
        if len(out) >= 25:  # keep some headroom for LLM to trim further
            break
    return out


def _refine_prompt(top_terms: List[Tuple[str, List[str]]]) -> str:
    lines = ["Top terms per sentiment (raw):"]
    for sentiment, terms in top_terms:
        terms = _prefilter_terms(list(dict.fromkeys(terms)))  # unique + light prefilter
        joined = ", ".join(terms)
        lines.append(f"- {sentiment}: {joined}")
    lines.append("")
    lines.append(
        "Clean and filter the lists. Keep only relevant sentiment-indicative terms for each sentiment. "
        "Return up to 15 concise, human-readable terms per sentiment that best indicate that sentiment. "
        "Output ONLY a compact JSON object mapping sentiment to a list of strings."
    )
    return "\n".join(lines)


def refine_top_terms(top_terms: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """Use Ollama to refine/filter top terms per sentiment.

    Returns the same structure as input: List of (sentiment, [terms]). If the call fails, returns the input.
    """
    prompt = _refine_prompt(top_terms)
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"System Instructions:\n{REFINE_SYSTEM}\n\nUser:\n{prompt}",
        "stream": False,
    }
    def _extract_json(text: str):
        txt = text.strip()
        # Strip common markdown fences
        if txt.startswith("```"):
            txt = txt.strip('`')
            # Try to find the first '{' to end '}' span
        # Extract first JSON object with a simple span search
        start = txt.find('{')
        end = txt.rfind('}')
        if start != -1 and end != -1 and end > start:
            return txt[start:end+1]
        return txt

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        parsed = _extract_json(text)
        obj = json.loads(parsed)
        if isinstance(obj, dict):
            out: List[Tuple[str, List[str]]] = []
            for sentiment, terms in obj.items():
                if not isinstance(terms, list):
                    continue
                # Normalize strings, drop empties, keep <=15
                cleaned = []
                seen = set()
                for t in terms:
                    s = str(t).strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    cleaned.append(s)
                    if len(cleaned) >= 15:
                        break
            out.append((sentiment, cleaned))
            if out:
                logging.info("[LLM] Refine terms: received filtered terms from LLM.")
                return out
    except Exception as e:
        logging.error(f"[LLM] Refine terms error: {e}")
    # Fallback to original
    # Apply prefilter as a minimal improvement
    fallback: List[Tuple[str, List[str]]] = []
    for sentiment, terms in top_terms:
        fallback.append((sentiment, _prefilter_terms(terms)))
    return fallback
