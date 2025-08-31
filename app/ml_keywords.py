import re
from typing import List


def extract_keywords_controlled(texts: List[str]) -> List[str]:
    """Map each review to a small set of standardized keywords.

    Vocabulary (extendable):
    - lange_wachten
    - duur
    - service
    - airco
    - eten_koud
    - bestelling_fout
    - hygiene
    - lawaai
    """
    vocab_order = [
        "lange_wachten",
        "duur",
        "service",
        "airco",
        "eten_koud",
        "bestelling_fout",
        "hygiene",
        "lawaai",
    ]

    # Regex patterns per keyword (lowercased input expected)
    patterns = {
        # lange_wachten
        "lange_wachten": re.compile(
            r"\b(lang|lange|lang(e)?\s+moet(en)?\s+wacht(en)?|wachttijd|lang\s+duurd|lang\s+duurt|half\s+uur|kwartier)\b"
        ),
        # duur / pricing complaints
        "duur": re.compile(r"\b(duur|prij(s|z)ig|overpriced|te\s+duur|te\s+duur|euro\s+\d{2,})\b"),
        # service / bediening complaints
        "service": re.compile(
            r"\b(service|bedien(ing)?|personeel|personel|onvriendelijk|genegeerd|slecht\s+bedien)\b"
        ),
        # airco / temperature ambience
        "airco": re.compile(r"\b(airco|airconditioning|benauwd|heet\s+binnen|warm\s+binnen|geen\s+airco)\b"),
        # food temperature
        "eten_koud": re.compile(r"\b(lauw|koud(e)?\s+eten|eten\s+koud|afgekoeld|niet\s+warm)\b|not_koud|eten_koud"),
        # order accuracy
        "bestelling_fout": re.compile(r"\b(bestelling\s+fout|verkeerd(e)?\s+bestelling|mis(s)?ing|vergeten)\b|bestelling\s*fout|not_bestelling"),
        # cleanliness
        "hygiene": re.compile(r"\b(vies|smerig|vuil|hygi[eë]ne|vliegen|insecten|schimmel)\b"),
        # noise/ambience
        "lawaai": re.compile(r"\b(lawaai|hard(e)?\s+muziek|herrie|druk(te)?)\b"),
    }

    out: List[str] = []
    for t in texts:
        s = (t or "").lower()
        found = []
        for key in vocab_order:
            rx = patterns[key]
            if rx.search(s):
                found.append(key)
        out.append(", ".join(found))
    return out
