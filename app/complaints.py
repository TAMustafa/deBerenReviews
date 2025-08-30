import re
from collections import Counter
from typing import List, Tuple


def complaint_taxonomy() -> dict:
    """Keyword-based complaint categories derived from hospitality best practices.

    Categories: service, wait_time, food_quality, portion_temp, pricing_value, ambience, order_accuracy, cleanliness.
    """
    return {
        "service": r"bedien|servic|onvriend|gastvrij|attent|aandacht|personeel|medewerker",
        "wait_time": r"wacht|wachttijd|lang|traag|op tijd|duur|snel|verlaat",
        "food_quality": r"slecht|niet lekker|taai|rauw|doorbak|verbrand|kwaliteit|smaak|vies|smerig|klef|droog",
        "portion_temp": r"koud|lauw|warmhoud|temperatuur|koud\w*|afgekoeld",
        "pricing_value": r"duur|rekening|prijs|te duur|prijs\-kwaliteit|overprijs|kosten",
        "ambience": r"muziek|lawaai|geluid|herrie|airco|warm|heet|klimaat|druk|sfeer",
        "order_accuracy": r"vergeten|ontbrak|fout|verkeerd|bon|bestelling|niet gekregen",
        "cleanliness": r"vies|smerig|vuil|schoon|hygi[eë]ne|insect|vlieg",
    }


def tag_complaints(texts: List[str]) -> Tuple[List[List[str]], Counter]:
    """Return per-text categories and overall counts."""
    cats = complaint_taxonomy()
    compiled = {k: re.compile(v) for k, v in cats.items()}
    per_text: List[List[str]] = []
    total = Counter()
    for t in texts:
        found = [k for k, rx in compiled.items() if rx.search(t)]
        per_text.append(found)
        total.update(found)
    return per_text, total
