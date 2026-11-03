"""Numbers only from tools (spec section 11): every number in an answer must come from a tool
output or the question. Rounding is allowed, and so are a fraction stated as a percentage and
small whole counts (0-10, as in "3 days" or a numbered list)."""

import re

DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")  # also the end of a range like 1.0-9.0
SMALL = 10
# Models write typographic dashes and spaces (non-breaking hyphens, en dashes, thin spaces).
PLAIN = str.maketrans(
    dict.fromkeys("\u2010\u2011\u2012\u2013\u2212", "-")
    | dict.fromkeys("\u00a0\u2007\u2009\u202f\u2014", " ")
)


def numbers(text: str) -> list[str]:
    """Dates first, then the other number tokens, as written."""
    text = text.translate(PLAIN)
    return DATE.findall(text) + NUMBER.findall(DATE.sub(" ", text))


def _value(token: str) -> float:
    return float(token.replace(",", ""))


def unsupported(answer: str, sources: list[str]) -> list[str]:
    """Number tokens in the answer that none of the sources supports."""
    dates = {d for s in sources for d in DATE.findall(s.translate(PLAIN))}
    known = [_value(t) for s in sources for t in numbers(s) if not DATE.fullmatch(t)]
    known += [100 * k for k in known if abs(k) <= 1]  # 0.9 may be stated as 90%
    bad = []
    for token in numbers(answer):
        if DATE.fullmatch(token):
            if token not in dates:
                bad.append(token)
            continue
        value = _value(token)
        if "." not in token and 0 <= value <= SMALL:  # a count; a quantity like 1.0 is checked
            continue
        places = len(token.split(".")[1]) if "." in token else 0
        if not any(abs(round(k, places) - value) < 1e-9 for k in known):
            bad.append(token)
    return bad
