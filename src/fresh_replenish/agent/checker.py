"""Numbers only from tools (spec section 11): every number in an answer must come from a tool
output or the question. Rounding is allowed, and so are a fraction stated as a percentage (with
a % sign), small whole counts (0-10, as in "3 days" or a numbered list) and ranks ("90th")."""

import re

DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Also the end of a range like 1.0-9.0; a % sign or a rank suffix stays on the token.
NUMBER = re.compile(r"(?<![\w.%])-?\d[\d,]*(?:\.\d+)?(?:%|(?:st|nd|rd|th)\b)?")
RANK = ("st", "nd", "rd", "th")
# The ways models type percentile labels and percents, read as "90th" and "90%".
TYPED = [(re.compile(pattern, re.IGNORECASE), plain) for pattern, plain in (
    (r"(?<=\d)[ -](?=(?:st|nd|rd|th)\b)", ""),  # 90-th, 90 th
    (r"(\d) *%?[ -]*percentile", r"\1th percentile"),  # 90 percentile, 90 % percentile
    (r"\b(low|median|high) *\((\d+) *%\)", r"\1 (\2th)"),  # Median (50 %)
    (r"\b10 *%? *(?:-|to) *90 *%", "10th-90th"),  # 10 %-90 %, the forecast range
    (r"(?<=\d)(?:[ -]?percent\b| +%)", "%"),  # 90 percent, 90 %
)]
# Numbers written as words above ten; ten and below are counts, like the digits 0-10.
WORDS = re.compile(
    r"\b(?:eleven|twelve|(?:thir|four|fif|six|seven|eigh|nine)teen"
    r"|(?:twen|thir|for|fif|six|seven|eigh|nine)ty|hundred|thousand|million)\b",
    re.IGNORECASE,
)
SMALL = 10
# Models write typographic dashes and spaces (non-breaking hyphens, en dashes, thin spaces).
PLAIN = str.maketrans(
    dict.fromkeys("\u2010\u2011\u2012\u2013\u2212", "-")
    | dict.fromkeys("\u00a0\u2007\u2009\u202f\u2014", " ")
)


def numbers(text: str) -> list[str]:
    """Dates first, then the other number tokens, as written (with any % sign or rank suffix)."""
    text = text.translate(PLAIN)
    for typed, plain in TYPED:
        text = typed.sub(plain, text)
    return DATE.findall(text) + NUMBER.findall(DATE.sub(" ", text))


def _value(token: str) -> float:
    return float(re.sub(r"(%|st|nd|rd|th)$", "", token).replace(",", ""))


def unsupported(answer: str, sources: list[str]) -> list[str]:
    """Number tokens (and numbers in words) in the answer that none of the sources supports."""
    dates = {d for s in sources for d in DATE.findall(s.translate(PLAIN))}
    known = [_value(t) for s in sources for t in numbers(s) if not DATE.fullmatch(t)]
    known += [-k for k in known if k < 0]  # -13.8 may be stated as a 13.8% fall
    percents = known + [100 * k for k in known if abs(k) <= 1]  # 0.9 may be stated as 90%
    words = {w.lower() for s in sources for w in WORDS.findall(s)}
    bad = []
    for token in numbers(answer):
        if DATE.fullmatch(token):
            if token not in dates:
                bad.append(token)
            continue
        if token.endswith(RANK):  # a rank like the 90th percentile, not a quantity
            continue
        value = _value(token)
        if "." not in token and 0 <= value <= SMALL:  # a count; a quantity like 1.0 is checked
            continue
        places = len(token.rstrip("%").split(".")[1]) if "." in token else 0
        pool = percents if token.endswith("%") else known
        if not any(abs(round(k, places) - value) < 1e-9 for k in pool):
            bad.append(token)
    return bad + [w for w in WORDS.findall(answer) if w.lower() not in words]
