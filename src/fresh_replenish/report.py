"""Small helpers shared by the Markdown reports."""

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    """Markdown table with floats rounded to 4 places (no `tabulate` dependency)."""
    rows = [list(df.columns), ["---"] * len(df.columns), *df.round(4).astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
