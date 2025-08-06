import re


def qualis_value(q: str | None) -> int:
    if not (m := re.search(r"^(A|B|C)(1|2|3|4)?$", q or "")):
        return 9
    grade, level = m.groups()
    return (ord(grade) - ord("A") << 2) + ord(level or "1") - ord("1")


def qualis_label(q: int) -> str:
    if q == 9:
        return "?"
    grade = chr((q >> 2) + ord("A"))
    level = chr((q & 3) + ord("1")) if grade != "C" else ""
    return grade + level
