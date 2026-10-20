"""When and where a photo was taken, as a filter: "beach photos from 2024", "last December", "everything from Goa".

Dates come from EXIF DateTimeOriginal (camera-local time) and places from EXIF GPS, named offline (places.py). Photos
without them never match a filter. Seasons are the northern hemisphere's (summer = June to August).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .places import fold, place_parts

MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
          "september": 9, "october": 10, "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4,
          "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_NAMES = [n.capitalize() for n in list(MONTHS)[:12]]
SEASONS = {"spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12}  # first month; each lasts three

_M = "|".join(sorted(MONTHS, key=len, reverse=True))
_Y = r"(?:19|20)\d\d"
_PREP = r"(?:(?P<prep>(?:taken\s+|shot\s+)?(?:in|from|during|on|of|over|for))\s+)?(?:the\s+)?"
YEAR_SPAN_RE = re.compile(rf"\b(?:(?:in|from|during|between)\s+)?(?P<y1>{_Y})\s*(?:-|–|to|and|until|till)\s*(?P<y2>{_Y})\b",
                          re.I)
SINCE_RE = re.compile(rf"\bsince\s+(?:(?P<m>{_M})\s+)?(?P<y>{_Y})\b", re.I)
RELATIVE_RE = re.compile(rf"{_PREP}\b(?:(?P<rel>last|this|past)\s+(?P<unit>year|month|week)|(?P<day>today|yesterday))\b",
                         re.I)
SEASON_RE = re.compile(rf"{_PREP}\b(?:(?P<rel>last|this)\s+)?(?P<s>spring|summer|autumn|fall|winter)\b"
                       rf"(?:\s+(?:of\s+)?(?P<y>{_Y})\b)?", re.I)
MONTH_RE = re.compile(rf"{_PREP}\b(?:(?P<rel>last|this)\s+)?(?P<m>{_M})\b(?:\s+(?:of\s+)?(?P<y>{_Y})\b)?", re.I)
YEAR_RE = re.compile(rf"{_PREP}\b(?P<y>{_Y})\b", re.I)
PLACE_PREP_RE = re.compile(r"^(?:in|from|at|near|around)$", re.I)
WORD_RE = re.compile(r"[^\W\d_][\w'’-]*", re.U)


@dataclass
class PhotoFilter:
    ranges: list[tuple[str, str]] = field(default_factory=list)  # [start, end) dates as "YYYY-MM-DD"; any of them
    months: tuple[int, ...] = ()  # these months of any year
    places: list[str] = field(default_factory=list)  # stored place names; any of them
    when: str = ""  # how the answer says it: "from 2024", "from December (any year)"
    where: str = ""  # "Goa"

    @property
    def dated(self) -> bool:
        return bool(self.ranges or self.months)

    def __bool__(self) -> bool:
        return self.dated or bool(self.places)

    def sql(self) -> tuple[str, list]:
        """A condition on the assets table (alias `a`)."""
        parts, params = [], []
        if self.ranges:
            parts.append("(" + " OR ".join("(a.taken_at >= ? AND a.taken_at < ?)" for _ in self.ranges) + ")")
            params += [x for r in self.ranges for x in r]
        if self.months:
            parts.append(f"(a.taken_at IS NOT NULL AND CAST(substr(a.taken_at, 6, 2) AS INTEGER) IN "
                         f"({','.join('?' * len(self.months))}))")
            params += list(self.months)
        if self.places:
            parts.append(f"a.place IN ({','.join('?' * len(self.places))})")
            params += self.places
        return " AND ".join(parts) or "1", params

    def describe(self) -> str:
        """ "from 2024 in Goa" """
        return " ".join(x for x in (self.when, f"in {self.where}" if self.where else "") if x)


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def _span(start: date, end: date) -> tuple[str, str]:
    return start.isoformat(), end.isoformat()


def _month_label(d: date) -> str:
    return f"{MONTH_NAMES[d.month - 1]} {d.year}"


def _season_span(first: int, year: int) -> tuple[date, date]:
    start = date(year, first, 1)
    return start, _add_months(start, 3)


def _season_label(name: str, start: date, end: date) -> str:
    last = _add_months(end, -1)
    years = f"{start.year}" if start.year == last.year else f"{start.year}–{last.year}"
    return f"{name} {years} ({MONTH_NAMES[start.month - 1]}–{MONTH_NAMES[last.month - 1]})"


def parse_when(text: str, today: date) -> tuple[PhotoFilter, str] | None:
    """The first time phrase in `text` as a filter, and the text without it. None if there is none."""
    def done(m: re.Match, f: PhotoFilter) -> tuple[PhotoFilter, str]:
        return f, (text[:m.start()] + " " + text[m.end():]).strip()

    if (m := YEAR_SPAN_RE.search(text)) is not None:
        y1, y2 = sorted((int(m["y1"]), int(m["y2"])))
        return done(m, PhotoFilter([_span(date(y1, 1, 1), date(y2 + 1, 1, 1))], when=f"from {y1} to {y2}"))
    if (m := SINCE_RE.search(text)) is not None:
        start = date(int(m["y"]), MONTHS[m["m"].lower()] if m["m"] else 1, 1)
        label = _month_label(start) if m["m"] else str(start.year)
        return done(m, PhotoFilter([_span(start, today + timedelta(days=1))], when=f"since {label}"))
    if (m := RELATIVE_RE.search(text)) is not None:
        tomorrow = today + timedelta(days=1)
        if m["day"]:
            day = today if m["day"].lower() == "today" else today - timedelta(days=1)
            return done(m, PhotoFilter([_span(day, day + timedelta(days=1))], when=f"from {m['day'].lower()}"))
        rel, unit = m["rel"].lower(), m["unit"].lower()
        if rel == "past" or unit == "week":
            days = {"year": 365, "month": 30, "week": 7}[unit]
            start = today - timedelta(days=today.weekday()) if (rel, unit) == ("this", "week") else \
                today - timedelta(days=days - 1)
            label = "this week" if (rel, unit) == ("this", "week") else f"the last {days} days"
            return done(m, PhotoFilter([_span(start, tomorrow)], when=f"from {label}"))
        if unit == "year":
            y = today.year - (rel == "last")
            return done(m, PhotoFilter([_span(date(y, 1, 1), date(y + 1, 1, 1))], when=f"from {rel} year ({y})"))
        start = _add_months(date(today.year, today.month, 1), -1 if rel == "last" else 0)
        return done(m, PhotoFilter([_span(start, _add_months(start, 1))],
                                   when=f"from {rel} month ({_month_label(start)})"))
    if (m := SEASON_RE.search(text)) is not None and (m["prep"] or m["rel"] or m["y"]):
        name, first = m["s"].lower(), SEASONS[m["s"].lower()]
        if m["y"] or m["rel"]:
            if m["y"]:
                year = int(m["y"])
            elif m["rel"].lower() == "this":  # the one we're in, else the next one this year
                year = today.year - (first == 12 and today.month < 3)
            else:  # the latest one that has ended
                year = next(y for y in range(today.year, today.year - 3, -1) if _season_span(first, y)[1] <= today)
            start, end = _season_span(first, year)
            return done(m, PhotoFilter([_span(start, end)], when=f"from {_season_label(name, start, end)}"))
        months = tuple((first - 1 + i) % 12 + 1 for i in range(3))
        return done(m, PhotoFilter(months=months, when=f"from {name} (any year)"))
    for m in MONTH_RE.finditer(text):
        word = m["m"].lower()
        if not (m["prep"] or m["rel"] or m["y"]) or (word in ("may", "mar") and not (m["rel"] or m["y"])
                                                      and m["prep"].lower() not in ("in", "from", "during")):
            continue  # "march" or "may" on their own are usually not months
        month = MONTHS[word]
        if m["y"] or m["rel"]:
            year = int(m["y"]) if m["y"] else today.year - (m["rel"].lower() == "last" and month >= today.month)
            start = date(year, month, 1)
            return done(m, PhotoFilter([_span(start, _add_months(start, 1))], when=f"from {_month_label(start)}"))
        return done(m, PhotoFilter(months=(month,), when=f"from {MONTH_NAMES[month - 1]} (any year)"))
    if (m := YEAR_RE.search(text)) is not None:
        y = int(m["y"])
        return done(m, PhotoFilter([_span(date(y, 1, 1), date(y + 1, 1, 1))], when=f"from {y}"))
    return None


def _place_index(known: list[str]) -> dict[str, tuple[str, list[str]]]:
    """folded name -> (name as stored, the stored places it is part of)."""
    idx: dict[str, tuple[str, list[str]]] = {}
    for place in known:
        for part in place_parts(place):
            idx.setdefault(fold(part), (part, []))[1].append(place)
    return idx


def parse_place(text: str, known: list[str]) -> tuple[PhotoFilter, str] | None:
    """A place the user's own photos were taken in, named after "in/from/at/near", or with a capital letter:
    "everything from Goa", "photos in São Paulo", "Paris sunsets". Only places in `known` are recognised."""
    if not known:
        return None
    idx = _place_index(known)
    words = list(WORD_RE.finditer(text))
    whole = fold(" ".join(w.group() for w in words))
    if whole in idx:
        name, places = idx[whole]
        return PhotoFilter(places=sorted(set(places)), where=name), ""
    for i, w in enumerate(words):
        after_prep = i > 0 and PLACE_PREP_RE.match(words[i - 1].group())
        if not (after_prep or w.group()[0].isupper()):
            continue
        for n in range(min(4, len(words) - i), 0, -1):
            phrase = fold(" ".join(x.group() for x in words[i:i + n]))
            if phrase in idx:
                name, places = idx[phrase]
                start = words[i - 1].start() if after_prep else w.start()
                rest = (text[:start] + " " + text[words[i + n - 1].end():]).strip()
                return PhotoFilter(places=sorted(set(places)), where=name), re.sub(r"\s+", " ", rest)
    return None


def parse_filter(text: str, today: date, known_places: list[str]) -> tuple[PhotoFilter | None, str]:
    """Time and place phrases in a question, as one filter, and the question without them."""
    f, rest = PhotoFilter(), text
    if (w := parse_when(rest, today)) is not None:
        f, rest = w
    if (p := parse_place(rest, known_places)) is not None:
        f.places, f.where = p[0].places, p[0].where
        rest = p[1]
    rest = re.sub(r"\s+", " ", rest).strip()
    rest = re.sub(r"\s+([?.!,])", r"\1", rest)
    return (f if f else None), rest
