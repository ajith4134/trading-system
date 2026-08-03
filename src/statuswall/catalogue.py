"""Reads the feature catalogue out of FEATURES.md rather than restating it.

The wall must never disagree with the catalogue about what features exist. A
second hand-maintained list would drift the first time someone edited one and
not the other, and the drift would be invisible - the wall would simply stop
mentioning a feature, which reads identically to a feature that was never
added. So FEATURES.md stays the single source and this parses it.

Only tables whose header is a feature table are read. FEATURES.md also carries
a worked Zomma example and an options-library table, and both would otherwise
arrive as features with phases like "0.130".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The two header shapes FEATURES.md uses for feature tables. Section 4 names its
# columns Family/Phase/Verdict; every other section uses Feature/Phase/Notes.
_FEATURE_HEADERS = frozenset({
    ("feature", "phase", "notes"),
    ("family", "phase", "verdict"),
})

_SECTION_RE = re.compile(r"^##\s+(?P<idx>[0-9]+[a-z]?)\.\s+(?P<title>.+?)\s*$")
_MISSED_RE = re.compile(r"\*\*\[MISSED[^\]]*\]\*\*\s*[—-]?\s*")


@dataclass(frozen=True)
class Feature:
    """One catalogued capability and where it sits in the document."""
    section_idx: str
    section_title: str
    name: str
    phase: str
    note: str
    missed: bool

    @property
    def key(self) -> str:
        """A stable identity for the evidence map, free of markdown emphasis.

        Keyed on the visible words rather than the raw cell so that bolding a
        feature name in FEATURES.md - an editorial change - does not silently
        detach it from its probe.
        """
        return normalise_key(self.name)


def normalise_key(name: str) -> str:
    """Reduce a feature cell to letters, digits and single spaces, lowercased."""
    plain = re.sub(r"[`*_]", "", name)
    plain = re.sub(r"[^a-z0-9]+", " ", plain.lower())
    return plain.strip()


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def _is_divider(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def read_catalogue(path: Path) -> list[Feature]:
    """Parse every feature table in FEATURES.md, in document order."""
    features: list[Feature] = []
    section_idx = ""
    section_title = ""
    in_feature_table = False

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for position, line in enumerate(lines):
        heading = _SECTION_RE.match(line)
        if heading:
            section_idx = heading.group("idx")
            section_title = heading.group("title")
            in_feature_table = False
            continue

        if not line.startswith("|"):
            in_feature_table = False
            continue

        cells = _split_row(line)

        # A header row is only a header if a divider follows it. Without that
        # check a data row whose first cell happens to read "Feature" would
        # restart the table.
        if len(cells) == 3 and position + 1 < len(lines) and _is_divider(lines[position + 1]):
            in_feature_table = tuple(c.lower() for c in cells) in _FEATURE_HEADERS
            continue

        if _is_divider(line) or not in_feature_table or len(cells) != 3:
            continue

        name, phase, note = cells
        missed = "[MISSED" in note
        features.append(Feature(
            section_idx=section_idx,
            section_title=section_title,
            name=name,
            phase=phase,
            note=_MISSED_RE.sub("", note).strip(),
            missed=missed,
        ))

    return features
