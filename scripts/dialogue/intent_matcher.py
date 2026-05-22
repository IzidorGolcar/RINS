"""Regex-based intent matcher for the constrained dialogue in Task 2.

The vocabulary is tiny (spec p.9–10):
    barrels | rings | anomaly_red | anomaly_green | nothing
plus confirm / reconsider signals for the woman re-ask flow.

Pure Python, no ML. Importable and unit-testable without ROS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

INTENT_BARRELS = 'barrels'
INTENT_RINGS = 'rings'
INTENT_ANOMALY_RED = 'anomaly_red'
INTENT_ANOMALY_GREEN = 'anomaly_green'
INTENT_NOTHING = 'nothing'

ALL_INTENTS = (
    INTENT_BARRELS,
    INTENT_RINGS,
    INTENT_ANOMALY_RED,
    INTENT_ANOMALY_GREEN,
    INTENT_NOTHING,
)


# Order matters: anomaly_red/green checked before plain "anomaly" so
# the colour qualifier wins over the generic keyword.
_INTENT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (INTENT_ANOMALY_RED,   re.compile(r'(anomal|defect|inspect|detect).*\bred\b'
                                      r'|\bred\b.*(anomal|defect|cell)', re.I)),
    (INTENT_ANOMALY_GREEN, re.compile(r'(anomal|defect|inspect|detect).*\bgreen\b'
                                      r'|\bgreen\b.*(anomal|defect|cell)', re.I)),
    (INTENT_BARRELS,       re.compile(r'\bbarrel|\bdrum', re.I)),
    (INTENT_RINGS,         re.compile(r'\bring(s)?\b|count.*\bring', re.I)),
    (INTENT_NOTHING,       re.compile(r'\bnothing\b|\bno\s+task\b|\bskip\b', re.I)),
)

_CONFIRM = re.compile(r'\b(yes|sure|confirmed|correct|right|yeah|yep)\b', re.I)
_RECONSIDER = re.compile(
    r'\b(no|not\s+(sure|really|exactly)|wait|actually|instead|well|sorry|'
    r'mistake|ah|umm|hmm|maybe)\b',
    re.I,
)

# QR-code payload → intent mapping. The world meshes carry these
# payloads on `qr_*.png` textures (see worlds/task2_*_demo_meshes/).
QR_PAYLOAD_TO_INTENT: dict[str, str] = {
    'qr_barrels':   INTENT_BARRELS,
    'barrels':      INTENT_BARRELS,
    'qr_rings':     INTENT_RINGS,
    'rings':        INTENT_RINGS,
    'qr_redbelt':   INTENT_ANOMALY_RED,
    'redbelt':      INTENT_ANOMALY_RED,
    'qr_greenbelt': INTENT_ANOMALY_GREEN,
    'greenbelt':    INTENT_ANOMALY_GREEN,
    'qr_nothing':   INTENT_NOTHING,
    'nothing':      INTENT_NOTHING,
}


@dataclass
class MatchResult:
    intent: Optional[str]    # None if no match
    confirm: bool            # contains a confirmation cue
    reconsider: bool         # contains a reconsider cue
    raw: str


def classify(utterance: str) -> MatchResult:
    """Return the most likely intent in ``utterance`` plus confirm/reconsider signals."""
    text = (utterance or '').strip()
    intent: Optional[str] = None
    for label, pat in _INTENT_PATTERNS:
        if pat.search(text):
            intent = label
            break
    return MatchResult(
        intent=intent,
        confirm=bool(_CONFIRM.search(text)),
        reconsider=bool(_RECONSIDER.search(text)),
        raw=text,
    )


def classify_qr(payload: str) -> Optional[str]:
    """Map a QR-code text payload to an intent (case-insensitive)."""
    if not payload:
        return None
    return QR_PAYLOAD_TO_INTENT.get(payload.strip().lower())


# Self-test — run with `python3 intent_matcher.py`.
def _self_test() -> None:
    cases = [
        ('Inspect the barrels.',            INTENT_BARRELS,        False, False),
        ('Count the rings.',                INTENT_RINGS,          False, False),
        ('Detect anomalies in the red cell.', INTENT_ANOMALY_RED,  False, False),
        ('Inspect anomalies in the green cell.', INTENT_ANOMALY_GREEN, False, False),
        ('Well, maybe you should inspect the barrels?', INTENT_BARRELS, False, True),
        ('Yes, I am sure.',                 None,                  True,  False),
        ('Ah, not really. Count the rings.', INTENT_RINGS,         False, True),
        ('Nothing.',                        INTENT_NOTHING,        False, False),
        ('',                                None,                  False, False),
    ]
    failed = 0
    for utt, want_intent, want_conf, want_rec in cases:
        r = classify(utt)
        ok = (r.intent == want_intent and r.confirm == want_conf and r.reconsider == want_rec)
        flag = 'OK ' if ok else 'FAIL'
        print(f'{flag} {utt!r:55s} -> intent={r.intent} confirm={r.confirm} reconsider={r.reconsider}')
        failed += 0 if ok else 1
    qr_cases = [
        ('qr_barrels',   INTENT_BARRELS),
        ('QR_RINGS',     INTENT_RINGS),
        ('greenbelt',    INTENT_ANOMALY_GREEN),
        ('unknown_code', None),
    ]
    for payload, want in qr_cases:
        got = classify_qr(payload)
        ok = got == want
        flag = 'OK ' if ok else 'FAIL'
        print(f'{flag} QR {payload!r:20s} -> {got}')
        failed += 0 if ok else 1
    print(f'\n{failed} failure(s)')


if __name__ == '__main__':
    _self_test()
