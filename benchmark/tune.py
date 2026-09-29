"""Train-split-only CV for the grounding checks (src/veritas_gate/grounding.py).

Never touches the test split. Structural hyperparameters (novelty window size, novelty
threshold) may be tuned here freely. Writes nothing automatically: a human reads the printed
table, picks the winner, and copies it into grounding.py's _NOVELTY_WINDOW/_NOVELTY_THRESHOLD with
a comment citing the number this script printed.

    python benchmark/fetch_data.py     # once, if not already done
    python benchmark/tune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run as _run  # noqa: E402
from metrics import prf  # noqa: E402
from veritas_gate import grounding  # noqa: E402


def _rows_for(train: list, sources: dict, detector) -> list:
    """``detector(response_text, evidence_text) -> bool`` (fired or not). Builds pred/gold rows."""
    rows = []
    for r in train:
        src = sources.get(r["source_id"])
        if not src:
            continue
        rows.append({"pred": detector(r["response"], _run.evidence_for(src)),
                     "gold": bool(r.get("labels"))})
    return rows


def main() -> int:
    responses, sources = _run.load()
    train = [r for r in responses if r.get("split") == "train"]
    n = len(train)
    gold_pos = sum(1 for r in train if r.get("labels"))
    base = gold_pos / n if n else 0.0
    naive_f1 = 2 * base / (base + 1.0)
    print(f"TRAIN split (measured): n={n}, base rate={base*100:.1f}%, "
          f"naive always-positive F1={naive_f1*100:.1f}%\n")

    print("=== novelty window/threshold CV (measured) ===")
    best = None
    for window in (3, 5, 7, 10):
        for threshold in (0.6, 0.8, 1.0):
            def detector(resp, ev, w=window, t=threshold):
                return bool(grounding.novel_content_windows(resp, ev, window=w, threshold=t))
            m = prf(**_score_counts(_rows_for(train, sources, detector)))
            flag = " <- current grounding.py default" if (window, threshold) == (
                grounding._NOVELTY_WINDOW, grounding._NOVELTY_THRESHOLD) else ""
            print(f"  window={window:2d} threshold={threshold:.1f}  "
                  f"P={m['precision']*100:5.1f} R={m['recall']*100:5.1f} F1={m['f1']*100:5.1f}{flag}")
            if best is None or m["f1"] > best[1]["f1"]:
                best = ((window, threshold), m)
    print(f"  winner: window={best[0][0]} threshold={best[0][1]} F1={best[1]['f1']*100:.1f}%\n")

    print("=== standalone checks (measured, current grounding.py defaults) ===")
    standalone = {
        "broadened_numeric": lambda r, e: bool(grounding.broadened_numeric_ungrounded(r, e)),
        "entity_grounding": lambda r, e: bool(grounding.ungrounded_entities(r, e)),
        "novelty_window": lambda r, e: bool(grounding.novel_content_windows(r, e)),
    }
    fired = {}
    for name, detector in standalone.items():
        rows = _rows_for(train, sources, detector)
        fired[name] = rows
        m = prf(**_score_counts(rows))
        print(f"  {name:18s} P={m['precision']*100:5.1f} R={m['recall']*100:5.1f} "
              f"F1={m['f1']*100:5.1f}")

    print("\n=== ensemble, vote>=1 (measured) ===")
    combined = []
    for i in range(len(fired["broadened_numeric"])):
        pred = any(fired[k][i]["pred"] for k in standalone)
        combined.append({"pred": pred, "gold": fired["broadened_numeric"][i]["gold"]})
    m = prf(**_score_counts(combined))
    print(f"  vote>=1  P={m['precision']*100:5.1f} R={m['recall']*100:5.1f} F1={m['f1']*100:5.1f}  "
          f"(vs naive F1={naive_f1*100:.1f}%)")
    return 0


def _score_counts(rows: list) -> dict:
    tp = sum(1 for x in rows if x["pred"] and x["gold"])
    fp = sum(1 for x in rows if x["pred"] and not x["gold"])
    fn = sum(1 for x in rows if not x["pred"] and x["gold"])
    return {"tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    raise SystemExit(main())
