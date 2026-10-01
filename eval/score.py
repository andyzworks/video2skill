#!/usr/bin/env python3
"""Score SESD predictions on the Video2Skill benchmark (Section 2.3 of the paper).

Usage:
    python score.py --test test/hdepic.jsonl --pred my_model_hdepic.jsonl

Prediction file: one JSON object per test clip,
    {"clip_id": "...", "events": [{"start": 3.0, "end": 5.0, "skill": "grasp(object, source)"}, ...]}
Times are seconds from the start of the clip. "skill" is the model's own schema name; only
the part before "(" is used, lower-cased, so names are free-form and scoring is
naming-invariant. Clips are scored in the order of the test file, which is the order the
model must process them in: the skill library carries over from clip to clip.

Metrics (all x100 in the paper):
    Cov.      share of reference events matched by a prediction (best tIoU >= 0.3)
    Pair P    of matched event pairs the model puts in one skill, share in the same reference class
    Pair R    of matched event pairs in the same reference class, share the model puts in one skill
    ARI       adjusted Rand index between the model's skills and the reference classes
    Create R  first occurrences of a reference class that get a previously unused skill name
    Reuse R   later occurrences of a reference class that get a previously used skill name
Pair P/R, ARI and Create/Reuse R are computed over matched events only.
"""

import argparse
import json
import sys
from collections import Counter

MIN_IOU = 0.3


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def skill_head(name):
    """`grasp(object, source)` -> `grasp`."""
    return str(name or "").strip().lower().split("(", 1)[0].strip()


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def align(test, predictions):
    """Match each reference event to its highest-tIoU prediction in the same clip.

    Several reference events may share one prediction; unmatched predictions are not
    penalized. Returns (reference class, model skill) pairs in benchmark order.
    """
    matched, total = [], 0
    for clip in test:
        preds = []
        for p in predictions.get(clip["clip_id"], []):
            if p.get("start") is None or p.get("end") is None or not p.get("skill"):
                continue
            preds.append((float(p["start"]), float(p["end"]), skill_head(p["skill"])))
        for ref in clip["events"]:
            total += 1
            span = (float(ref["start"]), float(ref["end"]))
            best, best_iou = None, 0.0
            for p in preds:
                value = tiou(span, p)
                if value > best_iou:
                    best, best_iou = p, value
            if best is not None and best_iou >= MIN_IOU:
                matched.append((ref["skill"], best[2]))
    return matched, total


def pair_scores(matched):
    pairs = lambda n: n * (n - 1) // 2
    both = sum(pairs(n) for n in Counter(matched).values())
    ref_same = sum(pairs(n) for n in Counter(r for r, _ in matched).values())
    model_same = sum(pairs(n) for n in Counter(m for _, m in matched).values())
    all_pairs = pairs(len(matched))
    precision = both / model_same if model_same else 0.0
    recall = both / ref_same if ref_same else 0.0
    expected = ref_same * model_same / all_pairs if all_pairs else 0.0
    maximum = (ref_same + model_same) / 2
    ari = (both - expected) / (maximum - expected) if maximum != expected else 0.0
    return precision, recall, ari


def create_reuse(matched):
    seen_ref, seen_model = set(), set()
    hits = Counter()
    for ref, model in matched:
        key = "reuse" if ref in seen_ref else "create"
        hits[key, "total"] += 1
        if (model in seen_model) == (key == "reuse"):
            hits[key, "hit"] += 1
        seen_ref.add(ref)
        seen_model.add(model)
    rate = lambda k: hits[k, "hit"] / hits[k, "total"] if hits[k, "total"] else 0.0
    return rate("create"), rate("reuse")


def score(test, predictions):
    matched, total = align(test, predictions)
    precision, recall, ari = pair_scores(matched)
    create, reuse = create_reuse(matched)
    raw = {"Cov.": len(matched) / total if total else 0.0, "Pair P": precision, "Pair R": recall,
           "ARI": ari, "Create R": create, "Reuse R": reuse}
    # Scores are kept to 4 decimals, then shown x100 with 1 decimal, as in the paper's tables.
    return {k: round(v, 4) for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, help="test/hdepic.jsonl or test/robointer.jsonl")
    parser.add_argument("--pred", required=True, help="predictions, one line per test clip")
    parser.add_argument("--json", action="store_true", help="print raw scores as JSON")
    args = parser.parse_args()

    test = read_jsonl(args.test)
    predictions = {}
    for row in read_jsonl(args.pred):
        predictions[row["clip_id"]] = row.get("events") or []
    missing = [c["clip_id"] for c in test if c["clip_id"] not in predictions]
    if missing:
        print(f"warning: {len(missing)} test clips have no prediction line and are scored as empty", file=sys.stderr)

    result = score(test, predictions)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("  ".join(f"{k} {100 * v:.1f}" for k, v in result.items()))


if __name__ == "__main__":
    main()
