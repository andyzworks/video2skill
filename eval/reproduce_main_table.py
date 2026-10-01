#!/usr/bin/env python3
"""Rescore the released model outputs and print Table 1 of the paper.

Usage:
    python reproduce_main_table.py --data /path/to/video2skill-bench
where --data is a local copy of https://huggingface.co/datasets/Sterzhang/video2skill-bench
(only test/*.jsonl is needed). Without --data, the test files are fetched with huggingface_hub.
"""

import argparse
from pathlib import Path

from score import read_jsonl, score

HERE = Path(__file__).resolve().parent
MODELS = [
    "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B",
    "Qwen3-VL-8B", "Qwen3.8-27B", "InternVL3.5-4B", "InternVL3.5-8B", "InternVL3.5-14B",
    "InternVL3.5-38B", "Ovis2.5-9B", "Cosmos-Reason2-8B", "Cosmos-Reason2-32B",
    "LLaVA-OneVision-2-8B", "GLM-4.1V-9B", "MiniCPM-V-4.5", "Gemma-4-31B",
]
DOMAINS = ["robointer", "hdepic"]
METRICS = ["Cov.", "Pair P", "Pair R", "ARI", "Create R", "Reuse R"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="local copy of the Hugging Face dataset")
    args = parser.parse_args()
    if args.data is None:
        from huggingface_hub import snapshot_download
        args.data = Path(snapshot_download("Sterzhang/video2skill-bench", repo_type="dataset", allow_patterns=["test/*"]))
    tests = {d: read_jsonl(args.data / "test" / f"{d}.jsonl") for d in DOMAINS}

    width = 22
    print(" " * width + "RoboInter".center(6 * 9) + "HD-EPIC".center(6 * 9))
    print("Model".ljust(width) + "".join(m.rjust(9) for m in METRICS * 2))
    for paradigm in ["unified", "factorized"]:
        print(f"-- {paradigm} --")
        for model in MODELS:
            cells = []
            for d in DOMAINS:
                preds = {row["clip_id"]: row["events"] for row in read_jsonl(HERE / "predictions" / paradigm / d / f"{model}.jsonl")}
                result = score(tests[d], preds)
                cells += [f"{100 * result[m]:.1f}" for m in METRICS]
            print(model.ljust(width) + "".join(c.rjust(9) for c in cells))


if __name__ == "__main__":
    main()
