"""Export token-weighted MTP generation throughput and population SD tables."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results/nvidia-rtx-3080/mtp-speed-bench-results"
PATTERN = re.compile(
    r"mtp_n(?P<n>\d+)_(?P<bench>throughput_[12]k)_"
    r"(?P<entropy>high_entropy|low_entropy)_osl(?P<osl>\d+)(?:_run\d+)?\.json"
)


def summarize(input_dir):
    groups = defaultdict(list)
    for path in sorted(input_dir.glob("mtp_n*.json")):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        n, bench, entropy, osl = (
            int(match["n"]), match["bench"], match["entropy"], int(match["osl"])
        )
        config = data["config"]
        assert (config["bench"], config["category"], config["osl"]) == (
            bench, entropy, osl
        ), f"Filename/config mismatch: {path}"
        for run in data["results"]:
            if not run.get("ok"):
                raise ValueError(f"Failed run in {path}")
            tokens = run["completion_tokens"]
            speed = run["predicted_per_second"]
            if tokens <= 0 or not math.isfinite(speed) or speed <= 0:
                raise ValueError(f"Invalid tokens or generation speed in {path}")
            assert run["category"] == entropy, f"Entropy mismatch: {path}"
            groups[n, bench, entropy, osl].append((tokens, speed))

    rows = []
    for (n, bench, entropy, osl), runs in sorted(groups.items()):
        total = sum(tokens for tokens, speed in runs)
        mean = math.fsum(tokens * speed for tokens, speed in runs) / total
        sd = math.sqrt(
            math.fsum(tokens * (speed - mean) ** 2 for tokens, speed in runs) / total
        )
        rows.append(dict(n_predict=n, bench=bench, entropy=entropy, osl=osl,
                         runs=len(runs), tokens=total, mean=mean, sd=sd))
    if not rows:
        raise ValueError(f"No matching results in {input_dir}")
    return rows


def render(rows, input_dir):
    lines = [
        "# RTX 3080: token-weighted generation throughput",
        "",
        f"Source: `{input_dir.resolve()}`",
        "",
        "Each cell is weighted mean ± one weighted population standard deviation, in tokens/s.",
        "Weights are `completion_tokens`; speed is `predicted_per_second` (server generation speed).",
        "`n_predict` below refers to the filename's MTP `n` / `runs.csv`'s `mtp_n_max` (0–5).",
        "OSL is the configured output sequence length.",
        "",
        "Weighted mean: `μ = Σ(tokens_i × speed_i) / Σ(tokens_i)`.",
        "Weighted population SD: `σ = sqrt(Σ(tokens_i × (speed_i − μ)²) / Σ(tokens_i))`.",
        "This SD describes the supplied runs; it is not a confidence interval or standard error.",
        "",
        f"Configurations: {len(rows)}. Included runs: {sum(r['runs'] for r in rows)}.",
    ]
    if all(row["runs"] == 1 for row in rows):
        lines.extend([
            "Every configuration has exactly one run, so its weighted mean equals that run's speed",
            "and its weighted population SD is 0.00. Run-to-run variability cannot be estimated",
            "from one observation; zero here does not imply reproducibility.",
        ])
    for bench in ("throughput_1k", "throughput_2k"):
        for entropy in ("high_entropy", "low_entropy"):
            subset = [r for r in rows if r["bench"] == bench and r["entropy"] == entropy]
            if not subset:
                continue
            lengths = sorted({r["osl"] for r in subset})
            indexed = {(r["n_predict"], r["osl"]): r for r in subset}
            lines.extend([
                "", f"## {bench} — {entropy.replace('_', ' ')}", "",
                "| n_predict | " + " | ".join(f"OSL {osl}" for osl in lengths) + " |",
                "|---:|" + "---:|" * len(lengths),
            ])
            for n in sorted({r["n_predict"] for r in subset}):
                cells = []
                for osl in lengths:
                    row = indexed.get((n, osl))
                    cells.append(f"{row['mean']:.2f} ± {row['sd']:.2f}" if row else "—")
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "benchmark_tables/nvidia-rtx-3080/mtp-speed-bench-results/weighted_throughput.md")
    args = parser.parse_args()
    rows = summarize(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, args.input_dir), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} configurations to {args.output.resolve()}")
    print(f"Saved CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
