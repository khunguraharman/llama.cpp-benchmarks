"""Summarize peak GPU memory across speculative benchmark runs.

Each point is the maximum logged sample across the entire benchmark file.
Allocated VRAM is reported in MiB; prompt length and entropy are never pooled.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results/nvidia-rtx-3080/mtp-speed-bench-results"
NAME = re.compile(r"(?:mtp|eagle3)_n(\d+)_(throughput_\d+k)_(high_entropy|low_entropy)_osl(\d+)\.gpu\.csv")


def load_peaks(input_dir):
    rows = []
    for path in sorted(input_dir.glob("*.gpu.csv")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        n, bench, entropy, osl = match.groups()
        samples = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for raw in csv.DictReader(stream, skipinitialspace=True):
                sample = {}
                for key in ("memory_used_mib",):
                    try:
                        value = float(raw[key])
                        if math.isfinite(value):
                            sample[key] = value
                    except (KeyError, TypeError, ValueError):
                        pass
                samples.append(sample)
        row = dict(n_predict=int(n), bench=bench, entropy=entropy, osl=int(osl), source=path.name)
        for source, target in (("memory_used_mib", "peak_memory_used_mib"),):
            values = [s[source] for s in samples if source in s]
            if not values:
                raise ValueError(f"No valid {source} samples in {path}")
            row[target] = max(values)
        rows.append(row)
    if not rows:
        raise ValueError(f"No matching GPU logs found in {input_dir}")
    return rows


def memory_components(spec):
    """Estimate draft tensor payload and allocated dense KV capacity, in MiB."""
    kv_bytes = (spec["sequences"] * spec["context_tokens"] * spec["draft_layers"]
                * spec["kv_heads"] * (spec["key_head_dim"] * spec["key_bytes"]
                                      + spec["value_head_dim"] * spec["value_bytes"]))
    return spec["weight_tensor_bytes"] / 2**20, kv_bytes / 2**20


def write_memory_summaries(rows, output_dir, memory_spec=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "peak_memory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if memory_spec is not None:
        weights, kv = memory_components(memory_spec)
        breakdown = []
        indexed = {(r["bench"], r["entropy"], r["osl"], r["n_predict"]): r for r in rows}
        for row in rows:
            if row["n_predict"] != 1:
                continue
            baseline = indexed[row["bench"], row["entropy"], row["osl"], 0]["peak_memory_used_mib"]
            jump = row["peak_memory_used_mib"] - baseline
            breakdown.append(dict(bench=row["bench"], entropy=row["entropy"], osl=row["osl"],
                baseline_peak_mib=baseline, n1_peak_mib=row["peak_memory_used_mib"],
                measured_jump_mib=jump, estimated_draft_weights_mib=weights,
                estimated_draft_kv_mib=kv, unattributed_difference_mib=jump-weights-kv))
        with (output_dir / "eagle3_memory_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
            writer.writeheader()
            writer.writerows(breakdown)
        (output_dir / "eagle3_memory_methodology.json").write_text(
            json.dumps(dict(spec=memory_spec,
                kv_formula="sequences * context_tokens * draft_layers * kv_heads * (key_head_dim * key_bytes + value_head_dim * value_bytes)",
                estimated_draft_weights_mib=weights, estimated_draft_kv_mib=kv,
                residual="Difference of device-wide peaks minus estimates; not a measured buffer allocation. Includes changed compute/feature buffers, allocator effects and other device usage; exact attribution unavailable.",
                breakdown=breakdown), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_plots/nvidia-rtx-3080/mtp/mtp-speed-bench-results")
    args = parser.parse_args()
    rows = load_peaks(args.input_dir)
    write_memory_summaries(rows, args.output_dir)
    print(f"Read {len(rows)} benchmark GPU logs.")


if __name__ == "__main__":
    main()
