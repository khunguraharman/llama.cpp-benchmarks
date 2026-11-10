"""Plot elapsed-time timelines from a llama-server agent run."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

from plot_localai_style import (
    BACKGROUND,
    HEADING,
    MUTED,
    SERIES_COLORS,
    apply_dashboard_style,
    set_dashboard_title,
    style_dashboard_legend,
)


ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results/nvidia-rtx-3080/agent_run/llama-server-Qwen3-4B-Q8_0.log"
OUTPUT = ROOT / "benchmark_plots/nvidia-rtx-3080/agent_run"
THROUGHPUT_WINDOW = 10
STAMP = r"\d+\.\d{2}\.\d{3}\.\d{3}"
SLOT_LINE = re.compile(
    rf"^({STAMP}) \w slot\s+.*?: id\s+(\d+) \| task (\d+) \|\s+(.*)$"
)
CACHE_STATE = re.compile(rf"^({STAMP}).*? - cache state: (\d+) prompts")
CACHE_ENTRY = re.compile(r" - prompt [^:]+:\s+(\d+) tokens, checkpoints:")
TIMING = re.compile(
    r"(prompt eval|eval) time\s*=\s*(?P<duration_ms>[\d.]+) ms\s*/\s*(?P<tokens>\d+) tokens "
    r"\([^,]+,\s*(?P<rate>[\d.]+) tokens per second\)"
)


@dataclass
class Request:
    start: float
    prompt_tokens: int
    reused: int | None = None
    prefill: tuple[int, float] | None = None
    generation: tuple[int, float] | None = None
    end: float = 0
    retained: int = 0
    prefill_seconds: float = 0


def seconds(stamp):
    minutes, seconds, milliseconds, microseconds = map(int, stamp.split("."))
    return 60 * minutes + seconds + milliseconds / 1_000 + microseconds / 1_000_000


def parse_log(path):
    requests = []
    pending = {}
    saved_cache = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if state := CACHE_STATE.match(line):
            count = int(state[2])
            entries = [
                CACHE_ENTRY.search(entry) for entry in lines[index + 1:index + 1 + count]
            ]
            if len(entries) != count or any(entry is None for entry in entries):
                raise ValueError(f"Incomplete prompt cache snapshot on line {index + 1}")
            saved_cache.append((seconds(state[1]), sum(int(entry[1]) for entry in entries)))

        if not (match := SLOT_LINE.match(line)):
            continue
        stamp, slot, task, event = match.groups()
        key = (slot, task)
        if event.startswith("new prompt,"):
            pending[key] = Request(seconds(stamp), int(event.rsplit("task.n_tokens = ", 1)[1]))
        elif key in pending:
            request = pending[key]
            if timing := TIMING.match(event):
                tokens, rate = int(timing["tokens"]), float(timing["rate"])
                if timing[1] == "prompt eval":
                    request.prefill = (tokens, rate)
                    request.prefill_seconds = float(timing["duration_ms"]) / 1_000
                else:
                    request.generation = (tokens, rate)
            elif event.startswith("stop processing: n_tokens = "):
                if request.prefill is None or request.generation is None:
                    raise ValueError(f"Missing metrics for task {task} in slot {slot}")
                # Batch-progress cache messages include newly evaluated tokens.
                request.reused = request.prompt_tokens - request.prefill[0]
                if not 0 <= request.reused <= request.prompt_tokens:
                    raise ValueError(f"Prompt token accounting mismatch for task {task} in slot {slot}")
                request.end = seconds(stamp)
                request.retained = int(event.split(",")[0].split(" = ")[1])
                requests.append(pending.pop(key))

    if not requests:
        raise ValueError(f"No completed requests in {path}")
    return requests, saved_cache


def plot_run(requests, saved_cache, output, run_label=""):
    origin = min(request.start for request in requests)
    elapsed = max(request.end for request in requests) - origin
    starts = [request.start - origin for request in requests]
    ends = [request.end - origin for request in requests]

    for font in (ROOT / "benchmark_plots/fonts").glob("*-static.ttf"):
        font_manager.fontManager.addfont(str(font))
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    generated, throughput, reused, evaluated, processing, cached = axes.flat
    for ax in fig.axes:
        apply_dashboard_style(fig, ax)
    green, terracotta = SERIES_COLORS[:2]

    generated_tokens = [request.generation[0] for request in requests]
    running_total = sum(generated_tokens)
    generated.bar(ends, generated_tokens, width=1.2, color=green)
    generated.set_yscale("log")
    generated.margins(y=.15)

    reused.scatter(starts, [request.prompt_tokens for request in requests], s=42,
                   facecolors="none", edgecolors=terracotta, label="Total input tokens")
    reused.scatter(starts, [request.reused for request in requests], s=14,
                   color=green, label="Reused cached tokens")
    evaluated.bar(starts, [request.prefill[0] for request in requests], width=1.2,
                  color=terracotta)
    evaluated.set_yscale("log")
    evaluated.margins(y=.15)

    rates = [request.generation[1] for request in requests]
    average_rates, floor_95_rates, floor_99_rates = [], [], []
    for stop in range(1, len(rates) + 1):
        window = rates[max(0, stop - THROUGHPUT_WINDOW):stop]
        floor_95, floor_99 = np.percentile(window, [5, 1])
        average_rates.append(np.mean(window))
        floor_95_rates.append(floor_95)
        floor_99_rates.append(floor_99)
    throughput.plot(ends, average_rates, color=green, linewidth=1.6, label="Average")
    throughput.plot(ends, floor_95_rates, color=terracotta, linewidth=1.6, label="p5")
    throughput.plot(ends, floor_99_rates, color=SERIES_COLORS[2], linewidth=1.6,
                    linestyle="--", label="p1")

    prefill_times, prefill_rates = zip(*sorted(
        (start, request.prefill[1]) for start, request in zip(starts, requests)
    ))
    processing.plot(prefill_times, prefill_rates, color=green, linewidth=1.5)

    if saved_cache:
        last_saved = next((count for stamp, count in reversed(saved_cache) if stamp <= origin), 0)
        cache_times = [0]
        cache_counts = [last_saved]
        for stamp, count in saved_cache:
            if 0 < stamp - origin <= elapsed:
                cache_times.append(stamp - origin)
                cache_counts.append(count)
        cached.step(cache_times + [elapsed], cache_counts + [cache_counts[-1]], where="post",
                    color=terracotta, label="Saved prompts in RAM")
    else:
        cached.plot([0, elapsed], [0, 0], color=terracotta, label="Saved prompts in RAM (disabled)")
    cached.scatter(ends, [request.retained for request in requests],
                   color=green, s=22, label="Active slot at request end")

    generated.set_xlim(-elapsed * .015, elapsed * 1.025)
    generated.xaxis.set_major_locator(MultipleLocator(30))
    generated.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    for ax, title, unit in (
        (generated, "Generated tokens per request", "Tokens (log scale)"),
        (throughput, f"Generation throughput · latest {THROUGHPUT_WINDOW} requests", "Tokens / s"),
        (reused, "Reused cache and total input tokens per request", "Tokens"),
        (evaluated, "Newly evaluated prompt tokens per request", "Tokens (log scale)"),
        (processing, "Prompt processing speed", "Tokens / s"),
        (cached, "Cached tokens", "Tokens"),
    ):
        set_dashboard_title(ax, title)
        ax.set_xlabel("Elapsed since first prompt (s)")
        ax.set_ylabel(unit)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        if ax.get_yscale() == "linear":
            ax.set_ylim(bottom=0)
        ax.tick_params(labelbottom=True)

    # Keep the speed changes visible on a tighter scale than the token counts.
    throughput.autoscale(axis="y")
    style_dashboard_legend(reused, loc="upper left", ncol=2)
    style_dashboard_legend(throughput, loc="lower left", ncol=3)
    style_dashboard_legend(cached, loc="upper left")

    fig.text(.075, .965, f"Qwen3-4B-Q8_0 agent run{run_label}", fontsize=23, fontweight="medium",
             fontfamily="Space Grotesk", color=HEADING)
    fig.text(.075, .933, f"RTX 3080  /  {len(requests)} requests  /  {running_total:,} generated tokens",
             fontsize=11, color=MUTED)
    fig.text(.075, .025,
             "Prompt and reuse metrics are placed at request start. Final generation and active-slot counts "
             "are placed at request end.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.075, right=.98, bottom=.10, top=.873, wspace=.26, hspace=.58)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"{len(requests)} requests, {elapsed:.3f} seconds, {running_total:,} generated tokens")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path,
                        help="Log to plot; defaults to both cached and no-cache runs")
    parser.add_argument("--output", type=Path, help="Output image for a single supplied log")
    args = parser.parse_args()
    if args.output and args.log is None:
        parser.error("--output requires a log path")
    logs = [args.log] if args.log else [LOG, LOG.with_name(f"{LOG.stem}-no-cache.log")]
    for log in logs:
        no_cache = log.stem.endswith("-no-cache")
        suffix = "-no-cache" if no_cache else ""
        output = args.output or OUTPUT / f"qwen3-4b-q8_0{suffix}-timelines.png"
        requests, saved_cache = parse_log(log)
        plot_run(requests, saved_cache, output, " · no cache" if no_cache else " · cache enabled")


if __name__ == "__main__":
    main()
