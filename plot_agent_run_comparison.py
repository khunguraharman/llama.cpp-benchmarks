"""Plot prompt processing, request processing, and cache use across agent runs."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from plot_agent_run import (
    AGENT_RUN_PLOTS, LOG, MODEL_COMPARISON_OUTPUT, OUTPUT, ROOT, THROUGHPUT_WINDOW, parse_log, plt,
)
from plot_localai_style import (
    BACKGROUND, HEADING, MUTED, SERIES_COLORS, apply_dashboard_style,
    set_dashboard_title, style_dashboard_legend,
)


# Task IDs at the observed switches in these logs. Conversation IDs aren't
# logged; the 2,098-token retry stays with the preceding 2,070-token request.
# Review these boundaries when replacing the logs.
SWITCH_TASKS = {
    "RAM cache enabled": (6045, 29478, 29771, 31372, 33803),
    "RAM cache disabled": (6045, 29478, 29771, 31372, 32716),
}
QWEN35_LOG = LOG.with_name("llama-server-DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q6_K-no-mtp.log")
QWEN35_OUTPUT = AGENT_RUN_PLOTS / "qwen3.5 cache vs no cache"
# Match the conversation switches by prompt length across the two logs.
QWEN35_SWITCH_TASKS = {
    "RAM cache enabled": (1099, 4292, 4536, 4715, 5013),
    "RAM cache disabled": (1099, 4372, 4616, 4922, 5220),
}
M4_LOG = ROOT / "results/m4-mac-mini/agent_run" / LOG.name
M4_QWEN35_LOG = M4_LOG.with_name(QWEN35_LOG.name)
M4_QWEN35_SWITCH_TASKS = (1157, 4313, 4557, 4736, 5029)
M4_SWITCH_TASKS = {
    label: (5891, 29540, 29896, 30193, 31215) for label in SWITCH_TASKS
}
BENCHMARKS = {
    LOG: ("RTX 3080", SWITCH_TASKS),
    M4_LOG: ("M4 Mac mini", M4_SWITCH_TASKS),
}
VISIT_COUNT = 3
FILE_PREFIX = "qwen3-4b-q8_0"
MODEL_LABELS = {
    FILE_PREFIX: "Qwen3-4B-Q8_0",
    "qwen3.5-4b-q6_k": "Qwen3.5-4B Q6_K",
}


def split_conversations(requests, switches):
    blocks = [[]]
    boundary = 0
    for request in sorted(requests, key=lambda request: request.start):
        if boundary < len(switches) and request.task_id == switches[boundary]:
            blocks.append([])
            boundary += 1
        blocks[-1].append(request)
    if boundary != len(switches) or not all(blocks):
        raise ValueError("Conversation switch tasks do not match the benchmark log")
    return blocks[::2], blocks[1::2]


def summarize(group):
    return dict(
        requests=len(group),
        tool_calls=None,
        prompt_tokens_processed=sum(r.prefill[0] for r in group),
        prompt_processing_seconds=sum(r.prefill_seconds for r in group),
        output_tokens=sum(r.generation[0] for r in group),
        elapsed_seconds=max(r.end for r in group) - min(r.start for r in group),
        active_request_seconds=sum(r.end - r.start for r in group),
    )


def create_figure(rows, title, subtitle, styles, runs, prefix):
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows + 1))
    for ax in axes.flat:
        apply_dashboard_style(fig, ax)
    if title:
        fig.text(.08, .95, f"{MODEL_LABELS[prefix]} · {title}", fontsize=23,
                 fontfamily="Space Grotesk", fontweight="medium", color=HEADING)
        fig.text(.08, .902 if rows == 2 else .865, subtitle, fontsize=11, color=MUTED)
    handles = [Line2D([], [], color=color, linewidth=3, label=label)
               for label, color in styles.items()]
    if any(restores for _, _, restores in runs.values()):
        handles.append(Line2D([], [], color=MUTED, linestyle="--", linewidth=1,
                              label="Prompt state loaded from RAM"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.subplots_adjust(left=.08, right=.97, bottom=.20 if rows == 1 else .11,
                        top=.90 if not title else (.72 if rows == 1 else .81),
                        wspace=.28, hspace=.54)
    return fig, axes.flat


def save_figure(fig, output, name, *, prefix=FILE_PREFIX):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{prefix}-{name}.{extension}",
                    dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def format_timeline(ax, title, unit, runs, *, log=False):
    for requests, _, restores in runs.values():
        origin = requests[0].start
        for stamp in restores:
            ax.axvline(stamp - origin, color=MUTED, linestyle="--", linewidth=1, alpha=.85)
    set_dashboard_title(ax, title)
    ax.set_xlabel("Elapsed since first prompt (s)")
    ax.set_ylabel(unit)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)


def plot_totals(ax, records, styles, field, title, unit):
    if title:
        set_dashboard_title(ax, title)
    ax.set_ylabel(unit)
    width = .68 / len(styles)
    for index, (run, color) in enumerate(styles.items()):
        values = [record[field] for record in records if record["run"] == run]
        offset = (index - (len(styles) - 1) / 2) * width
        bars = ax.bar([x + offset for x in range(len(values))], values, width=.94 * width,
                      color=color)
        labels = [f"{value:,.2f}" if field.endswith("_seconds") else f"{value:,.0f}"
                  for value in values]
        ax.bar_label(bars, labels=labels, padding=5, color=HEADING)
    ax.set_xticks([0, 1], ["Conversation 1", "Conversation 2"])
    ax.set_xlim(-.55, 1.55)
    ax.set_ylim(0, max(record[field] for record in records) * 1.2 or 1)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}" if field.endswith("_seconds") else "{x:,.0f}"))


def plot_prompts(runs, records, styles, hardware, output, prefix, *, totals_only=False):
    if totals_only:
        fig, (total_tokens, total_time) = create_figure(1, "", "", styles, {}, prefix)
    else:
        fig, (evaluated, speed, total_tokens, total_time) = create_figure(
            2, "Prompt processing", f"{hardware}  /  Request metrics at prompt start", styles, runs, prefix)
        elapsed = max(requests[-1].start - requests[0].start for requests, _, _ in runs.values())
        width = max(elapsed / 270, .01)
        for index, (run, color) in enumerate(styles.items()):
            requests, _, _ = runs[run]
            starts = [request.start - requests[0].start for request in requests]
            positive = [(start, request.prefill[0]) for start, request in zip(starts, requests)
                        if request.prefill[0] > 0]
            if positive:
                times, tokens = zip(*positive)
                # The bar edge is the prompt start, with each cache setting on its own side.
                evaluated.bar(times, tokens, width=-width if index == 0 else width,
                              align="edge", color=color)
            speed.plot(starts, [request.prefill[1] for request in requests],
                       color=color, marker=".", linewidth=1.4)

        format_timeline(evaluated, "Newly evaluated prompt tokens per request", "Tokens (log scale)", runs, log=True)
        format_timeline(speed, "Prompt processing speed", "Tokens / s", runs)
    plot_totals(total_tokens, records, styles, "prompt_tokens_processed",
                "Total Prompt Tokens Processed" if totals_only else "Total prompt tokens processed", "Tokens")
    plot_totals(total_time, records, styles, "prompt_processing_seconds",
                "" if totals_only else "Total time processing prompts",
                "Prompt processing time (s)" if totals_only else "Seconds")
    save_figure(fig, output, "prompt-processing", prefix=prefix)


def plot_requests(runs, records, visits_by_conversation, styles, hardware, output, prefix):
    fig, (visits_ax, active_time, output_tokens, throughput) = create_figure(
        2, "Request processing",
        f"{hardware}  /  Request time excludes gaps between requests", styles, runs, prefix)
    set_dashboard_title(visits_ax, "Prompt tokens processed by visit")
    visits_ax.set_xlabel("Visit number")
    visits_ax.set_ylabel("Relative to Visit 0")
    visits_ax.set_xticks(range(VISIT_COUNT))
    visits_ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}×"))
    visits_ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    field = "prompt_tokens_processed"
    for index, (run, color) in enumerate(styles.items()):
        for conversation, marker in enumerate(("o", "D"), start=1):
            visits = visits_by_conversation[run, conversation]
            baseline = visits[0][field]
            ratios = [visit[field] / baseline if baseline else float("nan") for visit in visits]
            offset = ((index * 2 + conversation - 1) - (2 * len(styles) - 1) / 2) * .065
            visits_ax.scatter([visit + offset for visit in range(VISIT_COUNT)], ratios,
                              color=color, marker=marker, edgecolors=BACKGROUND,
                              linewidths=.8, s=52, zorder=3)
    visits_ax.set_xlim(-.25, VISIT_COUNT - 1 + .25)
    visits_ax.set_ylim(bottom=0)
    visits_ax.legend(handles=[Line2D([], [], marker=marker, linestyle="none", color=MUTED,
                                    label=f"Conversation {conversation}")
                              for conversation, marker in enumerate(("o", "D"), start=1)],
                     loc="best", frameon=False, fontsize=9)

    plot_totals(active_time, records, styles, "active_request_seconds",
                "Time spent processing requests", "Seconds")
    plot_totals(output_tokens, records, styles, "output_tokens", "Total output tokens", "Tokens")
    for index, (run, color) in enumerate(styles.items()):
        requests, _, _ = runs[run]
        origin = requests[0].start
        completed = sorted(requests, key=lambda request: request.end)
        rates = []
        for stop in range(1, len(completed) + 1):
            window = completed[max(0, stop - THROUGHPUT_WINDOW):stop]
            duration = sum(request.generation_seconds for request in window)
            rates.append(sum(request.generation[0] for request in window) / duration
                         if duration else float("nan"))
        throughput.plot([request.end - origin for request in completed], rates,
                        color=color, marker=".", linewidth=1.4,
                        linestyle="-" if index == 0 else "--")
    format_timeline(throughput, f"Generation throughput · latest {THROUGHPUT_WINDOW} requests",
                    "Tokens / s (weighted average)", runs)
    throughput.autoscale(axis="y")
    throughput.set_xlim(left=0)
    save_figure(fig, output, "request-processing", prefix=prefix)


def plot_cache(runs, styles, hardware, output, prefix):
    fig, (reused, cached) = create_figure(
        1, "Cache use", f"{hardware}  /  Input and reuse at prompt start, active slot at request end",
        styles, runs, prefix)
    for run, color in styles.items():
        requests, snapshots, _ = runs[run]
        origin = requests[0].start
        elapsed = max(request.end for request in requests) - origin
        reused.scatter([request.start - origin for request in requests],
                       [request.prompt_tokens for request in requests],
                       s=38, facecolors="none", edgecolors=color)
        restored = [request for request in requests if request.prompt_state_reused is not None]
        reused.scatter([request.start - origin for request in restored],
                       [request.prompt_state_reused for request in restored],
                       s=34, marker="x", color=color)
        initial = next((count for stamp, count in reversed(snapshots) if stamp <= origin), 0)
        times, counts = [0], [initial]
        for stamp, count in snapshots:
            if 0 < stamp - origin <= elapsed:
                times.append(stamp - origin)
                counts.append(count)
        cached.step(times + [elapsed], counts + [counts[-1]], where="post", color=color)
        cached.scatter([request.end - origin for request in requests],
                       [request.retained for request in requests], color=color, s=19)

    format_timeline(reused, "Prompt-state cache reuse and total input tokens", "Tokens", runs)
    format_timeline(cached, "Cached tokens", "Tokens", runs)
    style_dashboard_legend(reused, loc="upper left", fontsize=9, handles=[
        Line2D([], [], marker="o", markerfacecolor="none", linestyle="none", color=MUTED,
               label="Total input tokens"),
        Line2D([], [], marker="x", linestyle="none", color=MUTED,
               label="Tokens reused from RAM prompt state"),
    ])
    style_dashboard_legend(cached, loc="upper left", fontsize=9, handles=[
        Line2D([], [], drawstyle="steps-post", color=MUTED, label="Saved prompts in RAM"),
        Line2D([], [], marker=".", linestyle="none", color=MUTED, label="Active slot at request end"),
    ])
    save_figure(fig, output, "cache-use", prefix=prefix)


def plot_model_comparison(output, hardware, *, qwen3_log=LOG):
    # The logs omit conversation IDs. These task IDs follow the observed prompt changes.
    models = {
        "Qwen3.5-4B Q6_K": [(QWEN35_LOG, QWEN35_SWITCH_TASKS["RAM cache enabled"])],
        "Qwen3-4B Q8_0": list(zip(
            (LOG, LOG.with_name(f"{LOG.stem}-no-cache.log")), SWITCH_TASKS.values(), strict=True,
        )),
    }
    if qwen3_log == M4_LOG:
        models = {
            "Qwen3.5-4B Q6_K": [(M4_QWEN35_LOG, M4_QWEN35_SWITCH_TASKS)],
            "Qwen3-4B Q8_0": [(M4_LOG, M4_SWITCH_TASKS["RAM cache enabled"])],
        }
    styles = dict(zip(models, SERIES_COLORS))
    handles = [Line2D([], [], color=color, linewidth=3, label=model)
               for model, color in styles.items()]
    output.mkdir(parents=True, exist_ok=True)
    prefix = "qwen3.5-4b-vs-qwen3"

    def dashboard(title, subtitle):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax in axes.flat:
            apply_dashboard_style(fig, ax)
        fig.text(.08, .95, f"Qwen3.5-4B vs Qwen3-4B · {title}", fontsize=23, color=HEADING)
        fig.text(.08, .902, f"{hardware}  /  {subtitle}", fontsize=11, color=MUTED)
        fig.subplots_adjust(left=.08, right=.97, bottom=.13, top=.81, wspace=.28, hspace=.57)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
                   ncol=2, frameon=False)
        return fig, axes.flat

    totals_note = "" if qwen3_log == M4_LOG else "  /  Totals averaged per run"
    prompt_fig, (evaluated, speed, prompt_tokens, prompt_time) = dashboard(
        "Prompt processing", f"Request metrics at prompt start{totals_note}",
    )
    request_fig, (generated, active, output_tokens, throughput) = dashboard(
        "Request processing", f"Request time excludes gaps{totals_note}",
    )
    timeline_fig, timeline = plt.subplots(figsize=(12, 5.8))
    apply_dashboard_style(timeline_fig, timeline)
    timeline_fig.subplots_adjust(left=.095, right=.97, bottom=.15, top=.77)
    timeline_fig.text(.095, .91, "Generated tokens per request", fontsize=24, color=HEADING)
    timeline_fig.text(.095, .85, f"{hardware}  /  Plotted at request completion",
                      fontsize=11, color=MUTED)
    style_dashboard_legend(timeline, loc="upper left", fontsize=10, handles=handles)

    conversation_records = []
    for index, (model, logs) in enumerate(models.items()):
        color = styles[model]
        conversation_totals = [[], []]
        for run_index, (path, switches) in enumerate(logs):
            requests, _, _ = parse_log(path)
            origin = min(request.start for request in requests)
            for totals, visits in zip(conversation_totals, split_conversations(requests, switches)):
                totals.append(summarize([request for visit in visits for request in visit]))

            starts = [request.start - origin for request in requests]
            evaluated.bar(starts, [request.prefill[0] for request in requests],
                          width=(-1.2, 1.2)[index], align="edge", color=color, alpha=.75)
            linestyle = ("-", "--")[run_index]
            speed.plot(starts, [request.prefill[1] for request in requests],
                       color=color, marker=".", linewidth=1.4, linestyle=linestyle)
            completed = sorted(requests, key=lambda request: request.end)
            times = [request.end - origin for request in completed]
            counts = [request.generation[0] for request in completed]
            for ax in (generated, timeline):
                ax.scatter(times, counts, color=color, s=24, alpha=.8,
                           edgecolors=BACKGROUND, linewidths=.4, zorder=3)
            rates = []
            for stop in range(1, len(completed) + 1):
                window = completed[max(0, stop - THROUGHPUT_WINDOW):stop]
                duration = sum(request.generation_seconds for request in window)
                rates.append(sum(request.generation[0] for request in window) / duration
                             if duration else float("nan"))
            throughput.plot(times, rates, color=color, marker=".", linewidth=1.4,
                            linestyle=linestyle)

        for conversation, totals in enumerate(conversation_totals, start=1):
            conversation_records.append(dict(
                run=model,
                conversation=conversation,
                prompt_tokens_processed=mean(run["prompt_tokens_processed"] for run in totals),
                prompt_processing_seconds=mean(run["prompt_processing_seconds"] for run in totals),
                active_request_seconds=mean(run["active_request_seconds"] for run in totals),
                output_tokens=mean(run["output_tokens"] for run in totals),
            ))

    format_timeline(evaluated, "Newly evaluated prompt tokens per request", "Tokens (log scale)", {}, log=True)
    format_timeline(speed, "Prompt processing speed", "Tokens / s", {})
    plot_totals(prompt_tokens, conversation_records, styles, "prompt_tokens_processed",
                "Total prompt tokens processed", "Tokens")
    plot_totals(prompt_time, conversation_records, styles, "prompt_processing_seconds",
                "Total time processing prompts", "Seconds")
    format_timeline(generated, "Generated tokens per request", "Tokens (log scale)", {}, log=True)
    format_timeline(throughput, f"Generation throughput · latest {THROUGHPUT_WINDOW} requests",
                    "Tokens / s (weighted average)", {})
    conversation_opacities = (1, .4)
    request_fig.legend(handles=[
        Patch(facecolor=MUTED, alpha=opacity, label=f"Conversation {conversation}")
        for conversation, opacity in enumerate(conversation_opacities, start=1)
    ], loc="upper right", bbox_to_anchor=(.97, .92), ncol=2, frameon=False)
    for ax, field, title, unit, precision in (
        (active, "active_request_seconds", "Time spent processing requests", "Seconds", 2),
        (output_tokens, "output_tokens", "Total output tokens", "Tokens", 0),
    ):
        set_dashboard_title(ax, title)
        ax.set_ylabel(unit)
        positions = range(len(models))
        first, second = ([record[field] for record in conversation_records
                          if record["conversation"] == conversation] for conversation in (1, 2))
        lower = ax.bar(positions, first, width=.56, color=list(styles.values()),
                       alpha=conversation_opacities[0])
        upper = ax.bar(positions, second, bottom=first, width=.56, color=list(styles.values()),
                       alpha=conversation_opacities[1])
        lower_labels = ax.bar_label(lower, labels=[f"{value:,.{precision}f}" for value in first],
                                    label_type="center", fontsize=8, color=HEADING)
        lower_labels[0].set_color(BACKGROUND)
        ax.bar_label(upper, labels=[f"{value:,.{precision}f}" for value in second],
                     label_type="center", fontsize=8, color=HEADING)
        values = [left + right for left, right in zip(first, second)]
        ax.bar_label(upper, labels=[f"{value:,.{precision}f}" for value in values],
                     padding=5, color=HEADING)
        ax.set_xticks(positions, list(models))
        ax.set_ylim(0, max(values) * 1.2)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    timeline.set(xlabel="Elapsed since first prompt (s)", ylabel="Generated tokens per request",
                 xlim=(0, None), ylim=(0, None))
    timeline.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    for fig, name in ((prompt_fig, "prompt-processing"), (request_fig, "request-processing"),
                      (timeline_fig, "agent-runs")):
        save_figure(fig, output, name, prefix=prefix)


def model_prefix(log):
    return "qwen3.5-4b-q6_k" if log in (QWEN35_LOG, M4_QWEN35_LOG) else FILE_PREFIX


def default_output(log):
    if log == QWEN35_LOG:
        return QWEN35_OUTPUT
    if log == M4_QWEN35_LOG:
        return default_output(M4_LOG).with_name(QWEN35_OUTPUT.name)
    if log == M4_LOG:
        return ROOT / "benchmark_plots/m4-mac-mini/agent_run" / OUTPUT.name
    return OUTPUT


def plot_generation_timeline(log, output, hardware, *, prefix=FILE_PREFIX):
    requests, _, _ = parse_log(log)
    origin = min(request.start for request in requests)
    times = [request.end - origin for request in requests]
    counts = [request.generation[0] for request in requests]
    completion_gaps = [right - left for left, right in zip(times, times[1:]) if right > left]
    width = min(max(times) / 160, .8 * min(completion_gaps, default=float("inf"))) or .5

    fig, ax = plt.subplots(figsize=(14, 5))
    apply_dashboard_style(fig, ax)
    fig.text(.08, .90, f"{MODEL_LABELS[prefix]} · Tokens generated per request", fontsize=23,
             fontfamily="Space Grotesk", fontweight="medium", color=HEADING)
    fig.text(.08, .84, f"{hardware}  /  {len(requests)} requests  /  Plotted at request completion",
             fontsize=11, color=MUTED)
    fig.subplots_adjust(left=.08, right=.97, bottom=.17, top=.76)
    ax.bar(times, counts, width=width, color=SERIES_COLORS[0], zorder=3)
    ax.set_xlabel("Elapsed since first prompt (s)")
    ax.set_ylabel("Generated tokens per request")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, "tokens-generated-vs-time", prefix=prefix)


def plot_comparison(log, output, hardware, switches, cached_only, *, prefix=FILE_PREFIX):
    logs = {"RAM cache enabled": log}
    if not cached_only:
        logs["RAM cache disabled"] = log.with_name(f"{log.stem}-no-cache.log")
    styles = dict(zip(logs, SERIES_COLORS))
    records = []
    visits_by_conversation = {}
    runs = {}
    for label, path in logs.items():
        requests, snapshots, restores = parse_log(path)
        requests.sort(key=lambda request: request.start)
        runs[label] = requests, snapshots, restores
        for conversation, visits in enumerate(split_conversations(requests, switches[label]), start=1):
            if len(visits) != VISIT_COUNT:
                raise ValueError(f"Expected three visits for {label}, conversation {conversation}")
            visits_by_conversation[label, conversation] = [summarize(visit) for visit in visits]
            records.append(dict(run=label, conversation=conversation,
                                **summarize([request for visit in visits for request in visit])))

    output.mkdir(parents=True, exist_ok=True)
    plot_prompts(runs, records, styles, hardware, output, prefix,
                 totals_only=log in BENCHMARKS and not cached_only)
    plot_requests(runs, records, visits_by_conversation, styles, hardware, output, prefix)
    plot_cache(runs, styles, hardware, output, prefix)
    with (output / f"{prefix}-run-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"{hardware}: wrote three plot sets to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Cached log path")
    parser.add_argument("--log", type=Path, help="Cached log path")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--hardware", help="Hardware label in the plot subtitle")
    parser.add_argument("--cached-switches", type=int, nargs=5, help="Cached log conversation switch task IDs")
    parser.add_argument("--no-cache-switches", type=int, nargs=5, help="No-cache log conversation switch task IDs")
    parser.add_argument("--cached-only", action="store_true", help="Plot only the cached series")
    parser.add_argument("--generation-timeline", action="store_true",
                        help="Plot generated tokens per request over time for the selected run")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare Qwen3 and Qwen3.5 for the selected benchmark")
    args = parser.parse_args()
    if args.path and args.log:
        parser.error("Supply the log path once")

    for font in (ROOT / "benchmark_plots/fonts").glob("*-static.ttf"):
        font_manager.fontManager.addfont(str(font))
    selected = args.log or args.path
    benchmarks = {
        **BENCHMARKS,
        QWEN35_LOG: ("RTX 3080", QWEN35_SWITCH_TASKS),
        M4_QWEN35_LOG: ("M4 Mac mini", dict.fromkeys(SWITCH_TASKS, M4_QWEN35_SWITCH_TASKS)),
    }
    if args.compare_models:
        log = selected.resolve() if selected else LOG
        if log not in BENCHMARKS or args.generation_timeline or args.cached_only or args.cached_switches or args.no_cache_switches:
            parser.error("--compare-models uses a built-in Qwen3 benchmark log")
        model_output = default_output(log).with_name(MODEL_COMPARISON_OUTPUT.name)
        plot_model_comparison(args.output or model_output, args.hardware or BENCHMARKS[log][0],
                              qwen3_log=log)
        return
    if args.generation_timeline:
        log = selected.resolve() if selected else LOG
        hardware = args.hardware or benchmarks.get(log, ("", {}))[0]
        output = args.output or default_output(log)
        plot_generation_timeline(log, output, hardware, prefix=model_prefix(log))
        return
    if selected is None and (args.output or args.hardware or args.cached_switches or args.no_cache_switches):
        selected = LOG
    for log in (selected.resolve(),) if selected else BENCHMARKS:
        hardware, defaults = benchmarks.get(log, ("", {}))
        switches = defaults.copy()
        if args.cached_switches is not None:
            switches["RAM cache enabled"] = args.cached_switches
        if args.no_cache_switches is not None:
            switches["RAM cache disabled"] = args.no_cache_switches
        if "RAM cache enabled" not in switches:
            parser.error("A custom log requires --cached-switches")
        if not args.cached_only and "RAM cache disabled" not in switches:
            parser.error("A custom no-cache log requires --no-cache-switches")
        output = args.output or default_output(log)
        plot_comparison(log, output, args.hardware or hardware, switches, args.cached_only,
                        prefix=model_prefix(log))


if __name__ == "__main__":
    main()
