"""Parse llama-server agent runs and generate comparison figures."""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
RAM_RESTORE = re.compile(rf"^({STAMP}).*? - found better prompt with ")
RETAINED_PREFIX = re.compile(r"cached n_tokens = (\d+), memory_seq_rm ")
TIMING = re.compile(
    r"(prompt eval|eval) time\s*=\s*(?P<duration_ms>[\d.]+) ms\s*/\s*(?P<tokens>\d+) tokens "
    r"\([^,]+,\s*(?P<rate>[\d.]+) tokens per second\)"
)


@dataclass
class Request:
    start: float
    prompt_tokens: int
    prompt_state_restored_at: float | None = None
    prompt_state_reused: int | None = None
    prefill: tuple[int, float] | None = None
    generation: tuple[int, float] | None = None
    end: float = 0
    retained: int = 0
    prefill_seconds: float = 0
    task_id: int | None = None
    generation_seconds: float = 0


def seconds(stamp):
    minutes, seconds, milliseconds, microseconds = map(int, stamp.split("."))
    return 60 * minutes + seconds + milliseconds / 1_000 + microseconds / 1_000_000


def parse_log(path):
    requests = []
    pending = {}
    saved_cache = []
    pending_restore = None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if restore := RAM_RESTORE.match(line):
            pending_restore = seconds(restore[1])
        elif "failed to restore state" in line or " - looking for better prompt," in line:
            pending_restore = None
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
            pending[key] = Request(seconds(stamp), int(event.rsplit("task.n_tokens = ", 1)[1]),
                                   prompt_state_restored_at=pending_restore,
                                   task_id=int(task))
            pending_restore = None
        elif key in pending:
            request = pending[key]
            # The first prefix count follows prompt trimming. Later batches
            # include tokens evaluated during this request.
            if request.prompt_state_restored_at is not None and request.prompt_state_reused is None:
                if prefix := RETAINED_PREFIX.match(event):
                    request.prompt_state_reused = int(prefix[1])
            if timing := TIMING.match(event):
                tokens, rate = int(timing["tokens"]), float(timing["rate"])
                if timing[1] == "prompt eval":
                    request.prefill = (tokens, rate)
                    request.prefill_seconds = float(timing["duration_ms"]) / 1_000
                else:
                    request.generation = (tokens, rate)
                    request.generation_seconds = float(timing["duration_ms"]) / 1_000
            elif event.startswith("stop processing: n_tokens = "):
                if request.prefill is None or request.generation is None:
                    raise ValueError(f"Missing metrics for task {task} in slot {slot}")
                if not 0 <= request.prefill[0] <= request.prompt_tokens:
                    raise ValueError(f"Prompt token accounting mismatch for task {task} in slot {slot}")
                request.end = seconds(stamp)
                request.retained = int(event.split(",")[0].split(" = ")[1])
                requests.append(pending.pop(key))

    if not requests:
        raise ValueError(f"No completed requests in {path}")
    ram_restores = [request.prompt_state_restored_at for request in requests
                    if request.prompt_state_restored_at is not None]
    return requests, saved_cache, ram_restores


if __name__ == "__main__":
    from plot_agent_run_comparison import main

    main()
