"""Export RTX 3080 decode results as Astro tables and structured XML."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/nvidia-rtx-3080/throughput"
OUTPUT = ROOT / "benchmark_tables/nvidia-rtx-3080/throughput"
MODELS = (
    ("llama-3.2-3B-bench-results.json", "Llama 3.2 3B"),
    ("phi-3.5-mini-bench-results.json", "Phi-3.5 Mini"),
    ("qwen3-4B-bench-results.json", "Qwen3 4B"),
)
QUANTS = ("Q5_K_M", "Q6_K", "Q8_0")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xml = ET.Element("throughput", device="NVIDIA GeForce RTX 3080", unit="tokens/s")
    ET.SubElement(xml, "statistics").text = (
        "Reported avg_ts +/- stddev_ts from llama-bench; one standard deviation "
        "across repeated runs, not a confidence interval."
    )
    html = [
        '<div class="throughput-tables">',
        "  <p>NVIDIA GeForce RTX 3080. Throughput in tokens/s, mean &plusmn; one standard deviation.</p>",
    ]
    for filename, name in MODELS:
        with (SOURCE / filename).open(encoding="utf-8") as source:
            rows = [row for row in json.load(source) if row["n_gen"] > 0 and row["n_prompt"] == 0]
        lengths = sorted({row["n_gen"] for row in rows})
        indexed = {}
        for row in rows:
            quant = next(q for q in QUANTS if row["model_filename"].endswith(f"-{q}.gguf"))
            key = (quant, row["n_gen"])
            if key in indexed:
                raise ValueError(f"Duplicate decode result: {name} {key}")
            indexed[key] = row
        model = ET.SubElement(xml, "model", name=name, source=filename)
        html.extend([
            "  <section>",
            f"    <h2>{name}</h2>",
            '    <div class="table-scroll">',
            "      <table>",
            f"        <caption>{name}: generation throughput</caption>",
            '        <thead><tr><th scope="col">Quantization</th>',
        ])
        html.extend(f'          <th scope="col">{length:,} generated tokens</th>' for length in lengths)
        html.extend(['          <th scope="col">Runs per cell</th>', "        </tr></thead>", "        <tbody>"])
        for quant in QUANTS:
            cells = [indexed[quant, length] for length in lengths]
            counts = {len(row["samples_ts"]) for row in cells}
            if len(counts) != 1:
                raise ValueError(f"Inconsistent repetition counts: {name} {quant}")
            html.append(f'          <tr><th scope="row">{quant}</th>')
            variant = ET.SubElement(model, "quantization", name=quant)
            for row in cells:
                ET.SubElement(
                    variant, "measurement",
                    generated_tokens=str(row["n_gen"]),
                    prompt_tokens=str(row["n_prompt"]),
                    runs=str(len(row["samples_ts"])),
                    mean=str(row["avg_ts"]),
                    stddev=str(row["stddev_ts"]),
                )
                html.append(f'            <td>{row["avg_ts"]:.2f} &plusmn; {row["stddev_ts"]:.2f}</td>')
            html.extend([f"            <td>{counts.pop()}</td>", "          </tr>"])
        html.extend(["        </tbody>", "      </table>", "    </div>", "  </section>"])
    html.extend([
        "  <p>Decode-only benchmarks with zero prompt tokens. Values use the benchmark's reported mean and standard deviation across 100 repetitions per cell; &plusmn; is not a confidence interval.</p>",
        "</div>",
        "",
        "<style>",
        "  .throughput-tables { min-width: 0; }",
        "  section { margin-block: 2rem; }",
        "  h2 { font-size: 1.25rem; }",
        "  .table-scroll { max-width: 100%; overflow-x: auto; }",
        "  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }",
        "  caption { text-align: left; padding-block: 0.5rem; }",
        "  th, td { padding: 0.75rem; border-bottom: 1px solid currentColor; white-space: nowrap; text-align: right; }",
        "  th:first-child { text-align: left; }",
        "</style>",
    ])
    (OUTPUT / "ThroughputTables.astro").write_text("\n".join(html) + "\n", encoding="utf-8")
    ET.indent(xml)
    ET.ElementTree(xml).write(OUTPUT / "throughput.xml", encoding="utf-8", xml_declaration=True)
    print(f"Exported three model tables and 36 measurements to {OUTPUT}")


if __name__ == "__main__":
    main()
