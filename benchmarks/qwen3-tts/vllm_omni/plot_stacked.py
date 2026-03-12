"""Plot stacked optimization benchmark results for Qwen3-TTS.

Generates figures matching the Qwen3-Omni blog style:
  - Pairwise bar charts (before vs after each optimization step)
  - Summary line plots (all steps, one line per concurrency)
  - vLLM-Omni vs HF Transformers comparison

Color scheme (matching Omni blog):
  Bar charts: dodger blue (#1E90FF) = before, lime green (#32CD32) = after
  Summary lines: blue (conc=1), green (conc=4), orange (conc=10)

Usage:
    python plot_stacked.py \
        --baseline results/bench_baseline_*.json \
        --batch results/bench_batch_*.json \
        --cuda-graph results/bench_cuda_graph_*.json \
        --async-chunk results/bench_async_chunk_*.json \
        --hf results/bench_hf_transformers_*.json \
        --output-dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Omni blog color scheme ----
COLOR_BEFORE = "#1E90FF"  # dodger blue
COLOR_AFTER = "#32CD32"  # lime green
COLOR_HF = "#FF6347"  # tomato red (for HF transformers)
COLOR_VLLM = "#1E90FF"  # dodger blue (for vLLM-omni)

# Summary line plot colors per concurrency level
CONC_COLORS = {1: "#1f77b4", 4: "#2ca02c", 10: "#ff7f0e"}
CONC_MARKERS = {1: "o", 4: "s", 10: "^"}

# Metric configurations
METRICS = {
    "e2el": {
        "key": "mean_e2e_ms",
        "ylabel": "Mean E2EL (ms)",
        "title_prefix": "Mean E2EL (ms)",
    },
    "ttfp": {
        "key": "mean_ttfp_ms",
        "ylabel": "Mean AUDIO TTFP (ms)",
        "title_prefix": "Mean AUDIO TTFP (ms)",
    },
    "rtf": {
        "key": "mean_rtf",
        "ylabel": "Mean AUDIO RTF",
        "title_prefix": "Mean AUDIO RTF",
    },
}

# Step labels for summary line plots and titles
STEP_LABELS = ["Baseline", "Batch", "CUDA Graph", "Async Chunk\n+ Streaming Output"]
STEP_KEYS = ["baseline", "batch", "cuda_graph", "async_chunk"]

# Pairwise comparison pairs
PAIRWISE = [
    ("baseline", "batch", "Baseline vs Batch"),
    ("batch", "cuda_graph", "Batch vs Batch + CUDA Graph"),
    ("cuda_graph", "async_chunk", "Batch + CUDA Graph vs Async Chunk"),
]


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def results_to_map(results: list[dict]) -> dict[int, dict]:
    return {r["concurrency"]: r for r in results}


def plot_pairwise_bar(
    before_results: list[dict],
    after_results: list[dict],
    before_label: str,
    after_label: str,
    metric_key: str,
    metric_cfg: dict,
    title: str,
    output_path: str,
):
    """Grouped bar chart: before vs after, one group per concurrency."""
    before_map = results_to_map(before_results)
    after_map = results_to_map(after_results)
    concurrencies = sorted(set(before_map.keys()) & set(after_map.keys()))

    if not concurrencies:
        print(f"  Skipping {output_path}: no common concurrency levels")
        return

    before_vals = [before_map[c][metric_cfg["key"]] for c in concurrencies]
    after_vals = [after_map[c][metric_cfg["key"]] for c in concurrencies]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(concurrencies))
    width = 0.3

    bars_before = ax.bar(
        x - width / 2,
        before_vals,
        width,
        label=before_label,
        color=COLOR_BEFORE,
        edgecolor="none",
    )
    bars_after = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label=after_label,
        color=COLOR_AFTER,
        edgecolor="none",
    )

    # Value labels on top of bars
    for bar_group in [bars_before, bars_after]:
        for bar in bar_group:
            height = bar.get_height()
            fmt = f"{height:,.2f}" if height < 10 else f"{height:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                fmt,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(f"{title}", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_cfg["ylabel"], fontsize=12)
    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_summary_lines(
    all_data: dict[str, list[dict]],
    metric_key: str,
    metric_cfg: dict,
    output_path: str,
):
    """Summary line plot: X = stacked optimization steps, one line per concurrency."""
    # Collect available steps
    available_steps = []
    available_labels = []
    for step_key, step_label in zip(STEP_KEYS, STEP_LABELS):
        if step_key in all_data:
            available_steps.append(step_key)
            available_labels.append(step_label)

    if len(available_steps) < 2:
        print(f"  Skipping summary {output_path}: need at least 2 steps")
        return

    # Collect concurrency levels present in all steps
    conc_sets = [set(results_to_map(all_data[s]).keys()) for s in available_steps]
    all_concs = sorted(set.intersection(*conc_sets))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(available_steps))

    for conc in all_concs:
        color = CONC_COLORS.get(conc, "gray")
        marker = CONC_MARKERS.get(conc, "x")
        vals = []
        for step_key in available_steps:
            step_map = results_to_map(all_data[step_key])
            vals.append(step_map[conc][metric_cfg["key"]])
        ax.plot(
            x,
            vals,
            color=color,
            marker=marker,
            markersize=10,
            linewidth=2.5,
            label=f"Concurrency {conc}",
        )

    ax.set_yscale("log")
    ax.set_title(
        f"{metric_cfg['title_prefix']} comparison: Qwen3-TTS",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(metric_cfg["ylabel"], fontsize=12)
    ax.set_xlabel("Stacked optimization", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, fontsize=10)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_vs_hf(
    vllm_results: list[dict],
    hf_results: list[dict],
    metric_key: str,
    metric_cfg: dict,
    output_path: str,
):
    """vLLM-Omni vs HF Transformers - single-request comparison.

    Matches Omni blog style: two single bars, HF on left (blue),
    vLLM-Omni on right (green), log Y-axis, "System" X-axis.
    """
    vllm_map = results_to_map(vllm_results)
    hf_map = results_to_map(hf_results)

    # Use concurrency 1 for single-request comparison (matching Omni blog)
    if 1 not in vllm_map or 1 not in hf_map:
        print(f"  Skipping {output_path}: no concurrency=1 data")
        return

    hf_val = hf_map[1][metric_cfg["key"]]
    vllm_val = vllm_map[1][metric_cfg["key"]]

    # Convert ms to seconds for E2EL and TTFP to match Omni blog units
    ylabel = metric_cfg["ylabel"]
    if metric_cfg["key"] in ("mean_e2e_ms", "mean_ttfp_ms"):
        hf_val /= 1000.0
        vllm_val /= 1000.0
        ylabel = ylabel.replace("(ms)", "(s)")

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["HF transformers (offline)", "vLLM-Omni (streaming)"]
    vals = [hf_val, vllm_val]
    colors = [COLOR_BEFORE, COLOR_AFTER]  # blue for HF (left), green for vLLM (right)

    bars = ax.bar(labels, vals, color=colors, edgecolor="none", width=0.5)

    # Value labels on top
    for bar in bars:
        height = bar.get_height()
        fmt = f"{height:,.2f}" if height < 10 else f"{height:,.1f}" if height < 100 else f"{height:,.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Determine title based on metric
    title_map = {
        "mean_e2e_ms": "E2E Latency (Single Request)",
        "mean_ttfp_ms": "TTFP (Single Request)",
        "mean_rtf": "RTF (Single Request)",
    }
    ax.set_title(title_map.get(metric_cfg["key"], metric_cfg["title_prefix"]), fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("System", fontsize=12)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def print_summary_table(all_data: dict[str, list[dict]]):
    """Print markdown summary table."""
    available_steps = [s for s in STEP_KEYS if s in all_data]
    if not available_steps:
        return

    conc_sets = [set(results_to_map(all_data[s]).keys()) for s in available_steps]
    all_concs = sorted(set.intersection(*conc_sets))

    print("\n## Benchmark Summary\n")

    for metric_name, cfg in METRICS.items():
        key = cfg["key"]
        print(f"\n### {cfg['title_prefix']}\n")
        header = (
            "| Concurrency | "
            + " | ".join(STEP_LABELS[STEP_KEYS.index(s)].replace("\n", " ") for s in available_steps)
            + " |"
        )
        sep = "| --- | " + " | ".join("---" for _ in available_steps) + " |"
        print(header)
        print(sep)
        for conc in all_concs:
            vals = []
            for step_key in available_steps:
                v = results_to_map(all_data[step_key])[conc][key]
                if metric_name == "rtf":
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(f"{v:,.1f}")
            print(f"| {conc} | " + " | ".join(vals) + " |")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stacked optimization benchmarks for Qwen3-TTS")
    parser.add_argument("--baseline", type=str, help="JSON result for baseline config")
    parser.add_argument("--batch", type=str, help="JSON result for batch config")
    parser.add_argument("--cuda-graph", type=str, help="JSON result for batch + CUDA graph config")
    parser.add_argument("--async-chunk", type=str, help="JSON result for async chunk + streaming config")
    parser.add_argument("--hf", type=str, help="JSON result for HF transformers baseline")
    parser.add_argument("--output-dir", type=str, default="results/", help="Output directory for figures")
    parser.add_argument("--timestamp", type=str, default="", help="Timestamp suffix for filenames")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = f"_{args.timestamp}" if args.timestamp else ""

    # Load all available data
    all_data: dict[str, list[dict]] = {}
    arg_map = {
        "baseline": args.baseline,
        "batch": args.batch,
        "cuda_graph": args.cuda_graph,
        "async_chunk": args.async_chunk,
    }
    for key, path in arg_map.items():
        if path:
            all_data[key] = load_results(path)
            print(f"Loaded {key}: {path}")

    hf_data = None
    if args.hf:
        hf_data = load_results(args.hf)
        print(f"Loaded HF: {args.hf}")

    if not all_data:
        print("ERROR: No result files provided.")
        return

    # Print summary table
    print_summary_table(all_data)

    # ---- 1. Pairwise bar charts ----
    print("\n--- Pairwise bar charts ---")
    pairwise_labels = {
        ("baseline", "batch"): ("Baseline", "Batch"),
        ("batch", "cuda_graph"): ("Batch", "Batch + CUDA Graph"),
        ("cuda_graph", "async_chunk"): ("Batch + CUDA Graph", "Async Chunk + Streaming"),
    }

    for before_key, after_key, title_suffix in PAIRWISE:
        if before_key not in all_data or after_key not in all_data:
            print(f"  Skipping {title_suffix}: missing data for {before_key} or {after_key}")
            continue

        before_label, after_label = pairwise_labels[(before_key, after_key)]

        for metric_name, metric_cfg in METRICS.items():
            fname = f"Mean_{metric_cfg['key']}_{before_key}_vs_{after_key}{ts}.png"
            plot_pairwise_bar(
                all_data[before_key],
                all_data[after_key],
                before_label,
                after_label,
                metric_name,
                metric_cfg,
                f"{metric_cfg['title_prefix']} comparison: {title_suffix}",
                str(out / fname),
            )

    # ---- 2. Summary line plots ----
    print("\n--- Summary line plots ---")
    for metric_name, metric_cfg in METRICS.items():
        fname = f"Summary_{metric_cfg['key']}_vs_features{ts}.png"
        plot_summary_lines(all_data, metric_name, metric_cfg, str(out / fname))

    # ---- 3. vLLM-Omni vs HF Transformers ----
    if hf_data:
        # Use the best config (async_chunk) for comparison, fallback to whatever is available
        best_key = None
        for k in reversed(STEP_KEYS):
            if k in all_data:
                best_key = k
                break

        if best_key:
            print(f"\n--- vLLM-Omni ({best_key}) vs HF Transformers ---")
            for metric_name, metric_cfg in METRICS.items():
                fname = f"{metric_cfg['title_prefix'].replace(' ', '_')}_vllm_omni_vs_transformers{ts}.png"
                plot_vs_hf(
                    all_data[best_key],
                    hf_data,
                    metric_name,
                    metric_cfg,
                    str(out / fname),
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
