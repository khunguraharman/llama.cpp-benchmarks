"""Compare M4 and RTX 3080 Eagle3 weighted throughput in one two-panel figure."""

from plot_mtp_hardware_comparison import main


if __name__ == "__main__":
    main(
        input_dirs=(
            "results/nvidia-rtx-3080/eagle3-speed-bench-results",
            "results/m4-mac-mini/mtp/qwen3-eagle3-speed-bench-results",
        ),
        output_subdir="spec-decoding",
        description=__doc__,
    )
