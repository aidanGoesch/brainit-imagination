import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_sweep import COND_ORDER, evaluate_run


def discover_runs(sweep_root):
    runs = []
    for run_dir in sorted(Path(sweep_root).glob("*")):
        result_path = run_dir / "result.csv"
        if not result_path.is_file():
            continue
        result = pd.read_csv(result_path).iloc[0].to_dict()
        runs.append((run_dir, result))
    noise_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return sorted(
        runs,
        key=lambda item: (
            noise_order.get(item[1].get("noise_level"), 99),
            float(item[1].get("signal_magnitude", 0)),
        ),
    )


def plot_rsms(sweep_root, output_path, subject="01", session="001"):
    runs = discover_runs(sweep_root)
    if not runs:
        raise ValueError(f"No completed runs found under {sweep_root}")

    first_ground_truth = pd.read_csv(
        runs[0][0] / "ground_truth_patterns.csv", index_col=0
    ).loc[COND_ORDER]
    rsms = [np.corrcoef(first_ground_truth.to_numpy())]
    titles = ["ground truth"]
    for run_dir, result in runs:
        _, rsm = evaluate_run(run_dir, subject=subject, session=session)
        rsms.append(rsm)
        if "noise_level" in result and pd.notna(result["noise_level"]):
            title = (
                f"{result['noise_level']}\n"
                f"mag={float(result['signal_magnitude']):.2f}"
            )
        else:
            title = f"mag={float(result['signal_magnitude']):.2f}"
        titles.append(title)

    ncols = min(5, len(rsms))
    nrows = int(np.ceil(len(rsms) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()
    for ax, rsm, title in zip(flat_axes, rsms, titles):
        image = ax.imshow(rsm, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(COND_ORDER)), COND_ORDER, fontsize=7)
        ax.set_yticks(range(len(COND_ORDER)), COND_ORDER, fontsize=7)
        ax.set_title(title, fontsize=9)
    for ax in flat_axes[len(rsms):]:
        ax.axis("off")
    fig.suptitle("Ground-truth and recovered condition RSMs")
    fig.colorbar(image, ax=flat_axes[:len(rsms)], shrink=0.7, label="correlation")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="./sweep_runs")
    parser.add_argument("--output", default="./rsm_sweep.png")
    parser.add_argument("--subject", default="01")
    parser.add_argument("--session", default="001")
    args = parser.parse_args()
    plot_rsms(
        args.sweep_root,
        args.output,
        subject=args.subject,
        session=args.session,
    )