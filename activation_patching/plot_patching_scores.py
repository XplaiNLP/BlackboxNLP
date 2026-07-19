"""Render patching-score heatmap from a saved .p pickle (no GPU needed)."""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.io as pio


def ensure_kaleido_chrome() -> None:
    """Install Chrome for kaleido/plotly if missing (needed for PDF in containers)."""
    try:
        import kaleido
        if hasattr(kaleido, "get_chrome_sync"):
            kaleido.get_chrome_sync()
            print("kaleido Chrome ready (get_chrome_sync)")
            return
    except Exception as exc:
        print(f"kaleido.get_chrome_sync: {exc}")

    for cmd in (
        [sys.executable, "-m", "plotly_get_chrome"],
        ["plotly_get_chrome"],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
            if proc.returncode == 0:
                print(f"Chrome installed via {' '.join(cmd)}")
                return
            print(f"{' '.join(cmd)}:", proc.stderr or proc.stdout or "non-zero exit")
        except Exception as exc:
            print(f"{' '.join(cmd)} skipped: {exc}")


def save_patching_heatmap(
    rewrite_scores: np.ndarray,
    token_labels: list[str],
    layer_labels: list,
    plot_path: str | Path,
    *,
    plot_title: str = "",
    skip_layer0: bool = True,
) -> list[Path]:
    """Save heatmap as PDF (and HTML fallback). Returns paths written."""
    written: list[Path] = []
    plot_path = Path(plot_path)

    scores = rewrite_scores[1:, :] if skip_layer0 else rewrite_scores
    layers = layer_labels[1:] if skip_layer0 else layer_labels

    fig = px.imshow(
        scores,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Token", "y": "Layer", "color": " "},
        x=token_labels,
        y=layers,
        title=plot_title,
    )

    pdf_path = plot_path.with_suffix(".pdf")
    html_path = plot_path.with_suffix(".html")

    try:
        ensure_kaleido_chrome()
        pio.write_image(fig, str(pdf_path))
        written.append(pdf_path)
        print(f"Wrote {pdf_path}")
    except Exception as exc:
        print(f"PDF export failed ({exc}); saving HTML instead")
        fig.write_html(str(html_path))
        written.append(html_path)
        print(f"Wrote {html_path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot patching scores from pickle")
    parser.add_argument(
        "-scores_path",
        type=str,
        required=True,
        help="Path to gender-...-patch_scores.p",
    )
    parser.add_argument(
        "-output_path",
        type=str,
        default="",
        help="Output plot path (.pdf). Defaults to scores_path with .pdf suffix",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_path)
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    with open(scores_path, "rb") as f:
        results = pickle.load(f)

    out = args.output_path or str(scores_path).replace(".p", "_plot.pdf")
    save_patching_heatmap(
        results["rewrite_scores"],
        results["token_labels"],
        results["layer_labels"],
        out,
    )


if __name__ == "__main__":
    main()
