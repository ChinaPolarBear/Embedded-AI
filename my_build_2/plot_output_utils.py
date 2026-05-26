from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def make_figure_output_dir(script_path: str) -> Path:
    script_name = Path(script_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(script_path).resolve().parent / "figures" / script_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig, output_dir: Path, name: str, dpi: int = 300) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "figure"
    output_path = output_dir / f"{safe_name}.png"

    suffix = 2
    while output_path.exists():
        output_path = output_dir / f"{safe_name}_{suffix}.png"
        suffix += 1

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved figure: {output_path}")
    return output_path
