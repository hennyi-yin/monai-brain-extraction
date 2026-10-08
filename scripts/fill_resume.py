"""Replace only the MONAI project placeholders after real held-out results exist."""
import argparse
import json
from pathlib import Path

from src.common import path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(path("results/summary.json").read_text(encoding="utf-8"))
    resume = result["resume"]
    source = args.input.read_text(encoding="utf-8")
    expected = (r"Achieved a Dice of \ph{[X]} on NFBS brain extraction (\ph{[N]} held-out T1w scans), "
                r"\ph{[Y]} points above FSL BET, by training a MONAI 3D U-Net with intensity normalization and spatial augmentation.")
    direction = "above" if resume["delta_points_unrounded"] >= 0 else "below"
    replacement = (f"Achieved a Dice of {resume['X']} on NFBS brain extraction ({resume['N']} held-out T1w scans), "
                   f"{abs(float(resume['Y'])):.1f} points {direction} FSL BET, by training a MONAI 3D U-Net "
                   "with intensity normalization and spatial augmentation.")
    if source.count(expected) != 1:
        raise ValueError("Expected exactly one matching project placeholder sentence")
    source = source.replace(expected, replacement)
    if r"\ph{" in source and not any(r"\ph{" in line for line in source.splitlines() if not line.startswith("%")):
        source = source.replace(r"\newcommand{\ph}[1]{\textcolor{red}{#1}}" + "\n", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
