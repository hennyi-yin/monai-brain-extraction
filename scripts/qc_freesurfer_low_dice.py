"""Inspect low-Dice saved masks without running the pipeline or scoring again."""
import nibabel as nib
import numpy as np
import pandas as pd

from src.common import ROOT, config, digest, load_split
from src.evaluate_freesurfer import plot_overlays
from src.freesurfer_common import OUTPUT, read_json


def main():
    csv = OUTPUT / "per_subject.csv"
    marker = read_json(OUTPUT / "evaluation_once.json")
    if marker["status"] != "complete" or marker["csv_sha256"] != digest(csv):
        raise ValueError("Expected the original completed score file")
    records = read_json(OUTPUT / "manifest.json")["subjects"]
    lookup = {row["id"]: row for row in records}
    items = {item["id"]: item for item in load_split(config(), ("test",))["test"]}
    scores = pd.read_csv(csv).sort_values("dice")
    low = list(scores.loc[scores.dice < .8, "id"])
    if not low:
        print("No masks below Dice 0.8")
        return
    plot_overlays([items[sid] for sid in low], records, OUTPUT / "figures/low_dice_qc.png")
    for sid in low:
        mask = np.asarray(nib.load(ROOT / lookup[sid]["output"]).dataobj) != 0
        image = np.asarray(nib.load(items[sid]["image"]).dataobj)
        print(sid, lookup[sid]["status"], "foreground", int(mask.sum()),
              "foreground on native zero intensity", int(np.count_nonzero(mask & (image == 0))))


if __name__ == "__main__":
    main()
