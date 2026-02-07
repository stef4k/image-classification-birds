import pandas as pd
from pathlib import Path
import fiftyone as fo

root = Path("data/extra_images")  # local path to same image tree
pred_df = pd.read_csv("outputs/extra_images_cv_ensemble_predictions_with_conf.csv")
pred_map = {str(r["relative_path"]).replace("\\", "/"): r for _, r in pred_df.iterrows()}

samples = []
matched = 0
missing = 0
for p in root.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        continue

    rel = str(p.relative_to(root)).replace("\\", "/")
    row = pred_map.get(rel)

    s = fo.Sample(filepath=str(p))
    s["ground_truth"] = fo.Classification(label=p.parent.name)  # remove if unlabeled

    if row is not None:
        matched += 1
        conf = float(row["confidence"])
        s["prediction"] = fo.Classification(
            label=str(row["predicted_label"]),
            confidence=conf,
        )
        # Expose confidence as a top-level scalar field for easy filtering/sorting in the grid.
        s["prediction_confidence"] = conf
    else:
        missing += 1

    samples.append(s)

dataset_name = "birds_extra_cv_ensemble"
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)
ds = fo.Dataset(dataset_name)
ds.add_samples(samples)
print(f"Loaded {len(samples)} samples | matched predictions: {matched} | missing: {missing}")
session = fo.launch_app(ds)
session.wait()
