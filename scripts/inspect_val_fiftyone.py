import json
import numpy as np
import fiftyone as fo

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import extract_embeddings
from birds_ml.model import load_model
from birds_ml.utils import ensure_dir

def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.cache_dir)

    # Load meta for consistent label mapping + model info
    meta = json.loads((cfg.outputs_dir / "meta.json").read_text())
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    # Load trained model
    model_path = cfg.outputs_dir / f"{meta['kind']}_{meta['backbone']}.joblib"
    model = load_model(str(model_path))

    # Load val samples with same mapping
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    # Load or extract embeddings
    cache_path = cfg.cache_dir / f"emb_{cfg.backbone}_val.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        Xv, yv = z["X"], z["y"]
        print(f"Loaded cache: {cache_path}  X={Xv.shape}")
    else:
        Xv, yv, _ = extract_embeddings(
            val_samples, cfg.backbone, cfg.batch_size, cfg.num_workers, cfg.device
        )
        np.savez_compressed(cache_path, X=Xv, y=yv)
        print(f"Saved cache: {cache_path}  X={Xv.shape}")

    # Predict labels
    pred_idx = model.predict(Xv).astype(int)
    pred_name = [idx_to_class[i] for i in pred_idx]
    true_name = [idx_to_class[int(y)] for y in yv]
    correct = [p == t for p, t in zip(pred_name, true_name)]

    # Confidence + hardness = 1 - confidence (only if model supports predict_proba)
    conf = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xv)
        conf = probs.max(axis=1).astype(float)
        hardness = (1.0 - conf).astype(float)
    else:
        conf = [None] * len(val_samples)
        hardness = [None] * len(val_samples)

    # Build FiftyOne dataset
    ds_name = "birds_val_preds"
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)

    dataset = fo.Dataset(ds_name)

    for i, s in enumerate(val_samples):
        sample = fo.Sample(filepath=str(s.path))

        sample["ground_truth"] = fo.Classification(label=true_name[i])
        sample["prediction"] = fo.Classification(label=pred_name[i])
        sample["correct"] = bool(correct[i])

        if conf[i] is not None:
            sample["confidence"] = float(conf[i])
            sample["hardness"] = float(hardness[i])

        dataset.add_sample(sample)

    dataset.persistent = True

    print(f"Dataset created: {dataset.name} with {len(dataset)} samples")
    print("Tip: in the App, filter `correct == false` and sort by `hardness` descending.")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()
