import json
from pathlib import Path
import numpy as np

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import extract_embeddings
from birds_ml.model import load_model
from birds_ml.metrics import compute_metrics, report

def main():
    cfg = Config()
    meta = json.loads((cfg.outputs_dir / "meta.json").read_text())
    class_to_idx = meta["class_to_idx"]

    model = load_model(str(cfg.outputs_dir / f"{meta['kind']}_{meta['backbone']}.joblib"))

    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

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

    pred = model.predict(Xv)
    m = compute_metrics(yv, pred)
    print(f"\n[VAL] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}\n")
    print(report(yv, pred))

if __name__ == "__main__":
    main()
