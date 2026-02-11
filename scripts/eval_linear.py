import argparse
import json
import numpy as np
from dataclasses import replace
from pathlib import Path

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import extract_embeddings, build_embedding_transforms
from birds_ml.model import load_model
from birds_ml.metrics import compute_metrics, report


def main():
    cfg = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["svm", "logreg"], default=None)
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default=None)
    ap.add_argument("--meta_path", type=str, default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--use_crops", action="store_true", help="Use cropped datasets")
    ap.add_argument("--img_size", type=int, default=None, help="Override meta img_size")
    ap.add_argument("--resize_train", type=int, default=None, help="Override meta resize_train")
    ap.add_argument("--crop_scale_min", type=float, default=None)
    ap.add_argument("--crop_scale_max", type=float, default=None)
    ap.add_argument("--rotation_deg", type=float, default=None)
    ap.add_argument("--hflip_p", type=float, default=None)
    args = ap.parse_args()
    cfg = replace(cfg, use_crops=args.use_crops)
    crop_suffix_from_args = "_cropped" if cfg.use_crops else ""

    # Choose meta file: explicit path, specific kind/backbone, or latest
    if args.meta_path:
        meta_candidate = Path(args.meta_path)
        meta_path = meta_candidate if meta_candidate.is_absolute() else (cfg.outputs_dir / meta_candidate)
    elif args.kind and args.backbone:
        meta_filename = f"meta_{args.kind}_{args.backbone}{crop_suffix_from_args}.json"
        meta_path = cfg.outputs_dir / meta_filename
    else:
        meta_path = cfg.outputs_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    if not args.use_crops and "use_crops" in meta:
        cfg = replace(cfg, use_crops=bool(meta["use_crops"]))

    crop_suffix = "_cropped" if cfg.use_crops else ""
    class_to_idx = meta["class_to_idx"]
    backbone = meta["backbone"]
    kind = meta["kind"]
    img_size = args.img_size if args.img_size is not None else meta.get("img_size", 224)
    resize_train = args.resize_train if args.resize_train is not None else meta.get("resize_train", 256)
    crop_scale_min = args.crop_scale_min if args.crop_scale_min is not None else meta.get("crop_scale_min", 0.7)
    crop_scale_max = args.crop_scale_max if args.crop_scale_max is not None else meta.get("crop_scale_max", 1.0)
    rotation_deg = args.rotation_deg if args.rotation_deg is not None else meta.get("rotation_deg", 30.0)
    hflip_p = args.hflip_p if args.hflip_p is not None else meta.get("hflip_p", 0.5)

    model_name = meta.get("model_name", f"{kind}_{backbone}{crop_suffix}")
    model_path = cfg.outputs_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(str(model_path))

    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    tfm_suffix = (
        f"_img{img_size}"
        f"_rs{resize_train}"
        f"_cs{crop_scale_min:g}-{crop_scale_max:g}"
        f"_rot{rotation_deg:g}"
        f"_hf{hflip_p:g}"
    )
    cache_path = cfg.cache_dir / f"emb_{backbone}_val{crop_suffix}{tfm_suffix}.npz"
    if cache_path.exists() and not args.no_cache:
        z = np.load(cache_path, allow_pickle=True)
        Xv, yv = z["X"], z["y"]
        print(f"Loaded cache: {cache_path}  X={Xv.shape}")
    else:
        standard_tfm, _ = build_embedding_transforms(
            img_size=img_size,
            resize_train=resize_train,
            crop_scale_min=crop_scale_min,
            crop_scale_max=crop_scale_max,
            rotation_deg=rotation_deg,
            hflip_p=hflip_p,
        )
        Xv, yv, _ = extract_embeddings(
            val_samples, backbone, cfg.batch_size, cfg.num_workers, cfg.device, transform=standard_tfm
        )
        np.savez_compressed(cache_path, X=Xv, y=yv)
        print(f"Saved cache: {cache_path}  X={Xv.shape}")

    pred = model.predict(Xv)
    m = compute_metrics(yv, pred)
    print(f"\n[VAL] kind={kind} backbone={backbone} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}\n")
    print(report(yv, pred))


if __name__ == "__main__":
    main()
