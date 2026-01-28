from dataclasses import dataclass
from typing import Literal
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

Kind = Literal["svm", "logreg"]

@dataclass
class LinearCfg:
    kind: Kind = "svm"
    C: float = 1.0

def build_model(cfg: LinearCfg) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if cfg.kind == "svm":
        steps.append(("clf", LinearSVC(C=cfg.C, class_weight="balanced")))
    else:
        steps.append(("clf", LogisticRegression(
            C=cfg.C,
            max_iter=5000,
            multi_class="multinomial",
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
        )))
    return Pipeline(steps)

def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)
