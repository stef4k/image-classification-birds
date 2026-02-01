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
    svm_loss: str = "squared_hinge"
    svm_dual: bool = True
    svm_max_iter: int = 5000
    svm_tol: float = 1e-4
    logreg_penalty: str = "l2"
    logreg_l1_ratio: float = 0.5
    logreg_max_iter: int = 1000
    logreg_solver: str = "saga"

def build_model(cfg: LinearCfg) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if cfg.kind == "svm":
        steps.append(("clf", LinearSVC(
            C=cfg.C,
            class_weight="balanced",
            loss=cfg.svm_loss,
            dual=cfg.svm_dual,
            max_iter=cfg.svm_max_iter,
            tol=cfg.svm_tol,
        )))
    else:
        if cfg.logreg_penalty in ("l1", "elasticnet") and cfg.logreg_solver != "saga":
            raise ValueError("logreg_solver must be 'saga' for l1 or elasticnet penalties")
        logreg_kwargs = dict(
            C=cfg.C,
            max_iter=cfg.logreg_max_iter,
            class_weight="balanced",
            solver=cfg.logreg_solver,
            penalty=cfg.logreg_penalty,
        )
        if cfg.logreg_penalty == "elasticnet":
            logreg_kwargs["l1_ratio"] = cfg.logreg_l1_ratio
        steps.append(("clf", LogisticRegression(**logreg_kwargs)))
    return Pipeline(steps)

def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)
