from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=4)
