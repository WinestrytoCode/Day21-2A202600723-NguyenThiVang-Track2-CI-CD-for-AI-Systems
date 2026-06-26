import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70

# Bonus 2: anh xa ten thuat toan -> lop sklearn tuong ung.
# Cac tham so khong hop le voi tung thuat toan se duoc loc o build_model().
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "extra_trees": ExtraTreesClassifier,
    "logistic_regression": LogisticRegression,
}


def build_model(params: dict):
    """
    Bonus 2: Khoi tao mo hinh dua tren tham so model_type.

    Tach model_type ra khoi params; phan con lai la sieu tham so cua thuat toan.
    Mac dinh la random_forest de tuong thich nguoc voi cac test cu.
    """
    hp = dict(params)
    model_type = hp.pop("model_type", "random_forest")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model_type khong hop le: '{model_type}'. "
            f"Chon mot trong: {list(MODEL_REGISTRY)}"
        )

    cls = MODEL_REGISTRY[model_type]
    # logistic_regression khong nhan cac tham so dac thu cua cay quyet dinh
    if model_type == "logistic_regression":
        hp.pop("max_depth", None)
        hp.pop("n_estimators", None)
        hp.pop("min_samples_split", None)
        hp.setdefault("max_iter", 1000)
        return model_type, cls(random_state=42, **hp)

    return model_type, cls(random_state=42, **hp)


def check_data_drift(y_train) -> dict:
    """
    Bonus 5: Tinh phan phoi nhan va canh bao neu lop nao chiem < 10%.

    Tra ve dict ty le moi lop (de ghi vao metrics.json).
    """
    counts = y_train.value_counts().sort_index()
    total = int(counts.sum())
    distribution = {str(label): count / total for label, count in counts.items()}

    for label, ratio in distribution.items():
        if ratio < 0.10:
            print(
                f"CANH BAO LECH LAC DU LIEU: lop {label} chi chiem "
                f"{ratio:.1%} tong so mau (< 10%)."
            )
    return distribution


def build_report(y_eval, preds) -> str:
    """
    Bonus 3: Tao bao cao van ban gom confusion matrix va precision/recall
    cho tung lop (0, 1, 2).
    """
    cm = confusion_matrix(y_eval, preds)
    report = classification_report(
        y_eval, preds, digits=4, zero_division=0
    )
    lines = ["=== Confusion Matrix ===", str(cm), "", "=== Classification Report ===", report]
    return "\n".join(lines)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua model_type (tuy chon) va cac sieu tham so.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: kiem tra lech lac du lieu truoc khi huan luyen
    label_distribution = check_data_drift(y_train)

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen mo hinh (Bonus 2: chon thuat toan)
        model_type, model = build_model(params)
        mlflow.set_tag("model_type", model_type)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # Bonus 3: precision/recall cho tung lop, ghi nhan vao MLflow
        precision, recall, _, _ = precision_recall_fscore_support(
            y_eval, preds, labels=[0, 1, 2], zero_division=0
        )
        for cls_idx in range(3):
            mlflow.log_metric(f"precision_class_{cls_idx}", precision[cls_idx])
            mlflow.log_metric(f"recall_class_{cls_idx}", recall[cls_idx])

        # TODO 7: In ket qua ra man hinh
        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            # Bonus 5: phan phoi nhan
            "label_distribution": label_distribution,
            # Bonus 3: precision/recall tung lop
            "precision_per_class": {str(i): precision[i] for i in range(3)},
            "recall_per_class": {str(i): recall[i] for i in range(3)},
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Bonus 3: bao cao van ban (confusion matrix + classification report)
        report_text = build_report(y_eval, preds)
        with open("outputs/report.txt", "w") as f:
            f.write(report_text + "\n")
        mlflow.log_artifact("outputs/report.txt")
        print(report_text)

        # TODO 9: Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
