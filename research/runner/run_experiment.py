import json
from evaluation.metrics import compute_metrics


def run_experiment(model, dataloader, save_path):
    y_true, y_scores = model.predict(dataloader)

    metrics = compute_metrics(y_true, y_scores)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics:", metrics)
