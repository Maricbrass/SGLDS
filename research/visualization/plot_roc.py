import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc(y_true, y_scores, label):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.plot(fpr, tpr, label=label)
    plt.xscale("log")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.savefig("research/artifacts/plots/roc.png")
