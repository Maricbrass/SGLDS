import matplotlib.pyplot as plt


def plot_sensitivity(param_values, tpr_values):
    plt.figure()
    plt.plot(param_values, tpr_values)
    plt.xlabel("Einstein Radius")
    plt.ylabel("TPR")
    plt.title("Sensitivity Analysis")
    plt.savefig("research/artifacts/plots/sensitivity.png")
