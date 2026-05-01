import numpy as np


def aggregate(results):
    keys = results[0].keys()
    final = {}

    for k in keys:
        values = [r[k] for r in results]
           final[k] = f"{np.mean(values):.4f} +/- {np.std(values):.4f}"

    return final
