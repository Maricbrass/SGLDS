def compare(sim_metrics, real_metrics):
    for key in sim_metrics:
        drop = sim_metrics[key] - real_metrics[key]
        print(f"{key} drop: {drop:.4f}")
