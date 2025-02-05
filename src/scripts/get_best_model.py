# type: ignore
from ray.tune import ExperimentAnalysis

analysis = ExperimentAnalysis("/home/stenheli/ray_results/load_and_train_2025-02-04_16-35-56")

best_trial = analysis.get_best_trial(metric="val_challenge_score", mode="max")

best_config = best_trial.config
best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="val_challenge_score", mode="max")

print("Best Config:", best_config)
print("Best Checkpoint Path:", best_checkpoint.path)
