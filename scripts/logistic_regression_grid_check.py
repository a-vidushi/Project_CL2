"""Does a wider L2 / epoch grid stop logistic regression losing accuracy when discourse features are added?"""
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from wsd_difficulty import config, pipeline
from wsd_difficulty.experiments import evaluate_target_word

GRIDS = {"current": ((1e-5, 1e-4, 1e-3), (5, 10, 20)),
         "wider": ((1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1), (2, 5, 10, 20))}


def use_grid(name):
    config.L2_GRID, config.EPOCH_CHECKPOINTS = GRIDS[name]


def run(job, grid):
    use_grid(grid)
    return [dict(row, grid=grid) for row in evaluate_target_word(job)]


if __name__ == "__main__":
    prepared = pipeline.load_prepared()
    predictions = pd.read_csv(config.PROCESSED_DIR / pipeline.PREDICTIONS_FILE)
    first = predictions[(predictions.model == "logistic_regression") & (predictions.feature_set == "all")]
    predicted = dict(zip(first.instance_id, first.predicted_sense))
    sets = {name: types for name, types in {**config.FEATURE_SETS, **config.DISCOURSE_FEATURE_SETS}.items()
            if name in ("all", "all+lexical_chains", "all+rst", "all+discourse")}
    jobs = [job for job in pipeline.build_jobs(prepared, ["logistic_regression"], sets, predicted)
            if len(job.training_examples) <= 1000]
    jobs = random.Random(7).sample(jobs, 150)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run, job, grid) for job in jobs for grid in GRIDS]
        rows = [row for future in futures for row in future.result()]
    table = pd.DataFrame(rows)
    table.to_csv(sys.argv[1], index=False)
    summary = table.pivot_table(index="feature_set", columns="grid", values="correct", aggfunc="mean")
    print(f"{len(jobs)} words, {table.instance_id.nunique()} instances")
    print((summary * 100).round(2))
