from tqdm import tqdm
import numpy as np
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# HACK: Manually add the project root to sys.path to enable relative imports
sys.path.append(os.path.dirname(SCRIPT_DIR))

# The linter (e.g., flake8/isort) would normally complain here,
# but the sys.path modification is required for finding 'src'.
from src.simulator import BurstGI  # noqa


def run_experiment(conf: BurstGI, repetitions: int = 100, disable_progress_bar: bool = True,
                   disable_subprogress_bar: bool = True) -> None:
    v_total, times_total = np.zeros(conf.n + conf.m + 1), np.zeros(conf.n + conf.m + 1)
    wasted_jobs_total = 0
    for _ in tqdm(range(repetitions), disable=disable_progress_bar):
        times, v, wasted_jobs, *_ = conf.run(disable_progress_bar=disable_subprogress_bar)

        v_total += v
        times_total += times
        wasted_jobs_total += wasted_jobs

    return times_total / repetitions, v_total / repetitions, wasted_jobs / repetitions


if __name__ == "__main__":
    times, v, wasted_jobs = run_experiment(
        BurstGI(10_000, 3, 2, 0.5)
    )

    print()
    print(f"Wasted jobs:\t\t{wasted_jobs}")
    print(f"States Frequencies:\t({', '.join(v.astype(str))})")
    print(f"Times distribution:\t({', '.join(times.round(2).astype(str))})")
