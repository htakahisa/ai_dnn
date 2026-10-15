"""Isolate opponent simulations in Windows-compatible spawned processes."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os


def initialize_worker():
    # Each opponent already occupies one CPU worker. Avoid nested BLAS pools.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    torch.set_num_threads(1)


def run_opponent_tasks(function, tasks, workers):
    tasks = list(tasks)
    if workers < 1:
        raise ValueError('workers must be positive')
    if not tasks:
        return []
    count = min(workers, len(tasks))
    print(f'opponents={len(tasks)} processes={count}', flush=True)
    if count == 1:
        initialize_worker()
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=count,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        # Preserve input ordering and propagate worker failures to the caller.
        return list(pool.map(function, tasks))
