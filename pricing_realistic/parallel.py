"""Multiprocessing worker."""
from environments.demand_generator import make_default_env

from .config import CFG
from .runs import run_one, run_one_single_algorithm


def _run_task(args):
    """One (T, seed) task executed in a worker process."""
    T, seed, c_beta, delta, B_rkhs = args
    # Rebuild env deterministically in the worker (demand_seed fixed at 0).
    env, _ = make_default_env(cfg=CFG, seed=0)
    return run_one(T=T, env=env, seed=seed, c_beta=c_beta, delta=delta, B_rkhs=B_rkhs)


def _run_task_single(args):
    """One single-algorithm (T, seed, …, algorithm name) task."""
    T, seed, c_beta, delta, B_rkhs, algorithm = args
    env, _ = make_default_env(cfg=CFG, seed=0)
    return run_one_single_algorithm(
        algorithm,
        T=T,
        env=env,
        seed=seed,
        c_beta=c_beta,
        delta=delta,
        B_rkhs=B_rkhs,
    )

