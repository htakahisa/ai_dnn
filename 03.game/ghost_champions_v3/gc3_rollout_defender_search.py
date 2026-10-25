"""Setup and search episode entry point for the independent v3 policy."""
from .gc3_rollout import play_phase, Recorder, OPPONENTS


def play_search_round(code, seed, *, policy=None, gamma=.99, gae_lambda=.95):
    return play_phase(code, seed, phase="search", policy=policy, gamma=gamma, gae_lambda=gae_lambda)
