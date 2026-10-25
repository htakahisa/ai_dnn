"""Postplant episode entry point for the independent v3 policy."""
from .gc3_rollout import play_phase, setup_postplant, Recorder, OPPONENTS


def play_guard_round(code, seed, *, policy=None, gamma=.99, gae_lambda=.95):
    return play_phase(code, seed, phase="postplant", policy=policy, gamma=gamma, gae_lambda=gae_lambda)
