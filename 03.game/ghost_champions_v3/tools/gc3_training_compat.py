"""Allow only exact, reviewed source pairs when retaining an existing v3 run."""

# Both complete fingerprints must match. Any other policy, map or reward change
# still requires a new run.
RESUME_SOURCE_MIGRATIONS = {
    # Only the optional macro checkpoint preference in __init__.py changed.
    # Guard/search rollout factories do not use the main-AI macro pipeline.
    ("f20bb62b1d34afa9f1291ed227fdab26219d41a329d5904b9c8b2b5f57ea975d",
     "be6c1ec6d80c6155f2aad8c6d25e9358c8df58d3d17b8f03cb29ac46b3d38457"):
        "isolated_main_ai_macro_checkpoint_preference_v1",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "be6c1ec6d80c6155f2aad8c6d25e9358c8df58d3d17b8f03cb29ac46b3d38457"):
        "isolated_main_ai_macro_checkpoint_preference_v1",
    # Only __init__.py's optional attacker macro factory was added. Macro code
    # lives in macro_v1 and is not used by the guard/search rollout factories.
    # Their model/observation/action/reward/map/optimizer code is unchanged.
    ("f0acd0190bcfe8609a49d7cd2b32f5489856a34187dc38eb0a2e4ec009255463",
     "f20bb62b1d34afa9f1291ed227fdab26219d41a329d5904b9c8b2b5f57ea975d"):
        "isolated_optional_attacker_macro_factory_v1",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "f20bb62b1d34afa9f1291ed227fdab26219d41a329d5904b9c8b2b5f57ea975d"):
        "isolated_optional_attacker_macro_factory_v1",
    # Only __init__.py's preplant checkpoint preference changed. New navigation,
    # round-win credit and time cost are isolated under preplant/. No guard or
    # search action, observation, reward, map or optimizer changed.
    ("aa19eb8ae5c575513e2e6cea4d00e89cbaa10a98cbae406300353492f6d0fedf",
     "f0acd0190bcfe8609a49d7cd2b32f5489856a34187dc38eb0a2e4ec009255463"):
        "isolated_preplant_actual_outcome_default_v3",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "f0acd0190bcfe8609a49d7cd2b32f5489856a34187dc38eb0a2e4ec009255463"):
        "isolated_preplant_actual_outcome_default_v3",
    ("a660ee39adca9d6746d007f8264f4ea76c50c373764ed429d2b57ea1a1b497fb",
     "f0acd0190bcfe8609a49d7cd2b32f5489856a34187dc38eb0a2e4ec009255463"):
        "isolated_preplant_actual_outcome_default_v3",
    # Only another preplant inference default was added in __init__.py.
    # Resource carry/rewards live under preplant and do not enter guard/search.
    ("a660ee39adca9d6746d007f8264f4ea76c50c373764ed429d2b57ea1a1b497fb",
     "aa19eb8ae5c575513e2e6cea4d00e89cbaa10a98cbae406300353492f6d0fedf"):
        "isolated_preplant_resource_run_default_v2",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "aa19eb8ae5c575513e2e6cea4d00e89cbaa10a98cbae406300353492f6d0fedf"):
        "isolated_preplant_resource_run_default_and_attacker_setup_v2",
    # Reviewed preplant-only changes: Omoko defaults/launcher exclusions and
    # attacker setup movement. Search acts as D; guard starts after the plant.
    # Neither phase's observations, legal actions, rewards or PPO are changed.
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "a660ee39adca9d6746d007f8264f4ea76c50c373764ed429d2b57ea1a1b497fb"):
        "isolated_preplant_omoko_defaults_and_attacker_setup_permission_v1",
    ("8f68f90d404337b55be214316e77f54e955df40dc6216c0df4c9a846c4c29b56",
     "a660ee39adca9d6746d007f8264f4ea76c50c373764ed429d2b57ea1a1b497fb"):
        "isolated_preplant_omoko_defaults_and_attacker_setup_permission_v1",
    # Only the preplant default checkpoint paths in __init__.py changed.
    ("03001a9c8837581516413468222b83f70fb39c041b7b558e15c95084ec7935f4",
     "8f68f90d404337b55be214316e77f54e955df40dc6216c0df4c9a846c4c29b56"):
        "isolated_preplant_relative_movement_default_paths_v2",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "8f68f90d404337b55be214316e77f54e955df40dc6216c0df4c9a846c4c29b56"):
        "isolated_preplant_pipeline_history_and_relative_defaults_v2",
    ("ce6e9682690c084d52977d37c7d05fa0c9547d144dc4c9413d9ef384d289f6a6",
     "8f68f90d404337b55be214316e77f54e955df40dc6216c0df4c9a846c4c29b56"):
        "isolated_preplant_history_and_relative_defaults_v2",
    ("2113b506798909fb52c6d720e749dc393fda59c74e1281edf2d2e12c8ed8f050",
     "8f68f90d404337b55be214316e77f54e955df40dc6216c0df4c9a846c4c29b56"):
        "terminal_fix_and_isolated_preplant_relative_defaults_v2",
    ("2113b506798909fb52c6d720e749dc393fda59c74e1281edf2d2e12c8ed8f050",
     "b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642"):
        "terminal_postplant_without_policy_decisions_v1",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "ce6e9682690c084d52977d37c7d05fa0c9547d144dc4c9413d9ef384d289f6a6"):
        "isolated_preplant_bc_rl_and_two_phase_runtime_v1",
    ("2113b506798909fb52c6d720e749dc393fda59c74e1281edf2d2e12c8ed8f050",
     "ce6e9682690c084d52977d37c7d05fa0c9547d144dc4c9413d9ef384d289f6a6"):
        "terminal_fix_and_isolated_preplant_pipeline_v1",
    # Public outcome callbacks and the new preplant runtime do not change
    # guard/search observations, weights, rewards, candidates or PPO updates.
    ("ce6e9682690c084d52977d37c7d05fa0c9547d144dc4c9413d9ef384d289f6a6",
     "03001a9c8837581516413468222b83f70fb39c041b7b558e15c95084ec7935f4"):
        "isolated_preplant_public_history_and_round_outcome_callback_v1",
    ("b77babb22c6687d2cd024bdc6c5f6b32cb5de4c9f6b00319bdd3d02177a7f642",
     "03001a9c8837581516413468222b83f70fb39c041b7b558e15c95084ec7935f4"):
        "isolated_preplant_pipeline_and_public_history_v1",
    ("2113b506798909fb52c6d720e749dc393fda59c74e1281edf2d2e12c8ed8f050",
     "03001a9c8837581516413468222b83f70fb39c041b7b558e15c95084ec7935f4"):
        "terminal_fix_and_isolated_preplant_public_history_v1",
}
MIGRATION_REASON = "terminal_postplant_without_policy_decisions_v1"


def validate_resume_source(previous, current):
    if previous == current or (previous, current) in RESUME_SOURCE_MIGRATIONS:
        return
    raise RuntimeError("Source/maps/reward changed since this run; use a new RUN_NAME")


def migrated_manifest(previous, current, step, episode):
    validate_resume_source(previous["source_hash"], current)
    if previous["source_hash"] == current:
        return dict(previous)
    history = list(previous.get("source_migrations", ()))
    history.append(dict(previous_source_hash=previous["source_hash"], source_hash=current,
                        reason=RESUME_SOURCE_MIGRATIONS[(previous["source_hash"], current)],
                        checkpoint_step=step, checkpoint_episode=episode))
    return dict(previous, source_hash=current, source_migrations=history)
