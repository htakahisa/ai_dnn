"""FRC/right plant extension template. Stateless identity hooks; no learning loop.

Explicitly enable this module in tv3_train_attacker_plant.PLANT_CUSTOM_MODULES.
The same goal/input hooks run during collection, evaluation and inference.
Only public ExtensionContext data may influence goal/input hooks.
"""
EXTENSION_SPEC=dict(id='attacker_plant_frc_v1_R',revision=1,
                    role='attacker_plant',opponent='frc_v1',site='R')


def adjust_goal(context,goal):
    return goal


def adjust_inputs(context,inputs):
    # Optional observation/teacher/mask restrictions, before common movement restrictions.
    # Keep shape/action kinds; replacement ability targets must be engine-legal.
    # Do not force chosen inference actions.
    return inputs


def adjust_reward(context,reward):
    # Optional per-actor step reward. No enemy-private data is supplied here.
    return reward
