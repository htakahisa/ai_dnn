"""Initial defense uses the same independent v3 policy as search."""
from .gc3_learn_defender_search import GhostChampionsV3SearchController
GhostChampionsV3DefenderController = GhostChampionsV3SearchController
