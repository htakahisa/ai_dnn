"""Ghost Champions Defender Search configuration.

In ``map_data_search_gc.SEARCH_MAZE_STR``, digits 5--9 are defensive
position usage priorities only. A smaller digit is sampled more frequently:
5 is most used and 9 is least used. They are not player IDs, nor do they
change aggression or release behavior.
"""

GC_SEARCH_USAGE_MARKERS = (5, 6, 7, 8, 9)

# Relative sampling weights. Positions are selected without overlap inside a
# team, so multiple defenders cannot be assigned to the same cell.
GC_SEARCH_USAGE_WEIGHT_BY_MARKER = {
    5: 9.0,
    6: 6.0,
    7: 4.0,
    8: 2.0,
    9: 1.0,
}

GC_SEARCH_POSITION_RANDOMNESS = 0.35

# This applies to all assigned defensive cells equally, independent of their
# usage marker.
GC_SEARCH_RELEASE = {"min_seen": 2, "max_bfs": 10}
