"""Canvas effects shared by live games, tactical simulations, and replays."""

import math


def draw_raid_wind(canvas, trail, cell_size, offset_x=0):
    """Draw curved red gusts on each cell actually crossed by RAID."""
    dr, dc = trail["direction"]
    length = math.hypot(dr, dc)
    if not length:
        return []
    fx, fy = dc / length, dr / length
    nx, ny = -fy, fx
    remaining = int(trail.get("remaining_ticks", 1))
    color = "#ee3546" if remaining >= 3 else "#f66c76" if remaining == 2 else "#ffafb5"
    items = []
    for row, col in trail.get("cells", []):
        cx, cy = offset_x + (col + .5)*cell_size, (row + .5)*cell_size
        for lateral, scale in ((-.22, .78), (0, 1.0), (.22, .78)):
            points = []
            for forward, bend in ((-.44, .12), (-.16, -.06), (.18, -.06), (.44, .10)):
                points.extend((
                    cx + cell_size*(fx*forward*scale + nx*(lateral + bend)),
                    cy + cell_size*(fy*forward*scale + ny*(lateral + bend)),
                ))
            items.append(canvas.create_line(*points, fill=color, width=3,
                                            smooth=True, capstyle="round"))
            items.append(canvas.create_line(*points, fill="#ffd5d8", width=1,
                                            smooth=True, capstyle="round"))
    return items
