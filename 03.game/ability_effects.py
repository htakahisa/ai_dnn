"""Canvas effects shared by live games, tactical simulations, and replays."""

import math
from public_effects import displayed_projectile_path, displayed_area_cells

TUNNEL_WARNING_COLOR = "#fff6a0"
TUNNEL_ACTIVE_COLOR = "#ffd21a"
BALEMOON_WARNING_COLOR = "#ffc2c8"


def draw_balemoon_warnings(canvas, warnings, cell_size, offset_x=0):
    items = []
    for warning in warnings:
        for row, col in displayed_area_cells(warning):
            x, y = offset_x+col*cell_size, row*cell_size
            items.append(canvas.create_rectangle(x, y, x+cell_size, y+cell_size,
                         fill=BALEMOON_WARNING_COLOR, outline="#ee959e", stipple="gray25"))
    return items


def draw_ash_projectiles(canvas, projectiles, cell_size, offset_x=0):
    items = []
    for projectile in projectiles:
        path = displayed_projectile_path(projectile)
        if not path:
            continue
        progress = len(path)-1
        if progress > 0:
            coords = [coordinate for row, col in path[:progress+1]
                      for coordinate in (offset_x+(col+.5)*cell_size, (row+.5)*cell_size)]
            items.append(canvas.create_line(*coords, fill="#ee5264", width=2, dash=(3, 4)))
        row, col = path[progress]
        x, y = offset_x+(col+.5)*cell_size, (row+.5)*cell_size
        items.append(canvas.create_oval(x-4, y-4, x+4, y+4,
                                       fill="#ff9ca8", outline="#dc2638", width=2))
    return items


def draw_heal_sparkle(canvas, row, col, cell_size, remaining, offset_x=0):
    cx, cy = offset_x + (col+.5)*cell_size, (row+.5)*cell_size
    radius = cell_size*(.48 if remaining % 2 else .60)
    items = [canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                               fill="white", outline="white", width=2, stipple="gray25")]
    for dx, dy in ((-.48, -.30), (.48, .30), (.25, -.55), (-.25, .55)):
        x, y = cx+dx*cell_size, cy+dy*cell_size
        reach = cell_size*(.12 if remaining % 2 else .20)
        items.append(canvas.create_line(x-reach, y, x+reach, y, fill="white", width=2))
        items.append(canvas.create_line(x, y-reach, x, y+reach, fill="white", width=2))
    return items


def draw_destruction_areas(canvas, areas, cell_size, offset_x=0):
    items = []
    for area in areas:
        for row, col in displayed_area_cells(area):
            x, y = offset_x+col*cell_size, row*cell_size
            items.append(canvas.create_rectangle(x, y, x+cell_size, y+cell_size,
                         fill="#dc2638", outline="#ff6670", stipple="gray50"))
        row, col = area["pos"]
        items.append(canvas.create_text(offset_x+(col+.5)*cell_size, (row+.5)*cell_size,
                     text=f"Lv{area['level']}", fill="#fff2f2", font=("Arial", 7, "bold")))
    return items


def draw_contract_status(canvas, row, col, cell_size, remaining, offset_x=0):
    x, y = offset_x+col*cell_size, row*cell_size
    return [canvas.create_oval(x-2, y-2, x+cell_size+2, y+cell_size+2,
                             outline="#ff233f", width=3, dash=(3, 2)),
            canvas.create_text(x+cell_size/2, y+cell_size+5, text=f"契約 {remaining}",
                               fill="#ff233f", font=("Arial", 7, "bold"))]


def draw_serenade_flash(canvas, width, height, cell_size, offset_x=0):
    return [canvas.create_rectangle(offset_x, 0, offset_x+width*cell_size, height*cell_size,
                                    fill="white", outline="", stipple="gray50")]


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
