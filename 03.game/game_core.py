# game_core.py
"""Character クラスとゲーム全体で共有する基本設定を集約するモジュール。"""

WINNING_ROUNDS = 13
TICK_TIME = 100
MAX_HP = 100
BODY_DAMAGE = 40
HEADSHOT_DAMAGE = 160
SHOOT_INTERVAL_TICKS = 1
SIDE_PANEL_WIDTH = 260
PLANT_REQUIRED_TICKS = 4
DEFUSE_REQUIRED_TICKS = 6
SMOKE_DURATION_TICKS = 15
MOVING_ACCURACY = 0.50
MOVING_TARGET_HIT_MULTIPLIER = 0.70
BLIND_DURATION_TICKS = 3
FLASH_BURST_DURATION_TICKS = 2
BLIND_ACCURACY_MULTIPLIER = 0.30
FLASH_SPEED_CELLS_PER_TICK = 3
FLASH_MAX_FLIGHT_TICKS = 5
RECON_SPEED_CELLS_PER_TICK = 3
REVEAL_DURATION_TICKS = 5
REVEALED_DODGE_MULTIPLIER = 0.50
RECON_REVEAL_SIZE = 9
COMBO_DISPLAY_TICKS = 3
COMBO_BANNER_HEIGHT = 112
ROUND_DURATION_TICKS = 90
SPIKE_DETONATION_TICKS = 45
RECON_BURST_DISPLAY_TICKS = 1
SMOKE_WARNING_TICKS = 3
ROUND_TRANSITION_TICKS = 2

ABILITY_TYPES = ["flash", "smoke", "recon", "none"]
FLASH_BLIND_TICKS = 3
SMOKE_RADIUS = 2
RECON_RADIUS = 4


class Character:
    def __init__(self, name, team, pos, text_color, bg_color, has_spike=False):
        self.name = name
        self.team = team
        self.pos = list(pos)
        self.text_color = text_color
        self.bg_color = bg_color
        self.is_alive = True
        self.just_died = False
        self.has_spike = has_spike
        self.plant_timer = 0
        self.defuse_timer = 0
        self.ability_type = "none"     # "flash" / "smoke" / "recon" / "none"
        self.flash_charges = 0
        self.smoke_charges = 0
        self.recon_charges = 0
        self.blind_remaining = 0       # フラッシュを受けている残りtick
        self.reveal_remaining = 0      # リコンで可視化されている残りtick