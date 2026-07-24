# game_core.py
"""Character クラスとゲーム全体で共有する基本設定を集約するモジュール。"""






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