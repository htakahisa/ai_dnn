from __future__ import annotations

import hashlib
import random
from typing import Any
import numpy as np

IQ_CAP = 200.0


def _alive(c: Any) -> bool:
    return bool(getattr(c, "is_alive", True))


def _pos(v: Any):
    if v is None:
        return None
    try:
        return int(v[0]), int(v[1])
    except Exception:
        return None


def _effective_iq(viewer: Any) -> float:
    try:
        return max(
            0.0, float(getattr(viewer, "effective_iq", getattr(viewer, "iq", 50.0)))
        )
    except Exception:
        return 50.0


def _quality(viewer: Any) -> float:
    x = max(0.0, min(1.0, _effective_iq(viewer) / IQ_CAP))
    return x * x


def _seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


class PerceivedCharacter:
    __slots__ = ("_real", "_overrides")

    def __init__(self, real: Any, **overrides: Any):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_overrides", dict(overrides))

    @property
    def real_character(self):
        return object.__getattribute__(self, "_real")

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name in {"_real", "_overrides"}:
            object.__setattr__(self, name, value)
            return
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            overrides[name] = value
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)


class PerceivedGameView:
    __slots__ = ("_real", "_overrides", "_map")

    def __init__(
        self,
        real: Any,
        overrides: dict[str, Any],
        mapping: dict[int, PerceivedCharacter],
    ):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_overrides", overrides)
        object.__setattr__(self, "_map", mapping)

    @property
    def real_game(self):
        return object.__getattribute__(self, "_real")

    def perceived_character_for(self, real_char: Any):
        return object.__getattribute__(self, "_map")[id(real_char)]

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name in {"_real", "_overrides", "_map"}:
            object.__setattr__(self, name, value)
            return
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            overrides[name] = value
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)


class IQPerceptionEngine:
    def __init__(self):
        self._cache = {}
        self._last_tick = None

        # ラウンド中に一度でもスパイク解除を開始した選手を記憶する。
        # Character実体のidを使うため、同名選手がいても混同しない。
        self._defuse_touched_viewers = set()
        self._memory_round = None

    def clear_cache(self):
        self._cache.clear()
        self._last_tick = None
        self._defuse_touched_viewers.clear()
        self._memory_round = None

    def _sync_round_memory(self, game):
        """ラウンドが変わったら、解除位置記憶をリセットする。"""
        current_round = int(getattr(game, "current_round", 0))
        if self._memory_round != current_round:
            self._defuse_touched_viewers.clear()
            self._memory_round = current_round

    def _update_defuse_touch_memory(self, viewer, game):
        """解除ゲージが進んでいるなら、このラウンドの解除経験者として登録する。"""
        self._sync_round_memory(game)
        # Defenderが解除を1Tickでも開始した時点で記憶する。
        # defuse_timerは解除中に正の値になる既存フィールド。
        try:
            defuse_progress = float(getattr(viewer, "defuse_timer", 0.0))
        except (TypeError, ValueError):
            defuse_progress = 0.0

        is_defender = str(getattr(viewer, "team", "")) == "D"
        if is_defender and defuse_progress > 0.0:
            self._defuse_touched_viewers.add(id(viewer))

    def has_started_defuse(self, viewer, game) -> bool:
        """このラウンド中にviewerが一度でも解除を開始したか。"""
        self._update_defuse_touch_memory(viewer, game)
        return id(viewer) in self._defuse_touched_viewers

    def quality(self, viewer):
        return _quality(viewer)

    def _tick_key(self, game):
        return int(getattr(game, "current_round", 0)), int(
            getattr(game, "battle_tick", 0)
        )

    def _rng(self, game, viewer, channel, subject=""):
        r, t = self._tick_key(game)
        return random.Random(_seed(r, t, getattr(viewer, "name", ""), channel, subject))

    @staticmethod
    def _walkable(grid, p):
        r, c = p
        return (
            0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) != 1
        )

    def _nearest_walkable(self, grid, wanted, fallback):
        if self._walkable(grid, wanted):
            return wanted
        for radius in range(1, max(grid.shape) + 1):
            candidates = []
            wr, wc = wanted
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    p = (wr + dr, wc + dc)
                    if self._walkable(grid, p):
                        candidates.append(p)
            if candidates:
                return min(
                    candidates,
                    key=lambda p: abs(p[0] - fallback[0]) + abs(p[1] - fallback[1]),
                )
        return fallback

    def _blur_pos(self, game, viewer, value, maximum, channel, subject):
        p = _pos(value)
        if p is None:
            return None
        q = self.quality(viewer)
        radius = int(round((1.0 - q) * maximum))
        if radius <= 0:
            return [p[0], p[1]]
        rng = self._rng(game, viewer, channel, subject)
        wanted = (
            p[0] + rng.randint(-radius, radius),
            p[1] + rng.randint(-radius, radius),
        )
        corrected = self._nearest_walkable(np.asarray(game.grid), wanted, p)
        return [corrected[0], corrected[1]]

    def _enemy_hp(self, game, viewer, enemy):
        hp = max(0, int(getattr(enemy, "hp", 0)))
        q = self.quality(viewer)
        if q >= 1.0:
            return hp
        if q < 0.12:
            return 100 if _alive(enemy) else 0
        step = max(5, int(round(5 + (1.0 - q) * 20)))
        return int(round(hp / step) * step)

    def _timer(self, value, viewer):
        try:
            value = float(value)
        except Exception:
            return 0.0
        q = self.quality(viewer)
        step = max(1, int(round(1 + (1.0 - q) * 11)))
        return float(round(value / step) * step)

    def _omit_enemy(self, game, viewer, enemy):
        vp, ep = _pos(viewer.pos), _pos(enemy.pos)
        if vp is None or ep is None:
            return False
        if max(abs(vp[0] - ep[0]), abs(vp[1] - ep[1])) <= 3:
            return False
        q = self.quality(viewer)
        return (
            self._rng(game, viewer, "omit", getattr(enemy, "name", "")).random()
            < (1.0 - q) * 0.55
        )

    def build_game_view(self, *, viewer, game):
        self._update_defuse_touch_memory(viewer, game)
        started_defuse = id(viewer) in self._defuse_touched_viewers

        tick = self._tick_key(game)
        if tick != self._last_tick:
            self._cache.clear()
            self._last_tick = tick
        key = (
            id(game),
            tick,
            id(viewer),
            getattr(viewer, "name", ""),
            round(_effective_iq(viewer), 3),
            started_defuse,
        )
        if key in self._cache:
            return self._cache[key]

        proxies, mapping = [], {}
        for real in game.chars:
            if real is viewer:
                pos, hp, alive = list(real.pos), int(real.hp), _alive(real)
            elif real.team == viewer.team:
                pos = self._blur_pos(game, viewer, real.pos, 2, "ally", real.name)
                hp, alive = int(real.hp), _alive(real)
            else:
                pos = self._blur_pos(game, viewer, real.pos, 6, "enemy", real.name)
                hp, alive = self._enemy_hp(game, viewer, real), _alive(real)
            proxy = PerceivedCharacter(real, pos=pos, hp=hp, is_alive=alive)
            if real.team != viewer.team and self._omit_enemy(game, viewer, real):
                proxy.is_alive = False
            proxies.append(proxy)
            mapping[id(real)] = proxy

        real_spike_pos = _pos(getattr(game, "spike_pos", None))
        real_planted_pos = _pos(getattr(game, "planted_pos", None))

        # 一度でもスパイク解除を開始したディフェンダーは、そのラウンド中、
        # 設置済みスパイク位置をIQ誤差なしで正確に把握し続ける。
        # 未設置で地面に落ちているspike_posは、解除経験とは無関係なので
        # 従来どおりIQによる座標誤差を受ける。
        perceived_spike_pos = self._blur_pos(
            game,
            viewer,
            getattr(game, "spike_pos", None),
            4,
            "spike",
            "drop",
        )
        perceived_planted_pos = (
            [real_planted_pos[0], real_planted_pos[1]]
            if started_defuse and real_planted_pos is not None
            else self._blur_pos(
                game,
                viewer,
                getattr(game, "planted_pos", None),
                4,
                "spike",
                "plant",
            )
        )

        overrides = {
            "grid": game.grid,
            "chars": proxies,
            "spike_pos": perceived_spike_pos,
            "planted_pos": perceived_planted_pos,
            "target_plant_pos": self._blur_pos(
                game,
                viewer,
                getattr(game, "target_plant_pos", None),
                3,
                "site",
                "target",
            ),
            "round_timer": self._timer(getattr(game, "round_timer", 0), viewer),
            "detonate_timer": self._timer(getattr(game, "detonate_timer", 0), viewer),
        }
        view = PerceivedGameView(game, overrides, mapping)
        self._cache[key] = view
        return view

    def build_perceived_state(self, *, viewer, game_state, game_view):
        state = dict(game_state)
        state["grid"] = game_view.grid
        state["chars"] = game_view.chars
        state["spike_pos"] = game_view.spike_pos
        state["planted_pos"] = game_view.planted_pos
        state["target_plant_pos"] = game_view.target_plant_pos
        state["detonate_timer"] = game_view.detonate_timer
        if "round_timer" in state:
            state["round_timer"] = self._timer(state["round_timer"], viewer)

        spotted = state.get("spotted_info")
        if isinstance(spotted, dict) and float(spotted.get("spotted", 0.0)) > 0:
            spotted = dict(spotted)
            shifted = self._blur_pos(
                game_view.real_game,
                viewer,
                (spotted.get("site_r", 0), spotted.get("site_c", 0)),
                6,
                "spotted",
                "holder",
            )
            if shifted is not None:
                spotted["site_r"], spotted["site_c"] = float(shifted[0]), float(
                    shifted[1]
                )
            if (
                self._rng(game_view.real_game, viewer, "spotted_miss").random()
                < (1.0 - self.quality(viewer)) * 0.33
            ):
                spotted = {"spotted": 0.0, "site_r": 0.0, "site_c": 0.0}
            state["spotted_info"] = spotted
        return state
