"""Controller-independent round tactic classifier."""


class TacticalRoundTracker:
    def __init__(self, width, plant_cells=()):
        self.width = int(width)
        self.plant_cells = {tuple(map(int, p)) for p in plant_cells}
        self.reset()

    def reset(self):
        self.start_positions = {}
        self.first_site_tick = {}
        self.visited_sites = set()
        self.max_spread = 0
        self.ticks = 0
        self.initial_defender_site_counts = {"A": 0, "B": 0}
        self.defender_site_history = {}

    def _site(self, pos):
        if not pos:
            return None
        col = int(pos[1])
        if col < self.width / 3:
            return "A"
        if col > self.width * 2 / 3:
            return "B"
        return None

    def begin_round(self, chars):
        self.reset()
        self.start_positions = {
            str(c.name): tuple(c.pos)
            for c in chars if c.team == "A"
        }
        self.initial_defender_site_counts = self._defender_site_counts(chars)
        self.defender_site_history[0] = dict(self.initial_defender_site_counts)

    def _defender_site_counts(self, chars):
        counts = {"A": 0, "B": 0}
        for char in chars:
            if getattr(char, "team", None) != "D" or not getattr(char, "is_alive", True):
                continue
            site = self._site(getattr(char, "pos", None))
            if site in counts:
                counts[site] += 1
        return counts

    def observe(self, chars, tick):
        self.ticks = int(tick)
        self.defender_site_history[int(tick)] = self._defender_site_counts(chars)
        attackers = [c for c in chars if c.team == "A" and c.is_alive]
        sites = [self._site(c.pos) for c in attackers]
        self.visited_sites.update(site for site in sites if site)
        if attackers:
            cols = [int(c.pos[1]) for c in attackers]
            self.max_spread = max(self.max_spread, max(cols) - min(cols))
            for site in self.visited_sites:
                self.first_site_tick.setdefault(site, self.ticks)

    def finish(self, planted_pos=None, target_pos=None):
        final_pos = planted_pos or target_pos
        final_site = self._site(final_pos)
        if final_site is None and final_pos in self.plant_cells:
            final_site = "A" if final_pos[1] < self.width / 2 else "B"
        visited_other = bool(
            final_site and any(site != final_site for site in self.visited_sites)
        )
        if visited_other:
            tactic = "fake"
        elif self.max_spread >= max(8, self.width // 4):
            tactic = "split"
        elif self.ticks <= 35:
            tactic = "rush"
        else:
            tactic = "default"

        fake_effect = {
            "fake_or_rotate": bool(visited_other),
            "fake_entry_site": None,
            "fake_entry_tick": None,
            "final_site_entry_tick": self.first_site_tick.get(final_site),
            "defenders_at_fake_entry": 0,
            "defenders_at_final_entry": 0,
            "defenders_displaced_from_final": 0,
            "fake_effect_score": 0.0,
            "fake_effect_rate": 0.0,
        }
        if visited_other:
            other_sites = [site for site in self.visited_sites if site != final_site]
            bait_site = min(
                other_sites,
                key=lambda site: self.first_site_tick.get(site, 10**9),
            )
            bait_tick = self.first_site_tick.get(bait_site)
            final_tick = self.first_site_tick.get(final_site)
            bait_counts = self.defender_site_history.get(int(bait_tick or 0), {})
            final_counts = self.defender_site_history.get(int(final_tick or self.ticks), {})
            before = int(bait_counts.get(final_site, 0))
            after = int(final_counts.get(final_site, 0))
            displaced = max(0, before - after)
            fake_effect.update({
                "fake_entry_site": bait_site,
                "fake_entry_tick": bait_tick,
                "defenders_at_fake_entry": before,
                "defenders_at_final_entry": after,
                "defenders_displaced_from_final": displaced,
                "fake_effect_score": float(displaced),
                "fake_effect_rate": displaced / before * 100 if before else 0.0,
            })
        return {
            "attacker_strategy": tactic,
            "final_attack_site": final_site,
            "visited_sites": sorted(self.visited_sites),
            "max_attacker_spread": self.max_spread,
            **fake_effect,
        }
