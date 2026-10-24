"""Let the frozen ConCon route teacher drive real GC statistics and abilities.

Aliases satisfy the teacher's roster-slot lookup, not the student's observation.
Neither route goals nor teacher-private state are copied into the demonstrations.
"""


class AliasCharacter:
    def __init__(self, char, name):
        self.source, self.name = char, name

    def __getattr__(self, name):
        return getattr(self.source, name)


class AliasView:
    def __init__(self, view, roster_name="Gorigons"):
        from party_presets import get_preset
        roster = get_preset(roster_name)
        self._real = view
        own = [c for c in view.chars if c.team == "A"]
        if len(own) not in (0, 5):
            raise ValueError("ConCon teaching requires five roster slots")
        self.characters = {c.name:AliasCharacter(c, name) for c,name in zip(own, roster.players)}
        self.chars = [self.characters[c.name] if c.team == "A" else c for c in view.chars]
        self.reverse = {c.name:c.source.name for c in self.characters.values()}

    def __getattr__(self, name):
        return getattr(self._real, name)


class RosterTeacher:
    def __init__(self, teacher, roster_name="Gorigons"):
        self.inner = teacher
        self.roster_name = roster_name

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def set_game(self, game):
        self.view = AliasView(game, self.roster_name)
        self.inner.set_game(self.view)

    def reset_round(self):
        self.inner.reset_round()

    def decide_move(self, char, state):
        copied = dict(state, chars=self.view.chars)
        command = self.inner.decide_move(self.view.characters[char.name], copied)
        if isinstance(command, tuple) and len(command) == 2 and isinstance(command[1], dict):
            payload = dict(command[1])
            if "target_name" in payload:
                payload["target_name"] = self.view.reverse.get(payload["target_name"], payload["target_name"])
            return command[0], payload
        return command
