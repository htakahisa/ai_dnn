"""Independent season saves, with a stable directory for each user team."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil

from realtime_season import DEFAULT_SAVE_PATH, SeasonSaveError, SeasonStore, new_season


@dataclass(frozen=True)
class SeasonProfile:
    name: str
    path: Path
    legacy: bool = False


class SeasonProfiles:
    def __init__(self, directory=None):
        self.directory = Path(directory or Path(DEFAULT_SAVE_PATH).parent / "teams").resolve()
        self.legacy_path = Path(DEFAULT_SAVE_PATH).resolve() if directory is None else self.directory.parent / "save.json"

    def list(self):
        paths = list(self.directory.glob("*/save.json"))
        if self.legacy_path.is_file():
            paths.append(self.legacy_path)
        profiles = []
        for path in paths:
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                name = data["team_name"]
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("missing team name")
            except (OSError, ValueError, KeyError, TypeError):
                name = f"読込エラー: {path.parent.name}"
            profiles.append(SeasonProfile(name, path, path == self.legacy_path))
        return sorted(profiles, key=lambda profile: (profile.name.casefold(), str(profile.path)))

    def create(self, name):
        state = new_season().with_team_name(name)
        name = state.team_name
        if any(profile.name.casefold() == name.casefold() for profile in self.list()):
            raise SeasonSaveError("同じ名前のチームが既にあります。別の名前を入力してください。")
        # A prefix handles Windows device names; the hash distinguishes names
        # that become identical after replacing unsupported path characters.
        slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(" .") or "team"
        digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:12]
        directory = self.directory / f"team-{slug}-{digest}"
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise SeasonSaveError("このチームの保存フォルダが既にあります。別の名前を入力してください。") from exc
        store = SeasonStore(directory / "save.json")
        try:
            store.save(state)
        except (OSError, SeasonSaveError):
            # Only remove an empty folder. Keep any successfully committed save.
            if not any(directory.iterdir()):
                directory.rmdir()
            raise
        return store, state

    def load(self, profile):
        self._validate(profile)
        if not profile.path.is_file():
            raise SeasonSaveError("選択したチームのセーブが見つかりません。")
        store = SeasonStore(profile.path)
        return store, store.load_or_create()

    def _validate(self, profile):
        path = profile.path
        if profile.legacy and path == self.legacy_path and not path.is_symlink():
            return
        if (path.name != "save.json" or path.parent.parent != self.directory
                or path.is_symlink() or path.parent.is_symlink()
                or path.resolve().parent.parent != self.directory):
            raise SeasonSaveError("チームの保存先が不正です。")

    def delete(self, profile):
        self._validate(profile)
        if profile.legacy:
            raise SeasonSaveError("旧形式のセーブは保護されています。削除する場合はファイルを直接管理してください。")
        # Resolve and verify the exact target before recursive deletion.
        target = profile.path.parent.resolve()
        if target.parent != self.directory or target == self.directory:
            raise SeasonSaveError("チームの保存先が不正です。")
        shutil.rmtree(target)
