import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_champions_v2.tools.registry_compatibility import unused_team_additions, registry_change_allowed
from ghost_champions_v2.rl import training


OLD = '''import os
def _build_team_ai(key):
    normalized = str(key or "default").strip().lower()
    if normalized == "gc":
        return "gc-original"
    return "default"
'''
ADDITION = '''    if normalized in {"toru_ai_v4", "toru ai v4"}:
        from toruAI_v4.tv4_game_controller import ToruV4GameDefenderController
        return ToruV4GameDefenderController()
'''
NEW = OLD.replace('    return "default"', ADDITION + '    return "default"')


class RegistryCompatibilityTests(unittest.TestCase):
    def test_unselected_factory_addition_is_safe_without_importing_it(self):
        self.assertTrue(unused_team_additions(OLD, NEW, {"gc"}))

    def test_active_alias_side_effects_and_changes_elsewhere_are_rejected(self):
        variants = [
            NEW.replace('"toru_ai_v4"', '"gc"'),
            NEW.replace('normalized in {"toru_ai_v4", "toru ai v4"}', 'probe()'),
            NEW.replace('import os', 'import sys'),
            NEW.replace('"gc-original"', '"changed"'),
            NEW.replace('    return "default"', '    return "changed"'),
            NEW.replace('    if normalized in', '    normalized = "toru_ai_v4"\n    if normalized in'),
            NEW.replace('        return ToruV4GameDefenderController()',
                        '        return ToruV4GameDefenderController()\n    else:\n        return "changed"'),
        ]
        for source in variants:
            with self.subTest(source=source):
                self.assertFalse(unused_team_additions(OLD, source, {"gc"}))

    def test_modifying_an_existing_unused_branch_is_not_an_addition(self):
        self.assertFalse(unused_team_additions(NEW, NEW.replace('ToruV4GameDefenderController()', 'other()'), {"gc"}))

    def test_proof_checks_both_hashes_and_keeps_default_validation_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "run_game.py"
            path.write_bytes(NEW.encode())
            old_digest = hashlib.sha256(OLD.encode()).hexdigest()
            new_digest = hashlib.sha256(NEW.encode()).hexdigest()
            proof = dict(kind="unused_team_additions", previous_sources={old_digest: OLD})
            self.assertTrue(registry_change_allowed(proof, path, old_digest, new_digest, {"gc"}))
            self.assertFalse(registry_change_allowed(proof, path, old_digest, "wrong", {"gc"}))
            self.assertFalse(registry_change_allowed(proof, path, "wrong", new_digest, {"gc"}))
            parent = dict(schema_hash=training.SCHEMA_HASH, verification=False,
                          config={}, frozen_inputs={str(path): old_digest})
            frozen = {str(path): new_digest}
            with patch.object(training, "ROOT", root), patch.object(training, "OPPONENTS", {"GC": ("GC", "gc")}):
                with self.assertRaises(ValueError):
                    training.validate_evaluation_change(parent, {}, frozen)
                training.validate_evaluation_change(parent, {}, frozen, runtime_compatibility=proof)
                # The new run freezes the current bytes just as before.
                path.write_bytes(NEW.replace('"gc-original"', '"changed"').encode())
                with self.assertRaises(RuntimeError):
                    training.assert_frozen(frozen)
                self.assertFalse(registry_change_allowed(proof, path, old_digest, new_digest, {"gc"}))


if __name__ == "__main__":
    unittest.main()
