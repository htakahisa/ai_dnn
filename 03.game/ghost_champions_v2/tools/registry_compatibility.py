"""Prove that run_game.py only gained branches unused by this training run."""
import ast
import hashlib
from pathlib import Path
import subprocess


def unused_team_additions(before, after, active_keys):
    """Allow only new, unreachable top-level factory branches; fail closed."""
    try:
        old, new = ast.parse(before), ast.parse(after)
    except (SyntaxError, ValueError):
        return False
    factories = [[node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_build_team_ai"]
                 for tree in (old, new)]
    if any(len(nodes) != 1 for nodes in factories):
        return False
    original, current = (nodes[0] for nodes in factories)
    active = {key.strip().lower() for key in active_keys}
    normalization = ast.parse('normalized = str(key or "default").strip().lower()').body[0]
    if (not original.body or ast.dump(original.body[0]) != ast.dump(normalization)
            or any(isinstance(node, ast.Name) and node.id == "normalized" and isinstance(node.ctx, ast.Store)
                   for statement in original.body[1:] for node in ast.walk(statement))):
        return False

    def unused_branch(node):
        if not isinstance(node, ast.If) or node.orelse:
            return False
        test = node.test
        if (not isinstance(test, ast.Compare) or len(test.ops) != 1
                or not isinstance(test.left, ast.Name) or test.left.id != "normalized"):
            return False
        right = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq):
            values = [right]
        elif isinstance(test.ops[0], ast.In) and isinstance(right, (ast.Set, ast.Tuple, ast.List)):
            values = right.elts
        else:
            return False
        return bool(values) and all(isinstance(value, ast.Constant)
                                   and isinstance(value.value, str)
                                   and value.value not in active for value in values)

    # The normalization assignment and every original statement must be identical.
    kept, index, added = [], 0, 0
    for statement in current.body:
        if index < len(original.body) and ast.dump(statement) == ast.dump(original.body[index]):
            kept.append(statement)
            index += 1
        elif index >= 1 and unused_branch(statement):
            added += 1
        else:
            return False
    if index != len(original.body) or not added:
        return False
    current.body = kept
    return ast.dump(old) == ast.dump(new)


def git_source(root, path, expected):
    """Find the exact frozen bytes, including working-tree CRLF, in Git."""
    root, path = Path(root), Path(path)
    prefix = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-prefix"], text=True).strip()
    relative = path.relative_to(root).as_posix()
    revisions = subprocess.check_output(
        ["git", "-C", str(root), "log", "--format=%H", "--", relative], text=True).splitlines()
    for revision in revisions:
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:{prefix}{relative}"])
        for candidate in (source, source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")):
            if hashlib.sha256(candidate).hexdigest() == expected:
                return candidate.decode("utf-8-sig")
    raise ValueError(f"Cannot find the exact frozen {path.name} in Git history")


def registry_change_allowed(proof, path, old_digest, new_digest, active_keys):
    if not proof or proof.get("kind") != "unused_team_additions":
        return False
    source = proof.get("previous_sources", {}).get(old_digest)
    if not isinstance(source, str) or hashlib.sha256(source.encode("utf-8")).hexdigest() != old_digest:
        return False
    current = Path(path).read_bytes()
    return (hashlib.sha256(current).hexdigest() == new_digest
            and unused_team_additions(source, current.decode("utf-8-sig"), active_keys))


def build_registry_proof(root, manifests, frozen, active_keys):
    path = str((Path(root) / "run_game.py").resolve())
    current = frozen.get(path)
    previous = {manifest["frozen_inputs"].get(path) for manifest in manifests}
    proof = dict(kind="unused_team_additions", previous_sources={})
    for digest in sorted(previous - {current, None}):
        proof["previous_sources"][digest] = git_source(root, path, digest)
        if not registry_change_allowed(proof, path, digest, current, active_keys):
            raise ValueError("run_game.py changes are not limited to unused team additions")
    if not proof["previous_sources"]:
        raise ValueError("No run_game.py registry change to recover")
    return proof
