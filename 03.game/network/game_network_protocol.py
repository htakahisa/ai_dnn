"""Bounded JSON messages used by the network match transport."""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 4
MAX_MESSAGE_BYTES = 64_000
VALID_TEAMS = ("A", "D")
VALID_ABILITIES = {"SMOKE", "FLASH", "RECON", "RAMP", "DANCE", "ASH"}
VALID_ULTIMATES = {"RAID", "ESCAPE", "MONITOR", "TUNNEL", "NEON", "BALEMOON", "SERENADE"}
VALID_FACINGS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


class ProtocolError(ValueError):
    """Raised when a peer sends an invalid or oversized message."""


def encode_message(message: dict[str, Any]) -> bytes:
    if not isinstance(message, dict):
        raise ProtocolError("A protocol message must be a JSON object")
    raw = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Protocol message exceeds MAX_MESSAGE_BYTES")
    return raw + b"\n"


def decode_message(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Protocol message exceeds MAX_MESSAGE_BYTES")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Message is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("A protocol message must be a JSON object")
    return value


def validate_hello(message: dict[str, Any]) -> str:
    if message.get("type") != "hello":
        raise ProtocolError("The first message must be hello")
    if message.get("version") != PROTOCOL_VERSION:
        raise ProtocolError("Client protocol version is not supported")
    requested = message.get("team", "auto")
    if requested not in ("auto", *VALID_TEAMS):
        raise ProtocolError("team must be auto, A, or D")
    return requested


def validate_roster_selection(value: Any) -> dict[str, Any]:
    """Validate one player's roster and match labels from the network UI."""
    if not isinstance(value, dict):
        raise ProtocolError("hello.selection must be an object")
    roster = value.get("roster")
    if (not isinstance(roster, list) or len(roster) != 5
            or any(not isinstance(name, str) or not name or len(name) > 80 for name in roster)
            or len(set(roster)) != 5):
        raise ProtocolError("selection.roster must contain five distinct character names")
    clean: dict[str, Any] = {"roster": roster}
    for key in ("spike_holder_name", "igl_name"):
        name = value.get(key)
        if not isinstance(name, str) or name not in roster:
            raise ProtocolError(f"selection.{key} must be a member of selection.roster")
        clean[key] = name
    name = value.get("team_name")
    if name is not None and (not isinstance(name, str) or len(name) > 80):
        raise ProtocolError("selection.team_name must be a short string or null")
    clean["team_name"] = name
    return clean


def validate_command(message: dict[str, Any], expected_sequence: int) -> dict[str, Any]:
    if message.get("type") != "command":
        raise ProtocolError("Expected a command message")
    if message.get("sequence") != expected_sequence:
        raise ProtocolError(f"Expected command sequence {expected_sequence}")
    actions = message.get("actions")
    if not isinstance(actions, list) or len(actions) > 32:
        raise ProtocolError("actions must be a list containing at most 32 items")
    clean_actions = []
    for action in actions:
        if not isinstance(action, dict):
            raise ProtocolError("Each action must be an object")
        kind = action.get("kind")
        name = action.get("character")
        if not isinstance(name, str) or not name or len(name) > 82:
            raise ProtocolError(f"{kind}.character must be a non-empty name")
        if kind in {"move", "facing", "plant", "collect", "hold"}:
            if kind == "move":
                clean_actions.append({"kind": kind, "character": name,
                                      "target": _position(action.get("target"), "move.target")})
            elif kind == "facing":
                direction = action.get("direction")
                if direction not in VALID_FACINGS:
                    raise ProtocolError("facing.direction is invalid")
                clean_actions.append({"kind": kind, "character": name,
                                      "direction": direction})
            elif kind in {"plant", "collect"}:
                enabled = action.get("enabled")
                if type(enabled) is not bool:
                    raise ProtocolError(f"{kind}.enabled must be boolean")
                clean_actions.append({"kind": kind, "character": name,
                                      "enabled": enabled})
            else:
                clean_actions.append({"kind": kind, "character": name})
        elif kind in {"ability", "ultimate"}:
            payload = action.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError(f"{kind}.payload must be an object")
            clean_payload = {}
            if kind == "ability":
                ability = str(payload.get("ability", "")).upper()
                if ability not in VALID_ABILITIES:
                    raise ProtocolError("ability.payload.ability is invalid")
                clean_payload["ability"] = ability
                if "target" in payload:
                    clean_payload["target"] = _position(payload["target"], "ability.payload.target")
                if "target_name" in payload:
                    target_name = payload["target_name"]
                    if not isinstance(target_name, str) or not target_name or len(target_name) > 80:
                        raise ProtocolError("ability.payload.target_name is invalid")
                    clean_payload["target_name"] = target_name
            else:
                ultimate = str(payload.get("ultimate", "")).upper()
                if ultimate not in VALID_ULTIMATES:
                    raise ProtocolError("ultimate.payload.ultimate is invalid")
                clean_payload["ultimate"] = ultimate
                if "target" in payload:
                    clean_payload["target"] = _position(payload["target"], "ultimate.payload.target")
                if "facing" in payload:
                    direction = payload["facing"]
                    if direction not in VALID_FACINGS:
                        raise ProtocolError("ultimate.payload.facing is invalid")
                    clean_payload["facing"] = direction
            clean_actions.append({"kind": kind, "character": name,
                                  "payload": clean_payload})
        else:
            raise ProtocolError(f"Unsupported action kind: {kind}")
    return {"sequence": expected_sequence, "actions": clean_actions}


def _position(value: Any, field: str) -> list[int]:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(component) is not int or abs(component) > 10_000
                   for component in value)):
        raise ProtocolError(f"{field} must be a pair of integer coordinates")
    return value

