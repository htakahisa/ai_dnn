"""Authoritative two-client TCP server with one game tick per second."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

try:
    from .game_network_protocol import (
        MAX_MESSAGE_BYTES, PROTOCOL_VERSION, ProtocolError, decode_message,
        encode_message, validate_command, validate_hello, validate_roster_selection,
    )
except ImportError:  # pragma: no cover - direct script execution
    from game_network_protocol import (
        MAX_MESSAGE_BYTES, PROTOCOL_VERSION, ProtocolError, decode_message,
        encode_message, validate_command, validate_hello, validate_roster_selection,
    )

# Change these named settings for another server host or local tick cadence.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
CLIENTS_PER_MATCH = 2
READ_TIMEOUT_SECONDS = 120
TICK_INTERVAL_SECONDS = 1.0

AdvanceMatch = Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]]
SubmitActions = Callable[[str, list[dict[str, Any]]], Awaitable[None] | None]


@dataclass
class _Peer:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    team: str
    selection: dict[str, Any]


@dataclass
class _Room:
    peers: dict[str, _Peer] = field(default_factory=dict)
    command_sequences: dict[str, int] = field(default_factory=lambda: {"A": 1, "D": 1})
    tick_sequence: int = 1
    assignment_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tick_task: asyncio.Task | None = None


class NetworkMatchServer:
    """Tick the game at a fixed cadence and accept orders at any time."""

    def __init__(self, advance_match: AdvanceMatch, submit_actions: SubmitActions,
                 host: str = SERVER_HOST, port: int = SERVER_PORT,
                 initial_observations: dict[str, Any] | None = None,
                 on_match_ready: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
                 observe_match: Callable[[], dict[str, dict[str, Any]]] | None = None):
        self.advance_match = advance_match
        self.submit_actions = submit_actions
        self.host = host
        self.port = int(port)
        self.initial_observations = initial_observations or {}
        self.on_match_ready = on_match_ready
        self.observe_match = observe_match
        self.room = _Room()
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_peer, self.host, self.port,
            limit=MAX_MESSAGE_BYTES + 1,
        )

    async def serve_forever(self) -> None:
        if self.server is None:
            await self.start()
        assert self.server is not None
        async with self.server:
            await self.server.serve_forever()

    async def close(self) -> None:
        if self.room.tick_task is not None:
            self.room.tick_task.cancel()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        for peer in tuple(self.room.peers.values()):
            peer.writer.close()
        self.room.peers.clear()

    async def _read_message(self, reader: asyncio.StreamReader) -> dict[str, Any]:
        raw = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_SECONDS)
        if not raw:
            raise ConnectionError("Peer disconnected")
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ProtocolError("Protocol message exceeds MAX_MESSAGE_BYTES")
        return decode_message(raw.rstrip(b"\r\n"))

    async def _send(self, peer: _Peer, message: dict[str, Any]) -> None:
        peer.writer.write(encode_message(message))
        await peer.writer.drain()

    async def _handle_peer(self, reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter) -> None:
        peer = None
        try:
            hello = await self._read_message(reader)
            requested = validate_hello(hello)
            selection = validate_roster_selection(hello.get("selection"))
            async with self.room.assignment_lock:
                team = self._assign_team(requested)
                peer = _Peer(reader, writer, team, selection)
                self.room.peers[team] = peer
                await self._send(peer, {
                    "type": "joined", "version": PROTOCOL_VERSION, "team": team,
                    "command_sequence": self.room.command_sequences[team],
                    "waiting_for": [side for side in ("A", "D") if side not in self.room.peers],
                })
                if len(self.room.peers) == CLIENTS_PER_MATCH:
                    overlap = (set(self.room.peers["A"].selection["roster"])
                               & set(self.room.peers["D"].selection["roster"]))
                    if overlap:
                        error = {
                            "type": "error",
                            "message": "Both teams selected overlapping characters. Choose a different preset or custom roster.",
                        }
                        for connected in tuple(self.room.peers.values()):
                            await self._send(connected, error)
                            connected.writer.close()
                        self.room.peers.clear()
                        return
                    match_configuration = None
                    try:
                        if self.on_match_ready is not None:
                            match_configuration = self.on_match_ready(
                                {side: self.room.peers[side].selection for side in ("A", "D")}
                            )
                            if isinstance(match_configuration, dict):
                                self.initial_observations = match_configuration.get(
                                    "observations", self.initial_observations,
                                )
                        missing = [side for side in ("A", "D") if side not in self.initial_observations]
                        if missing:
                            raise ValueError(f"Initial observations missing for team(s): {', '.join(missing)}")
                    except Exception as exc:
                        detail = f"{type(exc).__name__}: {exc}"
                        print(f"[SERVER][MATCH_START_ERROR] {detail}", flush=True)
                        error = {"type": "error", "message": f"Server failed to initialize match ({detail})"}
                        for connected in tuple(self.room.peers.values()):
                            try:
                                await self._send(connected, error)
                            except (ConnectionError, OSError, ProtocolError):
                                pass
                            connected.writer.close()
                        self.room.peers.clear()
                        return
                    for connected in self.room.peers.values():
                        await self._send(connected, {
                            "type": "match_ready", "tick_sequence": self.room.tick_sequence,
                            "rosters": (match_configuration or {}).get("rosters"),
                        })
                        await self._send(connected, {
                            "type": "observation", "sequence": self.room.tick_sequence - 1,
                            "observation": self.initial_observations[connected.team],
                        })
                    self._start_tick_task()
            while True:
                message = await self._read_message(reader)
                expected = self.room.command_sequences[peer.team]
                command = validate_command(message, expected)
                result = self.submit_actions(peer.team, command["actions"])
                if asyncio.iscoroutine(result):
                    await result
                self.room.command_sequences[peer.team] = expected + 1
                if self.observe_match is not None:
                    observations = self.observe_match()
                    for team, connected in tuple(self.room.peers.items()):
                        await self._send(connected, {
                            "type": "observation",
                            "sequence": self.room.tick_sequence - 1,
                            "observation": observations[team],
                        })
                await self._send(peer, {
                    "type": "command_ack", "sequence": expected,
                    "tick_sequence": self.room.tick_sequence,
                })
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.TimeoutError):
            pass
        except (ProtocolError, ValueError) as exc:
            try:
                writer.write(encode_message({"type": "error", "message": str(exc)}))
                await writer.drain()
            except (ConnectionError, OSError):
                pass
        finally:
            if peer is not None and self.room.peers.get(peer.team) is peer:
                self.room.peers.pop(peer.team, None)
                if self.room.tick_task is not None:
                    self.room.tick_task.cancel()
                    self.room.tick_task = None
                await self._broadcast({"type": "peer_disconnected", "team": peer.team})
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    def _assign_team(self, requested: str) -> str:
        if len(self.room.peers) >= CLIENTS_PER_MATCH:
            raise ProtocolError("This match already has two connected clients")
        if requested in ("A", "D"):
            if requested in self.room.peers:
                raise ProtocolError(f"Team {requested} is already occupied")
            return requested
        return next(side for side in ("A", "D") if side not in self.room.peers)

    def _start_tick_task(self) -> None:
        if self.room.tick_task is None or self.room.tick_task.done():
            self.room.tick_task = asyncio.create_task(self._tick_loop())

    async def _tick_loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + TICK_INTERVAL_SECONDS
        try:
            while len(self.room.peers) == CLIENTS_PER_MATCH:
                await asyncio.sleep(max(0.0, next_tick - loop.time()))
                if len(self.room.peers) != CLIENTS_PER_MATCH:
                    return
                match_over = False
                tick_failed = False
                try:
                    result = self.advance_match()
                    if asyncio.iscoroutine(result):
                        result = await result
                    if not isinstance(result, dict) or set(result) != {"A", "D"}:
                        raise RuntimeError("advance_match must return separate A and D views")
                    match_over = all(
                        isinstance(result[team], dict) and result[team].get("match_over")
                        for team in ("A", "D")
                    )
                except Exception as exc:
                    tick_failed = True
                    detail = f"{type(exc).__name__}: {exc}"
                    print(f"[SERVER][TICK_SKIPPED] sequence={self.room.tick_sequence} {detail}", flush=True)
                    # A failed simulation tick is skipped. The room and clients
                    # stay connected, and the next scheduled tick still runs.
                    result = None

                if result is not None:
                    for team, peer in tuple(self.room.peers.items()):
                        await self._send(peer, {
                            "type": "observation", "sequence": self.room.tick_sequence,
                            "observation": result[team],
                        })
                        if match_over:
                            await self._send(peer, {"type": "match_over", "sequence": self.room.tick_sequence})
                elif self.observe_match is not None:
                    try:
                        observations = self.observe_match()
                        for team, peer in tuple(self.room.peers.items()):
                            await self._send(peer, {
                                "type": "observation", "sequence": self.room.tick_sequence,
                                "observation": observations[team],
                            })
                    except Exception as exc:
                        print(f"[SERVER][OBSERVATION_ERROR] {type(exc).__name__}: {exc}", flush=True)
                if tick_failed:
                    await self._broadcast({
                        "type": "tick_warning",
                        "message": "A server tick failed and was skipped; the match will continue.",
                    })
                self.room.tick_sequence += 1
                if match_over:
                    return
                next_tick += TICK_INTERVAL_SECONDS
                # Do not emit a burst of catch-up ticks after a slow engine frame.
                if next_tick < loop.time():
                    next_tick = loop.time() + TICK_INTERVAL_SECONDS
        except asyncio.CancelledError:
            raise
        finally:
            if self.room.tick_task is asyncio.current_task():
                self.room.tick_task = None

    async def _broadcast(self, message: dict[str, Any]) -> None:
        for peer in tuple(self.room.peers.values()):
            try:
                await self._send(peer, message)
            except (ConnectionError, OSError):
                pass


if __name__ == "__main__":
    try:
        from .game_network_engine import NetworkGameAdapter
    except ImportError:  # pragma: no cover - direct script execution
        from game_network_engine import NetworkGameAdapter

    match = NetworkGameAdapter()

    def configure_match(selections):
        return match.configure_rosters(selections)

    server = NetworkMatchServer(
        match.advance_match,
        match.submit_actions,
        SERVER_HOST,
        SERVER_PORT,
        initial_observations=match.initial_observations(),
        on_match_ready=configure_match,
        observe_match=match.initial_observations,
    )
    print(f"Network match server listening on {SERVER_HOST}:{SERVER_PORT}")
    print(f"The authoritative game advances every {TICK_INTERVAL_SECONDS:g} second(s).")
    asyncio.run(server.serve_forever())

