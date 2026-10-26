"""Small asyncio client for the network match protocol."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any

try:  # Support both package imports and scripts using this directory directly.
    from .game_network_protocol import (
        MAX_MESSAGE_BYTES, PROTOCOL_VERSION, ProtocolError,
        decode_message, encode_message,
    )
except ImportError:  # pragma: no cover - direct script execution
    from game_network_protocol import (
        MAX_MESSAGE_BYTES, PROTOCOL_VERSION, ProtocolError,
        decode_message, encode_message,
    )

# Normal local-debug settings. Use 127.0.0.1 for two clients on one machine.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
REQUESTED_TEAM = "auto"


def get_private_lan_ipv4() -> str | None:
    """Return a local 192.168.x.x address when hostname lookup exposes one."""
    try:
        addresses = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return None
    for address in addresses:
        candidate = address[4][0]
        try:
            parsed = ipaddress.IPv4Address(candidate)
        except ipaddress.AddressValueError:
            continue
        if parsed in ipaddress.ip_network("192.168.0.0/16"):
            return str(parsed)
    return None


class NetworkMatchClient:
    def __init__(self, host: str = SERVER_HOST, port: int = SERVER_PORT,
                 team: str = REQUESTED_TEAM, selection: dict[str, Any] | None = None):
        self.host = host
        self.port = int(port)
        self.requested_team = team
        self.selection = dict(selection or {})
        self.team: str | None = None
        self.command_sequence = 1
        self._send_lock: asyncio.Lock | None = None
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def connect(self) -> dict[str, Any]:
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port, limit=MAX_MESSAGE_BYTES + 1,
        )
        self.writer.write(encode_message({
            "type": "hello", "version": PROTOCOL_VERSION,
            "team": self.requested_team, "selection": self.selection,
        }))
        await self.writer.drain()
        joined = await self.receive()
        if joined.get("type") != "joined":
            raise ProtocolError("Server did not confirm the client session")
        self.team = joined["team"]
        self.command_sequence = int(joined.get("command_sequence", 1))
        return joined

    async def send_actions(self, actions: list[dict[str, Any]]) -> None:
        if self.writer is None:
            raise ConnectionError("Client is not connected")
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        async with self._send_lock:
            self.writer.write(encode_message({
                "type": "command", "sequence": self.command_sequence, "actions": actions,
            }))
            await self.writer.drain()
            self.command_sequence += 1

    async def receive(self) -> dict[str, Any]:
        if self.reader is None:
            raise ConnectionError("Client is not connected")
        raw = await self.reader.readline()
        if not raw:
            raise ConnectionError("Server disconnected")
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ProtocolError("Server message exceeds MAX_MESSAGE_BYTES")
        return decode_message(raw.rstrip(b"\r\n"))

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
            self.writer = None
            self.reader = None

