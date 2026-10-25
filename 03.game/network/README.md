# Network match foundation

This directory contains the isolated network layer and first playable two-client path.
The transport uses one persistent TCP connection per client and newline-framed,
bounded JSON messages. At one game tick per second, this is expected to be
adequate for local and early internet play.

## Current scope

`run_network_server.py` accepts one attacker and one defender, advances a
headless `VisualFPSBattle` every second through `game_network_engine.py`, and
returns a separate observation to each team after every tick. Each client sends
orders whenever the player changes them. The server applies accepted orders to
the authoritative controller immediately; the next scheduled tick uses those
orders. Client messages cannot provide game state or combat outcomes.

`game_network_controller.py` applies move and hold actions to the existing
path-following controller. The server checks character team ownership and
board bounds before applying those orders. `run_network_client.py` first
opens a network-only selection screen. Each client chooses one five-character
roster from an existing team preset or selects five characters manually with
`Original (custom roster)`, plus its match team (A, D, or auto) and the server
IP/host and port. A client never chooses its opponent's roster. The server uses the roster chosen
by its Team A client for the attacker roster and the roster chosen by its Team
D client for the defender roster. If both rosters contain the same character,
the server rejects the match and asks the clients to choose different rosters.
After both clients connect, the client uses the same
`VisualFPSBattle` window, character panels, selection, movement, facing,
plant/collect buttons, ability buttons, and ultimate controls as `run_game.py`.
The client sends each input to the server, and the server executes abilities
and ultimates against its authoritative match. The client renders only the
characters present in its team-filtered observation. Network playback controls
that would simulate the match locally are replaced by a server tick/status bar.
The game continues to advance once per second while a player chooses an order.

## Local transport debugging

Start in `03.game/network/` after changing named constants if needed. Open three
PowerShell terminals, each with `network/` as its working directory.

```powershell
python run_network_server.py
```

The default server listens on `0.0.0.0:8765`, so remote clients can reach the
server machine through its network interface. Allow the chosen port through the
server machine's firewall and enter that machine's reachable IP/host and port
in each client's roster selection screen. For local debugging, clients can
enter `127.0.0.1`. In two other terminals, start one client window per process:

```powershell
python run_network_client.py
```

The IP/host and port fields default to `SERVER_HOST` and `SERVER_PORT` from
`game_network_client.py`; each client can override those values before joining.

The server assigns the first `auto` client to team A and the second to team D.
Command sequence numbers prevent duplicate or stale orders. Server tick
sequence numbers label the observation stream. The server filters enemy
information separately for each client. Existing local match execution remains
on its original path.


