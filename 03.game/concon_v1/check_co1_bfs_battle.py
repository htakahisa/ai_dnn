"""Run real pre-plant battles with BFS movement and export an HTML replay."""

import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.check_co1_stuck import StopTracker, print_event, print_timeout, snapshot
from concon_v1.co1_attacker_common import ACTION_PLANT, ACTION_WAIT, CARDINAL_MOVES
from concon_v1.co1_attacker_scenarios import SCENARIOS, get_scenario
from concon_v1.co1_battle_training import BattleRouteEnv, OPPONENTS
from game_core import ROUND_DURATION_TICKS


def bfs_action(position, route, mask):
    """Follow BFS distances while respecting synchronization and ally yielding."""
    if mask[ACTION_PLANT]:
        return ACTION_PLANT
    if mask[ACTION_WAIT] and int(route.distance_map[tuple(position)]) == 0:
        return ACTION_WAIT
    moves = np.flatnonzero(mask[:4]).tolist()
    if not moves:
        return ACTION_WAIT
    return min(moves, key=lambda action: (
        int(route.distance_map[position[0] + CARDINAL_MOVES[action][0],
                               position[1] + CARDINAL_MOVES[action][1]]), action))


def capture_frame(env):
    shots = []
    for shot in getattr(env.game, "last_shots", []):
        shots.append({"shooter": shot["shooter"].name, "target": shot["target"].name,
                      "from": list(map(int, shot["shooter"].pos)),
                      "to": list(map(int, shot["target"].pos)),
                      "hit": bool(shot["hit"]), "damage": float(shot["damage"])})
    return {"tick": env.elapsed_ticks, "shots": shots,
            "planted": bool(env.game.is_planted),
            "spike_pos": list(map(int, env.game.spike_pos)) if env.game.spike_pos is not None else None,
            "actors": [{"name": char.name, "team": char.team, "pos": list(map(int, char.pos)),
                        "hp": float(char.hp), "alive": bool(char.is_alive),
                        "carrier": bool(char.has_spike)} for char in env.game.chars]}


def run_battle(scenario, opponent, seed, max_ticks, stuck_ticks):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env = BattleRouteEnv(seed, [opponent], map_name=scenario)
    env.game.round_timer = max_ticks
    original_choose = env.route_controller._choose_policy_action

    def choose(char, observation, mask):
        # Choose inside the real decision, using its current route and perception.
        action = bfs_action(tuple(map(int, char.pos)), env.route_controller._routes[char.name], mask)
        previous = env.forced_actions
        env.forced_actions = [ACTION_WAIT] * len(env.attackers)
        env.forced_actions[env.attacker_indices[char.name]] = action
        try:
            return original_choose(char, observation, mask)
        finally:
            env.forced_actions = previous

    env.route_controller._choose_policy_action = choose
    tracker = StopTracker(stuck_ticks)
    frames = [capture_frame(env)]
    deaths = []
    first_stage_ticks = {}
    final = []
    while not env.done and env.elapsed_ticks < max_ticks:
        before = {char.name: bool(char.is_alive) for char in env.game.chars}
        _, _, _, _, masks, _ = env.step()
        frame = capture_frame(env)
        frames.append(frame)
        final = snapshot(env, masks, "battle", env.actions)
        for actor in final:
            if actor["alive"]:
                first_stage_ticks.setdefault(actor["stage"], env.elapsed_ticks)
        for actor in frame["actors"]:
            if before.get(actor["name"]) and not actor["alive"]:
                route_state = next((a for a in final if a["name"] == actor["name"]), None)
                deaths.append({**actor, "tick": env.elapsed_ticks,
                               "route": route_state,
                               "incoming_shots": [s for s in frame["shots"] if s["target"] == actor["name"]]})
        if not env.success:
            tracker.update(env.elapsed_ticks, final)
    reason = ("planted" if env.success else "time_expired" if env.game.round_timer <= 0
              else "attacker_eliminated" if not any(env.alive) else "defender_eliminated")
    return {"seed": seed, "opponent": opponent, "ticks": env.elapsed_ticks,
            "end_reason": reason, "planted": bool(env.success), "deaths": deaths,
            "events": tracker.events, "final": final, "first_stage_ticks": first_stage_ticks,
            "frames": frames}


def write_replay(path, report):
    # Escape '<' so serialized strings cannot terminate the script element.
    data = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    path.write_text(REPLAY_HTML.replace("__REPORT__", data), encoding="utf-8")


REPLAY_HTML = r'''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>ConCon BFS Battle Replay</title>
<style>body{background:#171b23;color:#eee;font:15px sans-serif;margin:24px}
button,select,input{margin:6px}canvas{display:block;max-width:100%;background:#252c38}
pre{white-space:pre-wrap}label{display:block}</style>
<h1>ConCon BFS Battle Replay</h1>
<p>青=A / 赤=D / 黄枠=スパイク所持 / ×=死亡 / 黄線=命中 / 灰線=外れ。座標は(row, col)、左上=(0, 0)。</p>
<select id="trial"></select><button id="play">再生 / 停止</button>
<button id="prev">前tick</button><button id="next">次tick</button>
<label>tick <input id="tick" type="range" min="0" value="0" style="width:65%"></label>
<div id="status"></div><canvas id="map"></canvas><pre id="detail"></pre>
<script>
const report=__REPORT__, trial=document.querySelector('#trial'), slider=document.querySelector('#tick');
const canvas=document.querySelector('#map'), ctx=canvas.getContext('2d'), size=18;
const text=document.querySelector('#detail'), status=document.querySelector('#status');
let timer=null;
report.trials.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}: ${t.opponent} seed=${t.seed} ${t.end_reason}`;trial.append(o)});
canvas.width=report.grid[0].length*size;canvas.height=report.grid.length*size;
function draw(){
 const t=report.trials[trial.value],f=t.frames[+slider.value];slider.max=t.frames.length-1;
 ctx.clearRect(0,0,canvas.width,canvas.height);
 report.grid.forEach((row,r)=>row.forEach((v,c)=>{ctx.fillStyle=v===1?'#657080':v===2?'#435d38':'#252c38';ctx.fillRect(c*size,r*size,size-1,size-1)}));
 Object.entries(report.waypoints).forEach(([marker,points])=>points.forEach(([r,c])=>{ctx.fillStyle='#bbc1d0';ctx.font='11px sans-serif';ctx.fillText(marker,c*size+4,r*size+12)}));
 if(f.spike_pos){const [r,c]=f.spike_pos;ctx.fillStyle='#ffe45e';ctx.fillRect(c*size+5,r*size+5,8,8)}
 f.shots.forEach(s=>{ctx.strokeStyle=s.hit?'#ffe45e':'#888';ctx.beginPath();ctx.moveTo((s.from[1]+.5)*size,(s.from[0]+.5)*size);ctx.lineTo((s.to[1]+.5)*size,(s.to[0]+.5)*size);ctx.stroke()});
 f.actors.forEach(a=>{const [r,c]=a.pos,x=(c+.5)*size,y=(r+.5)*size;ctx.fillStyle=a.team==='A'?'#58a6ff':'#ff667a';ctx.globalAlpha=a.alive?1:.45;
 if(a.alive){ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();if(a.carrier){ctx.strokeStyle='#ffe45e';ctx.stroke()}}
 else{ctx.strokeStyle=ctx.fillStyle;ctx.beginPath();ctx.moveTo(x-5,y-5);ctx.lineTo(x+5,y+5);ctx.moveTo(x+5,y-5);ctx.lineTo(x-5,y+5);ctx.stroke()}ctx.globalAlpha=1});
 status.textContent=`${report.map} | tick=${f.tick}/${t.ticks} | ${t.end_reason} | planted=${f.planted}`;
 const events=t.events.filter(e=>e.start_tick<=f.tick && f.tick<=e.end_tick);
 text.textContent=f.actors.map(a=>`${a.team} ${a.name} (${a.pos.join(', ')}) HP=${a.hp} ${a.alive?'':'DEAD'}`).join('\n')
 +'\n\nSHOTS\n'+f.shots.map(s=>`${s.shooter} → ${s.target} ${s.hit?'HIT':'MISS'} damage=${s.damage}`).join('\n')
 +'\n\nDEATHS\n'+t.deaths.filter(d=>d.tick<=f.tick).map(d=>`tick=${d.tick} ${d.name} (${d.pos.join(', ')}) stage=${d.route?.stage||'-'}`).join('\n')
 +'\n\nSTOPS\n'+events.map(e=>`${e.name}: ${e.reason} ticks=${e.start_tick}..${e.end_tick}`).join('\n');
}
function step(delta){slider.value=Math.max(0,Math.min(+slider.max,+slider.value+delta));draw()}
trial.onchange=()=>{slider.value=0;slider.max=report.trials[trial.value].frames.length-1;draw()};slider.oninput=draw;
document.querySelector('#prev').onclick=()=>step(-1);document.querySelector('#next').onclick=()=>step(1);
document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else{if(+slider.value===+slider.max)slider.value=0;timer=setInterval(()=>{step(1);if(+slider.value===+slider.max){clearInterval(timer);timer=null}},180)}};
slider.max=report.trials[0].frames.length-1;draw();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-map", "--map", choices=SCENARIOS, default="A1", dest="map_name")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=["gc_v1"])
    parser.add_argument("--rounds", type=int, default=3, help="rounds per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=ROUND_DURATION_TICKS)
    parser.add_argument("--stuck-ticks", type=int, default=5)
    parser.add_argument("--output", type=Path, help="output JSON path; also writes an HTML replay")
    args = parser.parse_args()
    if min(args.rounds, args.max_ticks, args.stuck_ticks) < 1:
        parser.error("--rounds, --max-ticks and --stuck-ticks must be positive")
    torch.set_num_threads(1)
    scenario = get_scenario(args.map_name)
    trials = []
    for opponent in dict.fromkeys(args.opponents):
        for trial in range(args.rounds):
            result = run_battle(scenario, opponent, args.seed + trial, args.max_ticks, args.stuck_ticks)
            trials.append(result)
            print(f"map={scenario.map_name} opponent={opponent} seed={result['seed']} "
                  f"end={result['end_reason']} ticks={result['ticks']}", flush=True)
            shots = [s for f in result["frames"] for s in f["shots"] if any(
                a["name"] == s["shooter"] and a["team"] == "A" for a in f["actors"])]
            print(f"  attacker_shots={len(shots)} hits={sum(s['hit'] for s in shots)}")
            for death in result["deaths"]:
                stage = death["route"]["stage"] if death["route"] else "-"
                print(f"  DEATH tick={death['tick']} {death['team']} {death['name']} "
                      f"pos={death['pos']} stage={stage}")
            for event in result["events"]:
                print_event(event)
            if result["end_reason"] == "time_expired":
                print_timeout(result)
    summary = {"rounds": len(trials), "avg_ticks": sum(t["ticks"] for t in trials) / len(trials),
               "end_reasons": dict(Counter(t["end_reason"] for t in trials)),
               "death_positions": dict(Counter(f"{d['team']} {d['pos']}" for t in trials for d in t["deaths"]))}
    report = {"map": scenario.map_name, "grid": scenario.grid.tolist(),
              "waypoints": scenario.waypoint_points, "max_ticks": args.max_ticks,
              "stuck_ticks": args.stuck_ticks, "summary": summary, "trials": trials}
    output = args.output or Path(__file__).resolve().parent / "data" / "diagnostics" / f"bfs_battle_{scenario.map_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_replay(output.with_suffix(".html"), report)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    print(f"Saved {output.resolve()}\nReplay {output.with_suffix('.html').resolve()}")


if __name__ == "__main__":
    main()
