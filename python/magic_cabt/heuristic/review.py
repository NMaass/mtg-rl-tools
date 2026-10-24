"""Self-contained local replay inspection; no CDN, telemetry or raw HTML labels."""
import json
from pathlib import Path


def render(run, output):
    run, output = Path(run), Path(output)
    games = json.loads((run / "games.json").read_text(encoding="utf-8"))
    payload = []
    for game in games:
        path = run / game.get("replay", "missing")
        frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()] if path.is_file() else []
        payload.append({"game": game, "frames": [f for f in frames if "observation" in f]})
    data = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    html = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Heuristic experiment replay review</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#181818}button,select,input{font:inherit;margin:.3rem;padding:.5rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid #ccc;padding:1rem}table{border-collapse:collapse;width:100%;background:white}td,th{padding:.5rem;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}.selected{font-weight:bold;background:#e7f1e7}details{margin:.7rem 0}#board{min-height:8rem}#position{display:inline-block;min-width:13rem;font-variant-numeric:tabular-nums}</style>
<h1>Heuristic experiment replay review</h1>
<p>Developer heuristic, not a validated expert. Each frame is the acting player's information state; this viewer changes perspective between decisions. Printed card rules do not capture all dynamic effects.</p>
<label>Game <select id="game"></select></label><p id="outcome"></p>
<button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button><span id="position"></span>
<label>Decision <input id="seek" type="range" min="0" value="0"></label>
<h2 id="heading"></h2><pre id="board"></pre><pre id="plan"></pre>
<table><thead><tr><th>Action</th><th>Score</th><th>Module evidence</th></tr></thead><tbody id="actions"></tbody></table>
<details><summary>Jev / warnings</summary><pre id="jev"></pre></details>
<details><summary>Raw recorded frame</summary><pre id="raw"></pre></details>
<script id="data" type="application/json">DATA</script>
<script>
'use strict';const games=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);let gi=0,fi=0,timer=null;
function text(id,value){el(id).textContent=value;}
games.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.textContent=g.game.id+' | '+g.game.agents.join(' vs ')+' | '+g.game.decks.join(' / ');el('game').append(o);});
function draw(){const g=games[gi];if(!g){return;}text('outcome',JSON.stringify(g.game));el('seek').max=Math.max(0,g.frames.length-1);el('seek').value=fi;el('actions').replaceChildren();const f=g.frames[fi];text('position',(fi+1)+' / '+g.frames.length);el('prev').disabled=fi===0;el('next').disabled=fi>=g.frames.length-1;if(!f){['heading','board','plan','jev','raw'].forEach(id=>text(id,'No recorded decisions: '+g.game.status));return;}
const c=f.observation.current,s=f.observation.select,x=f.explanation||{};text('heading','Turn '+c.turnNumber+' / '+c.step+' / seat '+f.player+' / '+s.type);
const name=o=>o.name||(o.ref||{}).name||'Unknown';text('board',(c.players||[]).map(p=>p.name+': '+p.life+' life; hand '+(p.hand||[]).map(name).join(', ')+' ('+p.handCount+' cards)').join('\n')+'\n\nBattlefield: '+(c.battlefield||[]).map(o=>name(o)+' '+o.power+'/'+o.toughness+(o.tapped?' tapped':'')).join(' | '));
text('plan',JSON.stringify({role:x.role,plan:x.plan,selection:f.selected,coverage:x.coverage},null,2));
(s.option||[]).forEach((a,i)=>{const r=document.createElement('tr'),e=(x.options||[]).find(o=>o.index===i);if((f.selected||[]).includes(i))r.className='selected';[i+': '+a.label,e?e.score.toFixed(3):'baseline',(e?e.components:[]).map(k=>k.module+' '+k.score.toFixed(2)+': '+k.reason).join('\n')].forEach(v=>{const td=document.createElement('td');td.textContent=v;td.style.whiteSpace='pre-wrap';r.append(td);});el('actions').append(r);});text('jev',JSON.stringify({jev:x.jev,warnings:x.warnings},null,2));text('raw',JSON.stringify(f,null,2));}
function stop(){if(timer)clearInterval(timer);timer=null;text('play','Play');}
el('game').onchange=e=>{stop();gi=Number(e.target.value);fi=0;draw();};el('seek').oninput=e=>{stop();fi=Number(e.target.value);draw();};el('prev').onclick=()=>{stop();fi=Math.max(0,fi-1);draw();};el('next').onclick=()=>{stop();fi=Math.min(games[gi].frames.length-1,fi+1);draw();};el('play').onclick=()=>{if(timer){stop();return;}text('play','Pause');timer=setInterval(()=>{if(fi>=games[gi].frames.length-1){stop();return;}fi++;draw();},600);};draw();
</script></html>'''.replace('DATA', data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
