"""Build a self-contained HTML page that plays a local MTGO clip next to a
reconstructed "parsed replay" of the canonical states, synced to video time.

Usage:
  python build_replay_viewer.py --bundle <extraction-bundle> --video <local.mp4> \
      --out <review.html>

The video path in the page is stored relative to the HTML file's directory, so
put the .html where that relative path resolves (or pass an absolute path).
Open it in a normal browser (local file playback + seeking works there).
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter


def load_timeline(bundle):
    states = [json.loads(l) for l in open(os.path.join(bundle, "canonical_states.jsonl"), encoding="utf-8")]
    def pl(s, seat, key):
        for p in s["players"]:
            if str(p["seat"]) == seat:
                return p.get(key)
        return None
    rows = []
    for s in states:
        z = s.get("zones") or {}
        bf = z.get("battlefield", [])
        conf = s.get("confidence", {})
        hist = s.get("public_history") or []
        rows.append({
            "t": round((s.get("timestamp_ms") or 0) / 1000, 2),
            "turn": s.get("turn_number"), "phase": s.get("phase"),
            "l1": pl(s, "1", "life"), "l2": pl(s, "2", "life"),
            "h1": pl(s, "1", "handCount"), "h2": pl(s, "2", "handCount"),
            "bf1": sum(1 for c in bf if str(c.get("controller")) == "1"),
            "bf2": sum(1 for c in bf if str(c.get("controller")) == "2"),
            "n1": sorted({c.get("name") for c in bf if str(c.get("controller")) == "1" and c.get("name")}),
            "n2": sorted({c.get("name") for c in bf if str(c.get("controller")) == "2" and c.get("name")}),
            "log": (hist[-1]["text"] if hist else ""),
            "cL": round(conf.get("/players/1/life", 0), 2),
            "cP": round(conf.get("/phase", 0), 2),
        })
    actions = []
    ap = os.path.join(bundle, "observed_actions.jsonl")
    if os.path.exists(ap):
        for l in open(ap, encoding="utf-8"):
            if l.strip():
                a = json.loads(l)
                actions.append({"t": round((a.get("timestamp_ms") or 0) / 1000, 2),
                                "type": a.get("action_type"), "text": a.get("text")})
    return rows, actions


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#14161a;color:#e6e6e6;font:14px/1.45 system-ui,Segoe UI,Arial}
header{padding:8px 14px;background:#0e1013;border-bottom:1px solid #2a2f37}
header b{color:#fff}
.wrap{display:flex;gap:14px;padding:14px;align-items:flex-start;flex-wrap:wrap}
.left{flex:1 1 60%;min-width:340px}
video{width:100%;background:#000;border-radius:6px}
.right{flex:1 1 34%;min-width:300px;max-width:520px}
.panel{background:#1b1f26;border:1px solid #2a2f37;border-radius:8px;padding:12px;margin-bottom:12px}
h3{margin:0 0 8px;font-size:12px;color:#aeb7c2;text-transform:uppercase;letter-spacing:.05em}
.hdr{display:flex;gap:14px;align-items:baseline;margin-bottom:6px}
.hdr .turn{font-size:20px;font-weight:700}
.hdr .phase{font-size:16px;font-weight:600;color:#ffbd6b}
.pl{display:flex;align-items:center;gap:12px;padding:8px;border-radius:8px;margin:6px 0}
.pl.opp{background:#241318;border:1px solid #4a2530}
.pl.loc{background:#122016;border:1px solid #234a30}
.life{font-size:34px;font-weight:800;min-width:64px;text-align:center}
.meta{font-size:12px;color:#9aa4b0}.meta b{color:#e6e6e6;font-size:14px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
.chip{background:#2a2f37;border-radius:4px;padding:1px 6px;font-size:11px}
.log{font-family:ui-monospace,Consolas,monospace;background:#0f1216;border-radius:6px;padding:8px;min-height:20px;font-size:12px}
.tl{height:150px;overflow:auto;border:1px solid #2a2f37;border-radius:6px}
.row{display:flex;gap:8px;padding:3px 8px;cursor:pointer;font-size:12px;border-bottom:1px solid #20242c}
.row:hover{background:#232935}.row.cur{background:#2b3b25}
.row .t{color:#8b95a3;width:52px;flex:none}
.small{color:#8b95a3;font-size:12px}
.note{font-size:12px;color:#c9a86a;background:#241d10;border:1px solid #4a3a1a;border-radius:6px;padding:8px;margin-top:8px}
</style></head><body>
<header><b>Parsed replay vs video</b> &mdash; <span id="ttl"></span> <span class="small">&nbsp;paddle OCR, __PROFILE__</span></header>
<div class="wrap">
 <div class="left">
   <video id="v" controls preload="metadata"></video>
   <div class="panel"><h3>Captured states (click to seek)</h3><div class="tl" id="tl"></div></div>
 </div>
 <div class="right">
   <div class="panel">
     <div class="hdr"><span class="turn" id="turn">turn -</span><span class="phase" id="phase">-</span><span class="small" id="st"></span></div>
     <div class="pl opp"><div class="life" id="l2">-</div><div><div class="meta">opponent (seat 2) &nbsp; hand <b id="h2">-</b> &nbsp; battlefield <b id="bf2">-</b></div><div class="chips" id="n2"></div></div></div>
     <div class="pl loc"><div class="life" id="l1">-</div><div><div class="meta">local (seat 1) &nbsp; hand <b id="h1">-</b> &nbsp; battlefield <b id="bf1">-</b></div><div class="chips" id="n1"></div></div></div>
     <div class="small" id="cf" style="margin-top:6px"></div>
   </div>
   <div class="panel"><h3>Game-log line read</h3><div class="log" id="log">-</div>
     <div class="note" id="actnote"></div></div>
 </div>
</div>
<script>
const D=__PAYLOAD__;
document.getElementById('ttl').textContent=D.title;
const v=document.getElementById('v');v.src=encodeURI(D.video);
const S=D.states;
document.getElementById('actnote').textContent=D.actions.length+' discrete action(s) captured from the log pane. Sparse action capture is expected when the client shows no full play-by-play log; states are dense.';
const tl=document.getElementById('tl');
S.forEach((s,i)=>{const r=document.createElement('div');r.className='row';
  r.innerHTML=`<span class="t">${s.t}s</span><span>T${s.turn??'?'} ${s.phase||'—'}</span><span style="margin-left:auto">${s.l2??'?'} / ${s.l1??'?'}</span>`;
  r.onclick=()=>{v.currentTime=s.t;};tl.appendChild(r);});
const rows=[...tl.children];
function chips(el,names){el.innerHTML='';(names||[]).slice(0,18).forEach(n=>{const c=document.createElement('span');c.className='chip';c.textContent=n;el.appendChild(c);});}
function idxAt(t){let lo=0,hi=S.length-1,r=0;while(lo<=hi){let m=(lo+hi)>>1;if(S[m].t<=t){r=m;lo=m+1}else hi=m-1;}return r;}
let cur=-1;
function upd(){const i=idxAt(v.currentTime);if(i===cur)return;cur=i;const s=S[i];const g=(id,val)=>document.getElementById(id).textContent=(val??'—');
  g('turn','turn '+(s.turn??'—'));g('phase',s.phase||'—');g('st','@'+s.t+'s');
  g('l2',s.l2);g('l1',s.l1);g('h2',s.h2);g('h1',s.h1);g('bf2',s.bf2);g('bf1',s.bf1);
  chips(document.getElementById('n2'),s.n2);chips(document.getElementById('n1'),s.n1);
  document.getElementById('log').textContent=s.log||'—';
  document.getElementById('cf').innerHTML=`field confidence &nbsp; life <b>${s.cL}</b> &nbsp; phase <b>${s.cP}</b>`;
  rows.forEach(r=>r.classList.remove('cur'));if(rows[i]){rows[i].classList.add('cur');rows[i].scrollIntoView({block:'nearest'});}}
v.addEventListener('timeupdate',upd);v.addEventListener('seeked',upd);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="MTGO parsed replay")
    ap.add_argument("--profile", default="")
    args = ap.parse_args()
    rows, actions = load_timeline(args.bundle)
    rel = os.path.relpath(os.path.abspath(args.video), os.path.dirname(os.path.abspath(args.out))).replace(os.sep, "/")
    payload = json.dumps({"states": rows, "actions": actions, "video": rel, "title": args.title}, separators=(",", ":"))
    html = (TEMPLATE.replace("__PAYLOAD__", payload).replace("__TITLE__", args.title)
            .replace("__PROFILE__", args.profile))
    open(args.out, "w", encoding="utf-8").write(html)
    print(f"wrote {args.out}  ({len(rows)} states, {len(actions)} actions)")


if __name__ == "__main__":
    main()
