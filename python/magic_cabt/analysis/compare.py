"""Build a standalone side-by-side viewer for model analysis records."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter
from datetime import datetime, timezone

from .backfill import backfill_bundle
from .schema import decision_fingerprint


def compare_bundle(bundle_dir, models, out_html, out_json=None,
                   device=None, top_k=5, title=None, progress=None):
    bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
    if not models:
        raise ValueError("at least one model is required")
    summaries = []
    names = set()
    for name, checkpoint in models:
        name = str(name).strip()
        if not name or name in names:
            raise ValueError("model names must be unique and non-empty")
        names.add(name)
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        if not os.path.isfile(checkpoint):
            raise IOError("checkpoint not found: %s" % checkpoint)
        summary = backfill_bundle(
            bundle_dir, checkpoint, device=device, top_k=top_k,
            progress=(lambda done, total, label=name:
                      progress(label, done, total)) if progress else None,
            source="head-to-head")
        summaries.append({
            "name": name,
            "checkpoint": checkpoint,
            "checkpointSha256": _sha256_file(checkpoint),
            "model": summary["model"],
            "scored": summary["scored"],
            "alreadyCached": summary["alreadyCached"],
        })
    report = build_comparison(bundle_dir, summaries, title=title)
    out_html = os.path.abspath(os.path.expanduser(out_html))
    out_json = os.path.abspath(os.path.expanduser(
        out_json or os.path.splitext(out_html)[0] + ".json"))
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    _atomic_write(out_json, json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _atomic_write(out_html, render_comparison_html(report))
    return {"html": out_html, "json": out_json, "comparison": report}


def build_comparison(bundle_dir, model_summaries, title=None):
    decisions_path = os.path.join(bundle_dir, "decisions.jsonl")
    analysis_path = os.path.join(bundle_dir, "analysis.jsonl")
    if not os.path.isfile(decisions_path):
        raise IOError("no decisions.jsonl in %s" % bundle_dir)
    decisions = list(_read_jsonl(decisions_path))
    analyses = list(_read_jsonl(analysis_path)) \
        if os.path.isfile(analysis_path) else []

    models = []
    by_checkpoint = {}
    for index, summary in enumerate(model_summaries):
        info = summary.get("model") or {}
        descriptor = {
            "index": index,
            "name": summary["name"],
            "checkpoint": summary.get("checkpoint"),
            "checkpointSha256": summary.get("checkpointSha256"),
            "checkpointId": info.get("checkpointId"),
            "modelId": info.get("modelId"),
            "trainingState": info.get("trainingState"),
            "transitionExamples": info.get("transitionExamples"),
            "decisionExamples": info.get("decisionExamples"),
            "scored": summary.get("scored", 0),
            "alreadyCached": summary.get("alreadyCached", 0),
        }
        models.append(descriptor)
        if descriptor["checkpointId"]:
            by_checkpoint[descriptor["checkpointId"]] = descriptor

    latest = {}
    for record in analyses:
        fingerprint = record.get("decisionFingerprint")
        checkpoint = (record.get("model") or {}).get("checkpointId")
        if not fingerprint or checkpoint not in by_checkpoint:
            continue
        key = (fingerprint, checkpoint)
        current = latest.get(key)
        if current is None or str(record.get("createdAt") or "") >= \
                str(current.get("createdAt") or ""):
            latest[key] = record

    agreement_counts = Counter()
    prompt_counts = Counter()
    coverage = Counter()
    top1 = Counter()
    reciprocal_rank = Counter()
    rows = []
    for source_index, decision in enumerate(decisions):
        select = _select(decision)
        options = select.get("option") or []
        if not options:
            continue
        selected = _selected_indices(decision)
        chosen = selected[0] if selected and isinstance(selected[0], int) else None
        fingerprint = decision_fingerprint(decision)
        cells, identities = [], []
        for descriptor in models:
            record = latest.get((fingerprint, descriptor["checkpointId"]))
            cell = _model_cell(record, options, chosen)
            cells.append(cell)
            if cell["covered"]:
                name = descriptor["name"]
                coverage[name] += 1
                rank = cell.get("playedRank")
                if rank == 1:
                    top1[name] += 1
                if rank:
                    reciprocal_rank[name] += 1.0 / rank
                identities.append(cell.get("topIdentity"))
            else:
                identities.append(None)
        present = [identity for identity in identities if identity is not None]
        if not present:
            agreement = "unscored"
        elif len(set(present)) == 1 and len(present) == len(models):
            agreement = "all-agree"
        elif len(set(present)) == 1:
            agreement = "partial-agree"
        else:
            agreement = "disagree"
        agreement_counts[agreement] += 1
        prompt = select.get("type") or "UNKNOWN"
        prompt_counts[prompt] += 1
        current = (decision.get("observation") or {}).get("current") or \
            decision.get("current") or {}
        rows.append({
            "row": len(rows) + 1,
            "sourceIndex": source_index,
            "decisionFingerprint": fingerprint,
            "gameId": decision.get("gameId"),
            "matchId": decision.get("matchId"),
            "gameNumber": decision.get("gameNumber"),
            "sequenceNumber": decision.get(
                "sequenceNumber", decision.get("sequence")),
            "turn": current.get("turnNumber"),
            "phase": current.get("phase") or current.get("step"),
            "promptType": prompt,
            "optionCount": len(options),
            "human": {
                "selectedIndices": selected,
                "label": _option_label(options, chosen),
                "canonicalKey": _option_key(options, chosen),
                "identity": _option_identity(options, chosen),
            },
            "agreement": agreement,
            "models": cells,
            "stateSummary": _state_summary(current),
            "options": [_public_option(option) for option in options],
        })

    model_metrics = []
    total = len(rows)
    for descriptor in models:
        name = descriptor["name"]
        count = coverage[name]
        model_metrics.append({
            "name": name,
            "covered": count,
            "coverage": count / total if total else 0.0,
            "playedTop1": top1[name] / count if count else None,
            "playedMRR": reciprocal_rank[name] / count if count else None,
        })
    return {
        "schemaVersion": 1,
        "kind": "magic-model-comparison-v1",
        "title": title or "MTG model head-to-head",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "bundle": {
            "path": os.path.abspath(bundle_dir),
            "summary": _read_json(os.path.join(bundle_dir, "summary.json")) or {},
            "decisionsSha256": _sha256_file(decisions_path),
            "analysisSha256": _sha256_file(analysis_path)
                if os.path.isfile(analysis_path) else None,
        },
        "models": models,
        "metrics": {
            "decisions": total,
            "agreement": dict(sorted(agreement_counts.items())),
            "promptCounts": dict(sorted(prompt_counts.items())),
            "models": model_metrics,
        },
        "rows": rows,
    }


def _model_cell(record, options, chosen_index):
    if not record:
        return {"covered": False, "top": [], "playedRank": None}
    analysis = record.get("analysis") or {}
    top = []
    for row in analysis.get("topK") or []:
        index = row.get("optionIndex")
        canonical = row.get("canonicalGroupKey") or _option_key(options, index)
        identity = ("canonical:" + str(canonical)) \
            if canonical is not None else _option_identity(options, index)
        top.append({
            "optionIndex": index,
            "label": row.get("label") or _option_label(options, index),
            "canonicalKey": canonical,
            "identity": identity,
            "score": row.get("score"),
            "probability": row.get("probability"),
        })
    first = top[0] if top else {}
    human_identity = _option_identity(options, chosen_index)
    canonical_rank = None
    seen = set()
    for row in top:
        identity = row.get("identity")
        if identity in seen:
            continue
        seen.add(identity)
        if identity == human_identity:
            canonical_rank = len(seen)
            break
    raw_rank = analysis.get("chosenRank")
    return {
        "covered": True,
        "topLabel": first.get("label"),
        "topCanonicalKey": first.get("canonicalKey"),
        "topIdentity": first.get("identity"),
        "topProbability": first.get("probability"),
        "playedRank": canonical_rank if canonical_rank is not None else raw_rank,
        "rawPlayedRank": raw_rank,
        "playedScore": analysis.get("chosenScore"),
        "stateValue": analysis.get("value"),
        "latencyMs": record.get("latencyMs"),
        "top": top,
    }


def render_comparison_html(report):
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(report.get("title") or "MTG model head-to-head")
    headers = "".join("<th>%s</th>" % html.escape(model["name"])
                      for model in report.get("models") or [])
    return """<!doctype html><html><head><meta charset='utf-8'>
<title>%(title)s</title><style>
body{font:14px system-ui;background:#111318;color:#eef2f7;margin:0}header{position:sticky;top:0;background:#111318;padding:16px;border-bottom:1px solid #333}main{padding:16px}input,select,button{background:#1b1f27;color:#eef2f7;border:1px solid #444;padding:8px;margin-right:8px}table{border-collapse:collapse;width:100%%;min-width:1000px}th,td{border:1px solid #333;padding:8px;vertical-align:top}th{background:#181c23;position:sticky;top:82px}.disagree{background:#301b1b}.all-agree{background:#183022}.small{color:#9da8b7;font-size:12px}pre{white-space:pre-wrap}</style></head><body>
<header><h2>%(title)s</h2><input id='q' placeholder='Search'><select id='a'><option value=''>All</option><option value='disagree'>Disagreement</option><option value='all-agree'>All agree</option><option value='partial-agree'>Partial</option></select><button id='m'>Human differs</button></header>
<main><table><thead><tr><th>Context</th><th>Human</th>%(headers)s<th>State/options</th></tr></thead><tbody id='rows'></tbody></table></main>
<script id='data' type='application/json'>%(payload)s</script><script>
const data=JSON.parse(document.getElementById('data').textContent),body=document.getElementById('rows'),q=document.getElementById('q'),a=document.getElementById('a');let mistakes=false;document.getElementById('m').onclick=()=>{mistakes=!mistakes;render()};q.oninput=a.onchange=render;const e=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function cell(x){if(!x.covered)return '<td>not scored</td>';return `<td><b>${e(x.topLabel)}</b><div class=small>${((x.topProbability||0)*100).toFixed(1)}%% · human rank ${e(x.playedRank)}</div><details><summary>top choices</summary><pre>${e(x.top.map((r,i)=>`${i+1}. ${r.label}`).join('\n'))}</pre></details></td>`}function render(){body.innerHTML='';for(const r of data.rows){if(a.value&&r.agreement!==a.value)continue;if(q.value&&!JSON.stringify(r).toLowerCase().includes(q.value.toLowerCase()))continue;if(mistakes&&!r.models.some(x=>x.covered&&x.playedRank!==1))continue;const tr=document.createElement('tr');tr.className=r.agreement;tr.innerHTML=`<td>turn ${e(r.turn)} ${e(r.phase)}<div class=small>${e(r.promptType)} · ${e(r.agreement)}</div></td><td><b>${e(r.human.label)}</b></td>${r.models.map(cell).join('')}<td>${e(r.stateSummary)}<details><summary>legal options</summary><pre>${e(r.options.map((o,i)=>`${i}. ${o.label}`).join('\n'))}</pre></details></td>`;body.appendChild(tr)}}render();
</script></body></html>""" % {"title": title, "headers": headers, "payload": payload}


def _option_identity(options, index):
    if not isinstance(index, int) or not 0 <= index < len(options):
        return None
    canonical = _option_key(options, index)
    if canonical is not None:
        return "canonical:" + str(canonical)
    option = options[index]
    return "option:%s:%s" % (index, option.get("type") or "")


def _select(record):
    direct = record.get("select")
    return direct if isinstance(direct, dict) else \
        (record.get("observation") or {}).get("select") or {}


def _selected_indices(record):
    selected = record.get("selectedIndices") or record.get("selected")
    if selected is None and isinstance(record.get("select"), list):
        selected = record.get("select")
    return selected or []


def _option_label(options, index):
    if not isinstance(index, int) or not 0 <= index < len(options):
        return None
    return options[index].get("label") or options[index].get("type") or str(index)


def _option_key(options, index):
    if not isinstance(index, int) or not 0 <= index < len(options):
        return None
    payload = options[index].get("payload") or {}
    return payload.get("canonicalKey") if isinstance(payload, dict) else None


def _public_option(option):
    payload = option.get("payload") or {}
    return {"index": option.get("index"), "type": option.get("type"),
            "label": option.get("label"),
            "canonicalKey": payload.get("canonicalKey")
                if isinstance(payload, dict) else None}


def _state_summary(state):
    life = ["P%s %s life" % (player.get("seat", "?"), player["life"])
            for player in state.get("players") or []
            if isinstance(player, dict) and player.get("life") is not None]
    return " · ".join(life) or "No compact state summary"


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, content):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _parse_model(value):
    if "=" not in value:
        raise ValueError("--model must use NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--model must use NAME=CHECKPOINT")
    return name.strip(), path.strip()


def build_parser():
    parser = argparse.ArgumentParser(prog="magic-cabt-compare-models")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", action="append", required=True,
                        metavar="NAME=CHECKPOINT")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--title", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = compare_bundle(
        args.bundle, [_parse_model(value) for value in args.model], args.out,
        out_json=args.json, device=args.device, top_k=args.top_k,
        title=args.title, progress=lambda name, done, total: print(
            "[%s] %d/%d" % (name, done, total), file=os.sys.stderr))
    print(json.dumps({"html": result["html"], "json": result["json"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
