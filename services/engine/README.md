# Deterministic XMage engine service

This directory exposes the pinned XMage rules/card engine as a deterministic agent environment.

It is deliberately **not** a second Magic rules implementation. The reference engine is XMage commit `fd40ad5c29a92cef824cf12ba6d0e4daa25db975`, plus the CABT bridge in this repository. A future Rust/Wasm engine should be proven against this interface before replacing it.

## Contract

A game is identified by:

- the pinned XMage revision;
- the ordered deck specifications;
- the integer seed;
- the max-turn setting;
- the ordered sequence of stable semantic action IDs.

For that input, verification requires fresh engine processes to reproduce:

1. the exact canonical agent observation;
2. the exact legal action IDs and labels;
3. the public-observation fingerprint;
4. a separate private engine-state fingerprint;
5. the canonical terminal result.

Runtime XMage UUIDs are not part of the agent contract. Known player/card identities are projected to stable refs such as `P0` and `P0:D00007`; unknown runtime UUIDs are redacted. If two legal choices cannot be distinguished semantically, action-ID construction fails closed with `AMBIGUOUS_SEMANTIC_ACTION` instead of assigning replay identity from enumeration order.

The private engine fingerprint includes hidden library/hand state and is **verification metadata, not model input**.

## Host API

`native.py` reads one JSON object per line from stdin and emits one JSON response per line.

Start:

```json
{"command":"new","spec":{"decks":["24 Forest\n36 Grizzly Bears","24 Forest\n36 Grizzly Bears"],"seed":7,"maxTurns":20}}
```

A nonterminal response contains:

```json
{
  "ok": true,
  "data": {
    "finished": false,
    "revision": "fd40ad5c...",
    "setup": "declared-library-order-v1",
    "position": 0,
    "fingerprint": "...",
    "observation": {
      "current": {},
      "select": {
        "playerIndex": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [
          {"actionId": "a_...", "type": "PASS_PRIORITY", "label": "Pass priority"}
        ]
      }
    }
  }
}
```

Apply an action using the **semantic ID**, plus the public fingerprint of the state it was chosen from:

```json
{"command":"step","fingerprint":"...","actionIds":["a_..."]}
```

The stale-state check prevents an action chosen for one position from being applied to another.

Other host commands:

- `observe`: canonical agent state only.
- `checkpoint`: replay prefix plus the private digest needed to certify a reconstructed root.
- `restore`: reconstruct from the seed/decks/action-ID prefix and reject if public or private state differs.
- `verify`: host-only public/private fingerprints. Do not pass this response to an agent.
- `autoplay`: smoke/benchmark utility using the bundled simple controls.

Numeric option indices remain accepted for compatibility but are converted immediately to action IDs. Persist action IDs, not indices.

## Build the pinned reference locally

Requirements: Java 17, Maven, Git, rsync, and Python 3.

The CI workflow is the canonical build recipe:

```sh
git init /tmp/xmage
git -C /tmp/xmage remote add origin https://github.com/magefree/mage.git
git -C /tmp/xmage fetch --depth 1 origin fd40ad5c29a92cef824cf12ba6d0e4daa25db975
git -C /tmp/xmage checkout --detach FETCH_HEAD
rsync -a Mage.Server.Plugins/Mage.Player.AI/ /tmp/xmage/Mage.Server.Plugins/Mage.Player.AI/
python3 services/engine/prepare_reference.py /tmp/xmage
mvn -f /tmp/xmage/pom.xml -B -pl Mage.Server.Plugins/Mage.Player.AI -am install -DskipTests
mvn -f /tmp/xmage/pom.xml -q -pl Mage.Server.Plugins/Mage.Player.AI dependency:build-classpath -Dmdep.outputFile=target/runtime-classpath.txt
export MAGIC_CABT_CLASSPATH="/tmp/xmage/Mage.Server.Plugins/Mage.Player.AI/target/classes:$(cat /tmp/xmage/Mage.Server.Plugins/Mage.Player.AI/target/runtime-classpath.txt)"
export PYTHONPATH="$PWD/python:$PWD/services/engine"
python3 services/engine/native.py
```

For verification, run `python3 services/engine/test_native.py` with the same environment.

## Current verification suite

The native test intentionally starts **fresh Java processes** and checks:

- three identical seeded creature-game traces;
- a different seed produces a different hidden-state digest;
- repeated Lightning Bolt games through cast, mana and target prompts;
- repeated creature games through an actual attack declaration;
- checkpoint reconstruction using stable action IDs;
- the same alternative in-game priority branch from independently restored roots;
- byte-equivalent canonical agent observations;
- no process-local UUIDs in agent traces/results;
- no private engine fingerprint in agent observations;
- stale-state rejection;
- semantic-action ambiguity rejection.

The Java bridge suite also executes directly against the real pinned XMage engine.

Passing these tests establishes deterministic behavior for the exercised paths. It does **not** prove that every Magic card interaction is correct; rules correctness remains anchored to the pinned XMage implementation and its upstream tests. Unsupported/ambiguous bridge projections fail closed rather than claiming deterministic support.
