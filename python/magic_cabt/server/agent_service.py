"""Small HTTP service for hosted CABT policy decisions.

This is not an XMage server by itself. It is the policy sidecar that an XMage /
CABT session manager can call when a seat is controlled by an agent. Requests
contain one CABT observation, responses contain selected legal option indices,
and every request/response/error can be appended to JSONL for replay/debugging.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from magic_cabt.agents import available_agents, make_agent

__all__ = ["AgentService", "AgentServiceHandler", "main"]


class AgentService(object):
    """Routes observations to registered agents and logs decisions/errors."""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def list_agents(self):
        return {"agents": available_agents()}

    def decide(self, request):
        agent_name = request.get("agent") or "random"
        observation = request.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("request needs observation object")
        agent = make_agent(agent_name, seed=request.get("seed"))
        started = time.time()
        try:
            selection = agent.select(observation)
            scores = agent.score(observation)
            response = {
                "ok": True,
                "agent": agent_name,
                "selectedIndices": selection,
                "scores": scores,
                "latencyMs": int((time.time() - started) * 1000),
            }
            self._append("decisions.jsonl", {
                "timestamp": int(time.time()),
                "request": _safe_request_summary(request),
                "response": response,
            })
            return response
        except Exception as exc:  # noqa: BLE001 - service boundary logs errors
            self._append("errors.jsonl", {
                "timestamp": int(time.time()),
                "agent": agent_name,
                "error": type(exc).__name__,
                "message": str(exc),
                "request": _safe_request_summary(request),
            })
            raise

    def _append(self, name, record):
        if not self.log_dir:
            return
        path = os.path.join(self.log_dir, name)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class AgentServiceHandler(BaseHTTPRequestHandler):
    service = AgentService()

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json(200, {"ok": True})
            return
        if self.path == "/agents":
            self._json(200, self.service.list_agents())
            return
        self._json(404, {"ok": False, "error": "NOT_FOUND"})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/decide":
            self._json(404, {"ok": False, "error": "NOT_FOUND"})
            return
        try:
            request = self._read_json()
            response = self.service.decide(request)
            self._json(200, response)
        except ValueError as exc:
            self._json(400, {"ok": False, "error": "BAD_REQUEST", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - service boundary
            self._json(500, {"ok": False, "error": "AGENT_ERROR", "message": str(exc)})

    def log_message(self, fmt, *args):
        # Keep service usable in tests and as a sidecar without noisy stdout.
        return

    def _read_json(self):
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            raise ValueError("empty request body")
        if length > 2 * 1024 * 1024:
            raise ValueError("request body too large")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise ValueError("request body is not valid JSON")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json(self, status, payload):
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="magic-cabt-agent-service",
        description="Run a local HTTP policy service for CABT agent seats.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", default=None,
                        help="append decisions/errors JSONL under this directory")
    args = parser.parse_args(argv)
    AgentServiceHandler.service = AgentService(log_dir=args.log_dir)
    server = HTTPServer((args.host, args.port), AgentServiceHandler)
    print("magic-cabt-agent-service listening on %s:%d" % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _safe_request_summary(request):
    observation = request.get("observation") if isinstance(request, dict) else {}
    select = observation.get("select") if isinstance(observation, dict) else {}
    return {
        "agent": request.get("agent"),
        "seed": request.get("seed"),
        "sequenceNumber": request.get("sequenceNumber"),
        "promptType": select.get("type") if isinstance(select, dict) else None,
        "legalActionCount": len(select.get("option") or []) if isinstance(select, dict) else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
