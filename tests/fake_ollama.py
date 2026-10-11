"""Tiny fake Ollama server for tests: answers by recognising which prompt stage is calling."""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class State:
    director_calls = 0
    revise_calls = 0
    prompts = []


def _reply(prompt, images):
    State.prompts.append(prompt)
    if prompt.startswith("Extract a FACT SHEET"):
        return {"product_name": "Recruitment OS", "one_liner": "Local-first AI recruitment operating system.",
                "audience": "talent teams", "problem": "scattered hiring tools",
                "claims": ["Screens job descriptions into a ranked shortlist", "Permanent candidate CRM", "Visual Kanban pipeline"],
                "differentiators": ["local-first"], "metrics": [{"value": "12", "label": "minutes to shortlist"}]}
    if images:                                     # vision stage
        return {"screen": "Dashboard", "purpose": "See open roles and pipeline health", "stage": "core",
                "visible_text": ["Dashboard", "Open roles"], "numbers": [{"value": "12", "label": "open roles"}],
                "regions": [{"label": "KPI row", "x": 0.05, "y": 0.1, "w": 0.6, "h": 0.2},
                            {"label": "Pipeline", "x": 0.1, "y": 0.4, "w": 0.8, "h": 0.4}]}
    if prompt.startswith("ROLE: You are the creative director"):
        State.director_calls += 1
        return {"title": "Hiring, run like a system", "cta": "See it in your stack", "hashtags": ["hiring", "ai"],
                "description": "A calm tour.", "art": "dark indigo gradient",
                "beats": [
                    {"screen_id": 0, "region": 0, "scene": "ui", "kicker": "STEP 1", "caption": "A revolutionary way to screen candidates!", "sub": "Rank applicants in 12 minutes", "vo": "Rank applicants quickly."},
                    {"screen_id": 1, "region": 1, "scene": "ui", "kicker": "STEP 2", "caption": "Pipeline state, always current", "sub": "One board for every role", "vo": "One board for every role."},
                    {"screen_id": 1, "region": 1, "scene": "ui", "kicker": "dup", "caption": "duplicate region should be dropped", "sub": "", "vo": ""},
                    {"screen_id": 0, "region": -1, "scene": "kpi", "kicker": "RESULT", "caption": "Shortlist speed", "kpi_value": "12 min", "kpi_label": "to shortlist", "vo": "Twelve minutes."}]}
    if prompt.startswith("You previously wrote this storyboard JSON"):
        State.revise_calls += 1
        raw = json.loads(prompt.split("STORYBOARD\n", 1)[1].split("\n\nPROBLEMS", 1)[0])
        raw["beats"][0]["caption"] = "Screen candidates to a ranked shortlist"
        return raw
    if prompt.startswith("Write platform-specific"):
        return {"linkedin": "A calm look at Recruitment OS.", "x": "Hiring as a system.", "youtube_shorts": "Hiring, run like a system",
                "instagram": "Pipeline, always current.", "hashtags": ["hiring"]}
    if prompt.startswith("Rewrite ONE beat"):
        return {"kicker": "STEP 1", "caption": "Ranked shortlist from any job description", "sub": "No manual screening", "vo": "Get a ranked shortlist."}
    if prompt.startswith("ROLE: You are the head writer"):
        return {"title": "Calm Mind", "lines": [f"Line number {i} stays calm" for i in range(1, 9)], "vo": [], "hashtags": ["stoic"], "description": "d", "art": "a"}
    return {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "gemma3:12b"}, {"name": "nomic-embed-text:latest"}]})
        else:
            self._send({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/show":
            self._send({"capabilities": ["completion", "vision"], "details": {"families": ["gemma3"]}})
        elif self.path == "/api/generate":
            if not body.get("prompt"):
                self._send({"response": ""}); return
            self._send({"response": json.dumps(_reply(body["prompt"], body.get("images")))})
        elif self.path == "/api/embed":
            t = body["input"][0]
            self._send({"embeddings": [[float(len(t) % 7), 1.0, float(sum(map(ord, t)) % 5)]]})
        else:
            self._send({})


def start():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"
