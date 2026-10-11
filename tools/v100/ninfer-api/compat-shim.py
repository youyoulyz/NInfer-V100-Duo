#!/usr/bin/env python3
"""Forward-compatibility shim for agent clients that outpace the server's strict parsers.

The engine validates the Responses and Anthropic request bodies field by field, which catches client
mistakes but also refuses the fields a newer agent build starts sending. That is exactly what
happened with Codex 0.162 (`client_metadata`, `prompt_cache_key`, `include`, a namespaced tool and a
disabled `web_search`) and Claude Code 2.1 (`output_config.effort = "high"`); the engine now tolerates
that set natively, and this shim is the answer for the *next* such field.

    ./compat-shim.py [--port 8098] [--upstream 127.0.0.1:8080]

Point the client at the shim's port instead of the server's. It forwards every request and response
body unchanged -- streaming included -- after:

  * dropping the Responses fields the engine does not act on (client telemetry, cache hints, extra
    response fields, reasoning summary) and the tools it cannot execute (anything that is not a
    `function`);
  * translating the two effort vocabularies onto the template's ladder (`minimal` -> `low`,
    `high`/`max` -> `xhigh`), which the engine now also does itself.

Every change is appended to `--log` (default: stderr) so it is visible what a client version needed.
"""
import argparse
import http.client
import http.server
import json
import sys
import urllib.parse

DROPPED_RESPONSES_FIELDS = ("client_metadata", "prompt_cache_key", "prompt_cache_options",
                            "prompt_cache_retention", "include")
EFFORT_TRANSLATION = {"minimal": "low", "high": "xhigh", "max": "xhigh"}


def rewrite(path, body, log):
    """Returns (body, notes) with the compatibility rewrites for this request applied."""
    if not isinstance(body, dict):
        return body, []
    notes = []
    is_responses = path.startswith("/v1/responses")
    if is_responses:
        for field in DROPPED_RESPONSES_FIELDS:
            if body.pop(field, None) is not None:
                notes.append("dropped " + field)
        if isinstance(body.get("reasoning"), dict) and "summary" in body["reasoning"]:
            body["reasoning"].pop("summary")
            notes.append("dropped reasoning.summary")
        if isinstance(body.get("reasoning"), dict):
            effort = body["reasoning"].get("effort")
            if effort in EFFORT_TRANSLATION:
                body["reasoning"]["effort"] = EFFORT_TRANSLATION[effort]
                notes.append("reasoning.effort %s->%s" % (effort, body["reasoning"]["effort"]))
        if isinstance(body.get("tools"), list):
            keep = [tool for tool in body["tools"]
                    if not isinstance(tool, dict) or tool.get("type") == "function"]
            if len(keep) != len(body["tools"]):
                notes.append("skipped %d non-function tool(s)" % (len(body["tools"]) - len(keep)))
                body["tools"] = keep
    elif path.startswith("/v1/messages") and isinstance(body.get("output_config"), dict):
        effort = body["output_config"].get("effort")
        if effort in EFFORT_TRANSLATION:
            body["output_config"]["effort"] = EFFORT_TRANSLATION[effort]
            notes.append("output_config.effort %s->%s" % (effort, body["output_config"]["effort"]))
    return body, notes


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = ("127.0.0.1", 8080)
    log = sys.stderr

    def forward(self, body=None):
        connection = http.client.HTTPConnection(*self.upstream, timeout=900)
        headers = {name: value for name, value in self.headers.items()
                   if name.lower() not in ("host", "content-length")}
        if body is not None:
            headers["content-length"] = str(len(body))
        connection.request(self.command, self.path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        self.send_response(response.status)
        for name, value in response.getheaders():
            if name.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(name, value)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        try:
            body = json.loads(raw)
        except ValueError:
            return self.forward(raw)
        body, notes = rewrite(urllib.parse.urlparse(self.path).path, body, self.log)
        if notes:
            print("[compat-shim] %s: %s" % (self.path, ", ".join(notes)), file=self.log, flush=True)
            raw = json.dumps(body).encode()
        self.forward(raw)

    def do_GET(self):
        self.forward()

    def do_DELETE(self):
        self.forward()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--upstream", default="127.0.0.1:8080")
    parser.add_argument("--log", default="/dev/stderr")
    args = parser.parse_args()
    host, _, port = args.upstream.partition(":")
    Handler.upstream = (host, int(port or 8080))
    Handler.log = sys.stderr if args.log == "/dev/stderr" else open(args.log, "a", buffering=1)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("[compat-shim] 127.0.0.1:%d -> %s:%d" % (args.port, Handler.upstream[0],
                                                  Handler.upstream[1]), file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
