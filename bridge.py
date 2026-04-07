#!/usr/bin/env python3
"""
Hermes Agent Bridge — OpenAI-compatible HTTP wrapper for Hermes Agent.

Exposes Hermes Agent (NousResearch) as an OpenAI-compatible API so that
any client expecting /v1/chat/completions (including AgentZero's BYOK system)
can talk to a local Hermes Agent instance over HTTP.
"""

import argparse
import asyncio
import collections
import hmac
import json
import threading
import secrets
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# ---------------------------------------------------------------------------
# Hermes Agent import
# ---------------------------------------------------------------------------

try:
    from run_agent import AIAgent
except ImportError:
    print(
        "Error: Could not import AIAgent from run_agent.\n"
        "Make sure Hermes Agent is installed and 'run_agent' is importable.\n"
        "See https://github.com/NousResearch/hermes-agent for installation."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".hermes"
CONFIG_PATH = CONFIG_DIR / "bridge.json"
MODEL_ID = "hermes-agent"


def load_config(port_override: Optional[int] = None) -> dict:
    """Load or create bridge config with auto-generated bearer token."""
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}

    changed = False

    if "token" not in config:
        config["token"] = secrets.token_urlsafe(32)
        changed = True

    if port_override is not None:
        config["port"] = port_override
        changed = True
    elif "port" not in config:
        config["port"] = 8642
        changed = True

    if changed:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(config, indent=2) + "\n")
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600 — owner read/write only
        tmp.replace(CONFIG_PATH)

    return config


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    """Extract client IP from the request."""
    return request.client.host if request.client else "unknown"


def verify_token(authorization: Optional[str], expected_token: str, client_ip: str = "unknown") -> None:
    """Validate Bearer token or raise 401. Tracks failed auth attempts per IP."""
    failed = False
    if not authorization:
        failed = True
    else:
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not hmac.compare_digest(parts[1], expected_token):
            failed = True

    if failed:
        if not _auth_fail_limiter.allow(client_ip):
            raise HTTPException(status_code=429, detail="Too many failed attempts")
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# ---------------------------------------------------------------------------
# Agent singleton
# ---------------------------------------------------------------------------


_agent: Optional[AIAgent] = None
_agent_lock = threading.Lock()

HERMES_CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _read_hermes_provider() -> dict:
    """Read provider and model from Hermes config.yaml."""
    result = {}
    if not HERMES_CONFIG_PATH.exists():
        return result
    try:
        # Minimal YAML parsing — avoid pulling in pyyaml dependency
        text = HERMES_CONFIG_PATH.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("provider:") and "provider:" in line and not line.startswith(" "):
                # Top-level provider under model: section — but we need the one
                # directly under "model:". Use simple state tracking.
                pass
        # More reliable: look for the model.provider pattern
        in_model = False
        for line in text.splitlines():
            if line.startswith("model:"):
                in_model = True
                continue
            if in_model and not line.startswith(" "):
                in_model = False
            if in_model and "provider:" in line:
                val = line.split("provider:", 1)[1].strip()
                if val:
                    result["provider"] = val
                break
    except Exception:
        pass
    return result


def get_agent() -> AIAgent:
    """Return the shared AIAgent instance, creating it on first call."""
    global _agent
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is None:
            hermes_cfg = _read_hermes_provider()
            provider = hermes_cfg.get("provider")
            kwargs = {"quiet_mode": True}
            if provider:
                kwargs["provider"] = provider
            _agent = AIAgent(**kwargs)
    return _agent


# ---------------------------------------------------------------------------
# OpenAI response helpers
# ---------------------------------------------------------------------------


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _unix_ts() -> int:
    return int(time.time())


def _make_chunk(
    completion_id: str,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    """Build a single SSE-formatted chat.completion.chunk."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": _unix_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _make_completion(
    completion_id: str,
    model: str,
    content: str,
) -> dict:
    """Build a non-streaming chat.completion response."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": _unix_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Streaming bridge
# ---------------------------------------------------------------------------


async def _stream_response(agent: AIAgent, message: str, comp_id: str, model: str):
    """Yield OpenAI-format SSE chunks from the synchronous agent callback."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_delta(text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, text)

    def run_agent() -> None:
        try:
            agent.chat(message, stream_callback=on_delta)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Initial chunk with role
    yield _make_chunk(comp_id, model, {"role": "assistant"})

    asyncio.get_event_loop().run_in_executor(None, run_agent)

    while True:
        delta = await queue.get()
        if delta is None:
            # Final chunk with finish_reason
            yield _make_chunk(comp_id, model, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            break
        elif isinstance(delta, Exception):
            # Surface agent errors as a content chunk, then stop
            yield _make_chunk(comp_id, model, {"content": "\n[Error: agent encountered an issue]"})
            yield _make_chunk(comp_id, model, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            break
        else:
            yield _make_chunk(comp_id, model, {"content": delta})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class _LimitBodyMiddleware:
    """Reject requests with bodies larger than 1 MB to prevent memory exhaustion.

    Counts actual bytes received (handles chunked transfer encoding),
    not just the Content-Length header.
    """

    MAX_BODY = 1_048_576  # 1 MB

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            bytes_received = 0

            async def limited_receive():
                nonlocal bytes_received
                message = await receive()
                if message.get("type") == "http.request":
                    bytes_received += len(message.get("body", b""))
                    if bytes_received > self.MAX_BODY:
                        from starlette.responses import JSONResponse

                        raise _BodyTooLarge()
                return message

            try:
                await self.app(scope, limited_receive, send)
            except _BodyTooLarge:
                from starlette.responses import JSONResponse

                resp = JSONResponse({"detail": "Request too large"}, status_code=413)
                await resp(scope, receive, send)
                return
        else:
            await self.app(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter.

    Tracks requests per IP. Returns True if the request should be allowed.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            dq = self._requests.setdefault(ip, collections.deque())
            # Evict expired entries
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True


# Shared rate limiters
_rate_limiter = _RateLimiter(max_requests=60, window_seconds=60)
_auth_fail_limiter = _RateLimiter(max_requests=5, window_seconds=60)


def create_app(token: str) -> FastAPI:
    app = FastAPI(title="Hermes Agent Bridge", version="1.0.0")
    app.add_middleware(_LimitBodyMiddleware)

    # -- Health ----------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # -- Models ----------------------------------------------------------

    @app.get("/v1/models")
    async def list_models(request: Request, authorization: Optional[str] = Header(None)):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": _unix_ts(),
                    "owned_by": "hermes-bridge",
                }
            ],
        }

    # -- Chat completions ------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise HTTPException(
                status_code=400,
                detail="'messages' field is required and must be a non-empty list",
            )

        # Extract the last user message as the prompt for the agent.
        # Hermes Agent manages its own conversation history internally,
        # so we pass only the latest user turn.
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Handle content that may be a list of parts (vision format)
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    user_message = " ".join(text_parts)
                else:
                    user_message = str(content)
                break

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found in 'messages' array",
            )

        stream = body.get("stream", False)
        comp_id = _completion_id()
        model = body.get("model", MODEL_ID)
        agent = get_agent()

        if stream:
            return StreamingResponse(
                _stream_response(agent, user_message, comp_id, model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming: run agent synchronously in thread pool
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, agent.chat, user_message)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception("Agent error during chat completion")
                raise HTTPException(
                    status_code=500,
                    detail="Agent error — check server logs",
                )
            return JSONResponse(_make_completion(comp_id, model, result))

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Hermes Agent Bridge — OpenAI-compatible API wrapper"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: 8642, or value from ~/.hermes/bridge.json)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    config = load_config(port_override=args.port)
    token = config["token"]
    port = config["port"]

    print("=" * 56)
    print("  Hermes Agent Bridge")
    print("=" * 56)
    print()
    print(f"  Local:  http://{args.host}:{port}/v1")
    print(f"  Token:  {token[:8]}...{token[-4:]}")
    print()
    print("  For mobile access, run:")
    print(f"    tailscale serve {port}")
    print("  Then use your https://<machine>.ts.net URL in AgentZero.")
    print()
    print("=" * 56)

    app = create_app(token)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
