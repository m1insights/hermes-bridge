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
import logging
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

# Operator dependencies — optional, bridge works without them for basic chat
try:
    import httpx
    from mcp_client import HermesMCPClient, MCPError
    _OPERATOR_AVAILABLE = True
except ImportError:
    _OPERATOR_AVAILABLE = False
if _OPERATOR_AVAILABLE:
    from file_reader import read_memory, read_skills, read_skill, read_status

logger = logging.getLogger(__name__)

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

    # ==================================================================
    # Hermes Operator — Control Plane Endpoints
    # Requires: pip install httpx + mcp_client.py + file_reader.py
    # If unavailable, bridge works normally for basic chat.
    # ==================================================================

    if not _OPERATOR_AVAILABLE:
        logger.info("Operator dependencies not installed — control plane disabled. Install httpx for full operator support.")
        return app

    mcp_client = HermesMCPClient()

    # Event distribution state
    _event_buffer: collections.deque = collections.deque(maxlen=500)
    _sse_subscribers: list[asyncio.Queue] = []
    _event_poll_task: Optional[asyncio.Task] = None
    _push_http_client: Optional[httpx.AsyncClient] = None

    PUSH_RELAY_URL = "https://agentzero-backend.onrender.com/v1/push/relay"

    def _require_mcp() -> None:
        """Raise 503 if MCP client is not connected."""
        if not mcp_client._initialized:
            raise HTTPException(status_code=503, detail="MCP client not connected")

    async def _send_push_nudge(event_type: str, approval_id: Optional[str] = None) -> None:
        """Send a zero-content push nudge via the backend relay if a device token is registered."""
        nonlocal _push_http_client
        cfg = {}
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        device_token = cfg.get("device_token")
        relay_secret = os.environ.get("PUSH_RELAY_SECRET", "")
        if not device_token or not relay_secret:
            return
        if _push_http_client is None:
            _push_http_client = httpx.AsyncClient(timeout=10.0)
        body: dict = {"device_token": device_token, "event_type": event_type}
        if approval_id:
            body["approval_id"] = approval_id
        try:
            await _push_http_client.post(
                PUSH_RELAY_URL,
                json=body,
                headers={"X-Relay-Secret": relay_secret},
            )
        except Exception:
            logger.warning("Failed to send push nudge for event_type=%s", event_type)

    async def _event_poll_loop() -> None:
        """Background loop: poll MCP events every 300ms and distribute."""
        last_event_id: Optional[str] = None
        while True:
            try:
                await asyncio.sleep(0.3)
                if not mcp_client._initialized:
                    continue
                try:
                    events = await mcp_client.poll_events(since=last_event_id)
                except MCPError:
                    continue
                if not events:
                    continue
                if isinstance(events, dict):
                    events = events.get("events", [])
                if not isinstance(events, list):
                    continue
                for event in events:
                    _event_buffer.append(event)
                    # Distribute to SSE subscribers
                    dead: list[asyncio.Queue] = []
                    for q in _sse_subscribers:
                        try:
                            q.put_nowait(event)
                        except asyncio.QueueFull:
                            dead.append(q)
                    for q in dead:
                        try:
                            _sse_subscribers.remove(q)
                        except ValueError:
                            pass
                    # Push nudge for certain event types
                    etype = event.get("type", "") if isinstance(event, dict) else ""
                    if etype in ("approval_requested", "run_completed", "run_failed"):
                        asyncio.create_task(_send_push_nudge(
                            etype,
                            event.get("approval_id") if isinstance(event, dict) else None,
                        ))
                    # Track last event ID for pagination
                    if isinstance(event, dict) and "id" in event:
                        last_event_id = event["id"]
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Event poll loop error")
                await asyncio.sleep(1.0)

    @app.on_event("startup")
    async def _startup_mcp():
        nonlocal _event_poll_task
        try:
            await mcp_client.start()
            logger.info("MCP client started")
        except Exception:
            logger.warning("MCP client failed to start — operator endpoints will return 503")
        _event_poll_task = asyncio.create_task(_event_poll_loop())

    @app.on_event("shutdown")
    async def _shutdown_mcp():
        nonlocal _event_poll_task, _push_http_client
        if _event_poll_task and not _event_poll_task.done():
            _event_poll_task.cancel()
            try:
                await _event_poll_task
            except asyncio.CancelledError:
                pass
            _event_poll_task = None
        await mcp_client.stop()
        if _push_http_client:
            await _push_http_client.aclose()
            _push_http_client = None

    # -- SSE Event Stream -----------------------------------------------

    @app.get("/hermes/events/stream")
    async def hermes_event_stream(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        _require_mcp()

        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        _sse_subscribers.append(queue)

        async def event_generator():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                return
            finally:
                try:
                    _sse_subscribers.remove(queue)
                except ValueError:
                    pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -- Approvals -------------------------------------------------------

    @app.get("/hermes/approvals")
    async def hermes_list_approvals(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        _require_mcp()
        result = await mcp_client.list_approvals()
        return JSONResponse({"approvals": result})

    @app.post("/hermes/approvals/{approval_id}/respond")
    async def hermes_respond_approval(
        approval_id: str,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        _require_mcp()
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        action = body.get("action")
        if action not in ("approve", "deny"):
            raise HTTPException(status_code=400, detail="'action' must be 'approve' or 'deny'")
        approved = action == "approve"
        result = await mcp_client.respond_approval(approval_id, approved)
        return JSONResponse({"result": result})

    # -- Memory ----------------------------------------------------------

    @app.get("/hermes/memory")
    async def hermes_memory(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        return JSONResponse(read_memory())

    # -- Skills ----------------------------------------------------------

    @app.get("/hermes/skills")
    async def hermes_skills(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        return JSONResponse({"skills": read_skills()})

    @app.get("/hermes/skills/{skill_name}")
    async def hermes_skill(
        skill_name: str,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        skill = read_skill(skill_name)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
        return JSONResponse(skill)

    # -- Conversations ---------------------------------------------------

    @app.get("/hermes/conversations")
    async def hermes_conversations(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        _require_mcp()
        result = await mcp_client.list_conversations()
        return JSONResponse({"conversations": result})

    @app.get("/hermes/conversations/{conversation_id}")
    async def hermes_conversation(
        conversation_id: str,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        _require_mcp()
        result = await mcp_client.get_conversation(conversation_id)
        return JSONResponse(result)

    # -- Status ----------------------------------------------------------

    @app.get("/hermes/status")
    async def hermes_status(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        ip = _client_ip(request)
        if not _rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        verify_token(authorization, token, client_ip=ip)
        status = read_status()
        status["mcp_connected"] = mcp_client._initialized
        return JSONResponse(status)

    # -- Push Registration -----------------------------------------------

    @app.post("/hermes/push/register")
    async def hermes_push_register(
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
        device_token = body.get("device_token")
        if not device_token or not isinstance(device_token, str):
            raise HTTPException(status_code=400, detail="'device_token' is required (string)")
        # Persist to bridge.json
        config = {}
        if CONFIG_PATH.exists():
            try:
                config = json.loads(CONFIG_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                config = {}
        config["device_token"] = device_token
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(config, indent=2) + "\n")
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)
        tmp.replace(CONFIG_PATH)
        return JSONResponse({"status": "registered"})

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
