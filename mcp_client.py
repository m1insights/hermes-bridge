"""
Hermes MCP Client — JSON-RPC stdio transport for Hermes Agent's MCP server.

Spawns `hermes mcp serve` as a subprocess and communicates via JSON-RPC 2.0
over stdin/stdout. Provides async methods for the AgentZero control plane
(approvals, events, conversations, messages).
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"

# Candidate locations for the hermes binary, checked in order.
_HERMES_CANDIDATES = [
    Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes",
    Path.home() / ".local" / "bin" / "hermes",
]


def _find_hermes_binary() -> str:
    """Locate the hermes binary. Returns the path string or raises FileNotFoundError."""
    for candidate in _HERMES_CANDIDATES:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    # Fall back to PATH lookup
    found = shutil.which("hermes")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find 'hermes' binary. Checked:\n"
        + "\n".join(f"  - {c}" for c in _HERMES_CANDIDATES)
        + "\n  - PATH lookup"
    )


class MCPError(Exception):
    """Raised when the MCP server returns a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.error_message = message
        self.data = data
        super().__init__(f"MCP error {code}: {message}")


class HermesMCPClient:
    """Async MCP client that manages a `hermes mcp serve` subprocess.

    Usage::

        client = HermesMCPClient()
        await client.start()
        approvals = await client.list_approvals()
        await client.stop()

    Or as an async context manager::

        async with HermesMCPClient() as client:
            approvals = await client.list_approvals()
    """

    def __init__(self, hermes_path: Optional[str] = None):
        self._hermes_path = hermes_path
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._stopping = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the MCP subprocess, run the initialize handshake."""
        if self._process is not None and self._process.returncode is None:
            return  # already running

        binary = self._hermes_path or _find_hermes_binary()
        logger.info("Starting MCP subprocess: %s mcp serve", binary)

        self._stopping = False
        self._process = await asyncio.create_subprocess_exec(
            binary, "mcp", "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

        # MCP initialize handshake
        result = await self._send_request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "hermes-bridge",
                "version": "1.0.0",
            },
        })
        logger.info("MCP initialized: %s", result)

        # Send initialized notification (no id, no response expected)
        await self._send_notification("notifications/initialized")
        self._initialized = True

    async def stop(self) -> None:
        """Gracefully shut down the MCP subprocess."""
        self._stopping = True
        self._initialized = False

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._process and self._process.returncode is None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception:
                pass
        self._process = None

        # Fail any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError(-1, "Client stopped"))
        self._pending.clear()

    async def restart(self) -> None:
        """Stop and re-start the MCP subprocess."""
        await self.stop()
        await self.start()

    async def ensure_connected(self) -> None:
        """If the subprocess died, restart it automatically."""
        if self._process is None or self._process.returncode is not None:
            logger.warning("MCP subprocess not running — restarting")
            await self.start()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: Any = None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        await self._ensure_process()
        rid = self._next_id()
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[rid] = fut

        await self._write(msg)

        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise MCPError(-1, f"Timeout waiting for response to {method}")

    async def _send_notification(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        await self._ensure_process()
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        await self._write(msg)

    async def _write(self, msg: dict) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        data = json.dumps(msg)
        line = data + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()
        logger.debug("MCP TX: %s", data)

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from subprocess stdout."""
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    # EOF — process exited
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("MCP: non-JSON line: %s", line[:200])
                    continue

                logger.debug("MCP RX: %s", line[:500])

                rid = msg.get("id")
                if rid is not None and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if "error" in msg:
                        err = msg["error"]
                        fut.set_exception(MCPError(
                            err.get("code", -1),
                            err.get("message", "Unknown error"),
                            err.get("data"),
                        ))
                    else:
                        fut.set_result(msg.get("result"))
                # Ignore notifications from the server (no pending future)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("MCP read loop error")
        finally:
            if not self._stopping:
                logger.warning("MCP read loop exited — subprocess may have died")
                # Fail all pending futures
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(MCPError(-1, "MCP subprocess exited"))
                self._pending.clear()

    async def _ensure_process(self) -> None:
        """Raise if process is not available."""
        if self._process is None or self._process.returncode is not None:
            raise MCPError(-1, "MCP subprocess not running — call start() or ensure_connected()")

    # ------------------------------------------------------------------
    # MCP tool calls — high-level API
    # ------------------------------------------------------------------

    async def _call_tool(self, tool_name: str, arguments: Optional[dict] = None) -> Any:
        """Call an MCP tool and return the result content."""
        await self.ensure_connected()
        params: dict[str, Any] = {"name": tool_name}
        if arguments:
            params["arguments"] = arguments
        result = await self._send_request("tools/call", params)
        # MCP tools/call returns {"content": [...]} — extract for convenience
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            # If single text item, unwrap it
            if (isinstance(content, list) and len(content) == 1
                    and isinstance(content[0], dict) and content[0].get("type") == "text"):
                try:
                    return json.loads(content[0]["text"])
                except (json.JSONDecodeError, KeyError):
                    return content[0].get("text", content)
            return content
        return result

    # -- Approvals / Permissions -----------------------------------------

    async def list_approvals(self) -> Any:
        """List pending permission approvals."""
        return await self._call_tool("permissions_list_open")

    async def respond_approval(self, approval_id: str, approved: bool, reason: str = "") -> Any:
        """Approve or deny a pending permission request."""
        args = {
            "approval_id": approval_id,
            "approved": approved,
        }
        if reason:
            args["reason"] = reason
        return await self._call_tool("permissions_respond", args)

    # -- Events ----------------------------------------------------------

    async def poll_events(self, since: Optional[str] = None) -> Any:
        """Poll for new events from Hermes."""
        args = {}
        if since:
            args["since"] = since
        return await self._call_tool("events_poll", args if args else None)

    # -- Conversations ---------------------------------------------------

    async def list_conversations(self) -> Any:
        """List all Hermes conversations."""
        return await self._call_tool("conversations_list")

    async def get_conversation(self, conversation_id: str) -> Any:
        """Get details of a specific conversation."""
        return await self._call_tool("conversation_get", {"conversation_id": conversation_id})

    # -- Messages --------------------------------------------------------

    async def read_messages(self, conversation_id: str, limit: int = 50) -> Any:
        """Read messages from a conversation."""
        return await self._call_tool("messages_read", {
            "conversation_id": conversation_id,
            "limit": limit,
        })

    async def send_message(self, conversation_id: str, content: str) -> Any:
        """Send a message to a conversation."""
        return await self._call_tool("messages_send", {
            "conversation_id": conversation_id,
            "content": content,
        })

    # -- Channels --------------------------------------------------------

    async def list_channels(self) -> Any:
        """List available channels."""
        return await self._call_tool("channels_list")

    # -- Attachments -----------------------------------------------------

    async def fetch_attachment(self, attachment_id: str) -> Any:
        """Fetch an attachment by ID."""
        return await self._call_tool("attachments_fetch", {"attachment_id": attachment_id})
