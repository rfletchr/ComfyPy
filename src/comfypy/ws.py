import json
import struct
import logging
from dataclasses import dataclass

import websocket

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Event types — what the iterator yields.
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ExecutionStarted:
    """The prompt was dequeued and execution began."""

    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class ExecutionCached:
    """All nodes were cached from a previous run — execution was skipped."""

    prompt_id: str
    nodes: list[str]


@dataclass(frozen=True, kw_only=True)
class QueueUpdated:
    """The execution queue changed (broadcast to all sockets)."""

    queue_remaining: int


@dataclass(frozen=True, kw_only=True)
class NodeExecuting:
    """A node started executing."""

    node_id: str
    display_node: str
    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class NodeExecuted:
    """A node finished executing."""

    node_id: str
    display_node: str
    output: dict
    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class NodeProgress:
    """Sampler / progress update from a node."""

    node_id: str
    value: int
    max: int
    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class ExecutionFinished:
    """The prompt finished executing (all nodes done)."""

    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class ExecutionSuccess:
    """The server confirmed execution completed without errors."""

    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class ExecutionError:
    """A node errored during execution."""

    prompt_id: str
    error: dict


@dataclass(frozen=True, kw_only=True)
class ExecutionInterrupted:
    """Execution was interrupted (user cancelled)."""

    prompt_id: str


@dataclass(frozen=True, kw_only=True)
class PreviewImage:
    """A preview image from a sampler (binary frame)."""

    image: bytes
    image_type: str | None  # "jpeg", "png", or None when unknown
    metadata: dict | None


@dataclass(frozen=True, kw_only=True)
class ServerEvent:
    """A server-sent event we don't have a typed wrapper for."""

    event_type: str
    data: dict


# ------------------------------------------------------------------
# WebSocket connection + frame parsing.
# ------------------------------------------------------------------


def connect_websocket(base_url: str, client_id: str) -> websocket.WebSocket:
    """Open a WebSocket, negotiate feature flags, return the connected socket.

    The caller is responsible for calling ``close()`` on the returned socket.
    """
    ws_url = base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    ws = websocket.WebSocket()
    ws.connect(f"{ws_url}/ws?clientId={client_id}")

    # --- initial handshake ---

    # 1. Server sends status + sid
    raw = ws.recv()
    msg = json.loads(raw)
    if msg["type"] != "status":
        ws.close()
        raise RuntimeError(f"Expected 'status' on connect, got '{msg['type']}'")
    sid = msg["data"]["sid"]

    # 2. Client sends its feature flags (empty — we don't request anything)
    ws.send(json.dumps({"type": "feature_flags", "data": {}}))

    # 3. Server sends its feature flags (we don't use them, but must consume the frame)
    raw = ws.recv()
    msg = json.loads(raw)
    if msg["type"] != "feature_flags":
        logger.warning("Expected 'feature_flags' after connect, got '%s'", msg["type"])

    if sid != client_id:
        ws.close()
        raise RuntimeError(f"Server returned sid '{sid}', expected '{client_id}'")

    logger.debug("WebSocket connected (client_id=%s)", client_id)
    return ws


def parse_message(raw: str | bytes) -> object:
    """Parse a single raw WebSocket frame into a typed event.

    *raw* is a ``str`` for text frames and ``bytes`` for binary frames
    (matching the return type of ``websocket.WebSocket.recv()``).
    """
    if isinstance(raw, str):
        return _parse_text(json.loads(raw))
    return _parse_binary(raw)


# ------------------------------------------------------------------
# Internal parsers.
# ------------------------------------------------------------------

_BINARY_PREVIEW_IMAGE = 1
_BINARY_UNENCODED_PREVIEW_IMAGE = 2
_BINARY_PREVIEW_IMAGE_WITH_METADATA = 4


def _parse_text(msg: dict) -> object:
    event_type: str = msg["type"]
    data: dict = msg["data"]

    if event_type == "status":
        return QueueUpdated(
            queue_remaining=data["status"]["exec_info"]["queue_remaining"],
        )

    if event_type == "execution_start":
        return ExecutionStarted(prompt_id=data["prompt_id"])

    if event_type == "executing":
        node = data["node"]
        if node is None:
            return ExecutionFinished(prompt_id=data["prompt_id"])
        return NodeExecuting(
            node_id=node,
            display_node=data["display_node"],
            prompt_id=data["prompt_id"],
        )

    if event_type == "executed":
        return NodeExecuted(
            node_id=data["node"],
            display_node=data["display_node"],
            output=data.get("output") or {},
            prompt_id=data["prompt_id"],
        )

    if event_type == "progress":
        return NodeProgress(
            node_id=data["node"],
            value=data["value"],
            max=data["max"],
            prompt_id=data["prompt_id"],
        )

    if event_type == "progress_state":
        nodes = data.get("nodes")
        if not nodes:
            return None
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        return [
            NodeProgress(
                node_id=n["node_id"],
                value=n["value"],
                max=n["max"],
                prompt_id=data["prompt_id"],
            )
            for n in nodes
        ]

    if event_type == "execution_cached":
        return ExecutionCached(
            prompt_id=data["prompt_id"],
            nodes=data["nodes"],
        )

    if event_type == "execution_error":
        return ExecutionError(prompt_id=data["prompt_id"], error=data)

    if event_type == "execution_success":
        return ExecutionSuccess(prompt_id=data["prompt_id"])

    if event_type == "execution_interrupted":
        return ExecutionInterrupted(prompt_id=data["prompt_id"])

    # Catch-all for custom-node and future events.
    return ServerEvent(event_type=event_type, data=data)


def _parse_binary(raw: bytes) -> object:
    if len(raw) < 4:
        logger.warning("Binary frame too short (%d bytes)", len(raw))
        return None

    event_type = struct.unpack(">I", raw[:4])[0]
    payload = raw[4:]

    if event_type == _BINARY_UNENCODED_PREVIEW_IMAGE:
        return PreviewImage(image=payload, image_type=None, metadata=None)

    if event_type == _BINARY_PREVIEW_IMAGE:
        # 4-byte type header (1=JPEG, 2=PNG) then image data
        if len(payload) < 4:
            logger.warning("PREVIEW_IMAGE payload too short (%d bytes)", len(payload))
            return None
        img_type_num = struct.unpack(">I", payload[:4])[0]
        img_type = {1: "jpeg", 2: "png"}.get(img_type_num, None)
        return PreviewImage(image=payload[4:], image_type=img_type, metadata=None)

    if event_type == _BINARY_PREVIEW_IMAGE_WITH_METADATA:
        if len(payload) < 4:
            logger.warning(
                "PREVIEW_IMAGE_WITH_METADATA payload too short (%d bytes)",
                len(payload),
            )
            return None
        meta_len = struct.unpack(">I", payload[:4])[0]
        meta_end = 4 + meta_len
        meta: dict | None = None
        if meta_len > 0 and meta_end <= len(payload):
            meta = json.loads(payload[4:meta_end])
        img = payload[meta_end:]
        return PreviewImage(image=img, image_type=None, metadata=meta)

    logger.debug("Unknown binary event type: %s", event_type)
    return None
