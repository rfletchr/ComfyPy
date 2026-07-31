import json
import urllib.request
import urllib.parse
import uuid

from comfypy import ws


__all__ = ["ComfyClient"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


class ComfyClient:
    """Synchronous HTTP client for the ComfyUI server.

    All methods map directly to the JSON-over-HTTP API.  No async, no
    framework dependency — the caller decides the concurrency model.

    Thread safety: each method call opens its own HTTP connection; no shared
    state across calls.  Instances are safe to share across threads.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8188", client_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self._ws = None  # lazily connected via _ensure_connected()

    # --------------------------------------------------------------
    # WebSocket
    # --------------------------------------------------------------

    def _ensure_connected(self):
        if self._ws is None:
            self._ws = ws.connect_websocket(self.base_url, self.client_id)

    def get_client_id(self) -> str:
        """Return the client ID, lazily connecting the WebSocket on first call."""
        self._ensure_connected()
        return self.client_id

    def iter_messages(self):
        """Iterate over typed events from the server.

        Each yielded value is a frozen dataclass instance —
        :class:`NodeExecuting`, :class:`NodeProgress`,
        :class:`ExecutionFinished`, etc.

        Blocks the calling thread on each ``recv()``.  The iterator stops when
        the WebSocket connection closes.
        """
        self._ensure_connected()
        while True:
            try:
                raw = self._ws.recv()
            except Exception:
                return
            event = ws.parse_message(raw)
            if event is None:
                continue
            if isinstance(event, list):
                yield from event
            else:
                yield event

    # --------------------------------------------------------------
    # System / info
    # --------------------------------------------------------------

    def get_embeddings(self) -> list[str]:
        """List text-encoder embedding names available on disk."""
        return get_json(self.base_url, "/embeddings")

    def get_models(self) -> list[str]:
        """List model-type folder names (e.g. "checkpoints", "loras", "vae")."""
        return get_json(self.base_url, "/models")

    def get_models_in_folder(self, folder: str) -> list[str]:
        """List model filenames in a specific model folder."""
        path = "/models/" + urllib.parse.quote(folder, safe="")
        return get_json(self.base_url, path)

    def get_extensions(self) -> list[str]:
        """List URL paths of available custom-node JS extensions."""
        return get_json(self.base_url, "/extensions")

    def get_system_stats(self) -> dict:
        """System statistics: OS, RAM, VRAM per device, versions."""
        return get_json(self.base_url, "/system_stats")

    def get_features(self) -> dict:
        """Server feature flags."""
        return get_json(self.base_url, "/features")

    # --------------------------------------------------------------
    # Node info
    # --------------------------------------------------------------

    def get_object_info(self) -> dict:
        """Metadata for every registered node type."""
        return get_json(self.base_url, "/object_info")

    def get_object_info_node(self, node_class: str) -> dict:
        """Metadata for a single node type."""
        path = "/object_info/" + urllib.parse.quote(node_class, safe="")
        return get_json(self.base_url, path)

    # --------------------------------------------------------------
    # View / metadata
    # --------------------------------------------------------------

    def view(
        self,
        filename: str,
        *,
        subfolder: str | None = None,
        type: str | None = None,
        preview: str | None = None,
        channel: str | None = None,
    ) -> bytes:
        """Download a file (image, video, etc.) from the server.

        Returns the raw bytes.  Use *preview* for a resized JPEG/WebP
        thumbnail (e.g. ``"webp;90"``).  Use *channel* to extract an
        alpha or RGB channel.
        """
        params = {"filename": filename}
        if subfolder is not None:
            params["subfolder"] = subfolder
        if type is not None:
            params["type"] = type
        if preview is not None:
            params["preview"] = preview
        if channel is not None:
            params["channel"] = channel
        qs = urllib.parse.urlencode(params)
        with urllib.request.urlopen(_build_url(self.base_url, "/view?" + qs)) as resp:
            return resp.read()

    def view_metadata(self, folder_name: str, filename: str) -> dict:
        """Read ``__metadata__`` from a safetensors file header."""
        path = "/view_metadata/" + urllib.parse.quote(folder_name, safe="")
        qs = urllib.parse.urlencode({"filename": filename})
        return get_json(self.base_url, path + "?" + qs)

    # --------------------------------------------------------------
    # Prompt / queue
    # --------------------------------------------------------------

    def get_prompt(self) -> dict:
        """Current prompt queue summary (queue_remaining)."""
        return get_json(self.base_url, "/prompt")

    def get_queue(self) -> dict:
        """Full queue: ``queue_running`` and ``queue_pending``."""
        return get_json(self.base_url, "/queue")

    def post_prompt(
        self,
        prompt: dict,
        *,
        extra_data: dict | None = None,
        prompt_id: str | None = None,
        number: float | None = None,
        front: bool = False,
        partial_execution_targets: list[str] | None = None,
    ) -> dict:
        """Submit a workflow for execution.

        The client ID is automatically included so execution events route
        to this client's WebSocket.  Call :meth:`connect` first, or this
        method will lazy-connect on first use.

        Returns ``{"prompt_id": …, "number": …, "node_errors": …}``.
        """
        body: dict = {
            "prompt": prompt,
            "client_id": self.get_client_id(),
        }
        if extra_data is not None:
            body["extra_data"] = extra_data
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if number is not None:
            body["number"] = number
        if front:
            body["front"] = True
        if partial_execution_targets is not None:
            body["partial_execution_targets"] = partial_execution_targets
        return post_json(self.base_url, "/prompt", body)

    def post_queue(
        self, *, clear: bool = False, delete: list[str] | None = None
    ) -> None:
        """Clear the queue and/or delete specific pending items by prompt ID."""
        body = {}
        if clear:
            body["clear"] = True
        if delete is not None:
            body["delete"] = delete
        _post_no_body(self.base_url, "/queue", body)

    def post_interrupt(self, prompt_id: str | None = None) -> None:
        """Interrupt execution (optionally scoped to a single prompt)."""
        body = {}
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        _post_no_body(self.base_url, "/interrupt", body)

    def post_free(
        self, *, unload_models: bool = False, free_memory: bool = False
    ) -> None:
        """Free GPU memory: unload models and/or force garbage collection."""
        body = {}
        if unload_models:
            body["unload_models"] = True
        if free_memory:
            body["free_memory"] = True
        _post_no_body(self.base_url, "/free", body)

    # --------------------------------------------------------------
    # History
    # --------------------------------------------------------------

    def get_history(
        self, *, max_items: int | None = None, offset: int | None = None
    ) -> dict:
        """Execution history, newest first."""
        params: dict[str, str] = {}
        if max_items is not None:
            params["max_items"] = str(max_items)
        if offset is not None:
            params["offset"] = str(offset)
        qs = urllib.parse.urlencode(params)
        path = "/history" + ("?" + qs if qs else "")
        return get_json(self.base_url, path)

    def get_history_prompt(self, prompt_id: str) -> dict:
        """History for a single prompt."""
        path = "/history/" + urllib.parse.quote(prompt_id, safe="")
        return get_json(self.base_url, path)

    def post_history(
        self, *, clear: bool = False, delete: list[str] | None = None
    ) -> None:
        """Clear the entire history and/or delete specific entries."""
        body = {}
        if clear:
            body["clear"] = True
        if delete is not None:
            body["delete"] = delete
        _post_no_body(self.base_url, "/history", body)

    # --------------------------------------------------------------
    # Upload
    # --------------------------------------------------------------

    def upload_image(
        self,
        image: bytes,
        filename: str,
        *,
        subfolder: str = "",
        type: str = "input",
        overwrite: bool = False,
    ) -> dict:
        """Upload an image to the server.

        Returns ``{"name": …, "subfolder": …, "type": …}``.
        """
        parts = [
            ("image", image, filename),
            ("subfolder", subfolder, None),
            ("type", type, None),
        ]
        if overwrite:
            parts.append(("overwrite", "true", None))
        return post_multipart(self.base_url, "/upload/image", parts)

    def upload_mask(self, image: bytes, filename: str, original_ref: str) -> dict:
        """Upload a mask, compositing it onto an existing image.

        *original_ref* is a JSON string with ``filename``, ``type``, and
        optionally ``subfolder`` of the source image.

        Returns ``{"name": …, "subfolder": …, "type": …}``.
        """
        parts = [
            ("image", image, filename),
            ("original_ref", original_ref, None),
        ]
        return post_multipart(self.base_url, "/upload/mask", parts)

    # --------------------------------------------------------------
    # Jobs API
    # --------------------------------------------------------------

    def get_jobs(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """List jobs with filtering and pagination.

        Returns ``{"jobs": […], "pagination": {…}}``.
        """
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = status
        if workflow_id is not None:
            params["workflow_id"] = workflow_id
        if sort_by is not None:
            params["sort_by"] = sort_by
        if sort_order is not None:
            params["sort_order"] = sort_order
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        qs = urllib.parse.urlencode(params)
        path = "/api/jobs" + ("?" + qs if qs else "")
        return get_json(self.base_url, path)

    def get_job(self, job_id: str) -> dict:
        """Full detail for a single job."""
        path = "/api/jobs/" + urllib.parse.quote(job_id, safe="")
        return get_json(self.base_url, path)

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a single job.

        Returns ``{"cancelled": bool}``.  Idempotent — calling on an
        already-finished job returns ``{"cancelled": false}``.
        """
        path = "/api/jobs/" + urllib.parse.quote(job_id, safe="") + "/cancel"
        return post_json(self.base_url, path)

    def cancel_jobs(self, job_ids: list[str]) -> dict:
        """Cancel multiple jobs in one request.

        Returns ``{"cancelled": bool}``.
        """
        return post_json(self.base_url, "/api/jobs/cancel", {"job_ids": job_ids})


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_url(base_url: str, path: str) -> str:
    return base_url + path


def get_json(base_url: str, path: str):
    with urllib.request.urlopen(_build_url(base_url, path)) as resp:
        return json.load(resp)


def post_json(base_url: str, path: str, body: dict | None = None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        _build_url(base_url, path), data=data, headers=headers
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _post_no_body(base_url: str, path: str, body: dict | None = None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        _build_url(base_url, path), data=data, headers=headers
    )
    urllib.request.urlopen(req)


def post_multipart(
    base_url: str,
    path: str,
    parts: list[tuple[str, str | bytes, str | None]],
):
    """Send a multipart/form-data POST.

    Each part is (name, value, filename | None).  If *filename* is ``None``
    the part is a text field; otherwise it is a file field with *value* as
    the raw bytes.
    """
    boundary = "----FormBoundary" + uuid.uuid4().hex
    body_bytes = b""

    for name, value, filename in parts:
        body_bytes += ("--" + boundary + "\r\n").encode()
        if filename is not None:
            body_bytes += (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n"
                f"\r\n"
            ).encode()
            if isinstance(value, str):
                value = value.encode()
            body_bytes += value + b"\r\n"
        else:
            body_bytes += (
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            ).encode()

    body_bytes += ("--" + boundary + "--\r\n").encode()

    req = urllib.request.Request(
        _build_url(base_url, path),
        data=body_bytes,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())
