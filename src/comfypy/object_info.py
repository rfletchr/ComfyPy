"""Typed, immutable object models for the ComfyUI ``/object_info`` API.

The raw ``GET /object_info`` response is a large, deeply-nested JSON mapping of node-class name to that node's metadata.
Navigating it with string keys is fragile and verbose, so this module turns it into a tree of frozen dataclasses --
:class:`NodeInfo`, :class:`InputDef`, and :class:`InputOptions` -- that callers can traverse with type-checked
attribute access.

Example::

    from comfypy import ComfyClient, parse_object_info

    raw = ComfyClient().get_object_info()
    nodes = parse_object_info(raw)

    sampler = nodes["KSampler"]
    print(sampler.display_name)
    for inp in sampler.input["required"].values():
        if inp.type == "INT":
            print(" ", inp.name, inp.options.min, inp.options.max)

Each section of :attr:`NodeInfo.input` is an ordered, immutable :class:`types.MappingProxyType` of input name ->
:class:`InputDef`, so declaration order is recoverable via ``tuple(node.input["required"])`` -- which lets us drop the
server's parallel ``input_order`` field as redundant.  The frozen dataclasses and immutable mappings together prevent
the parsed catalog from being mutated in place.  Long-tail widget-option keys not modelled on :class:`InputOptions`
are kept verbatim in :attr:`InputOptions.extra`, so custom-node data is never lost -- only structured.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: The three ``input`` section names ComfyUI itself emits.
REQUIRED = "required"
OPTIONAL = "optional"
HIDDEN = "hidden"

#: Canonical section keys that :attr:`NodeInfo.input` always materialises (even when a node omits them).
_KNOWN_SECTIONS = (REQUIRED, OPTIONAL, HIDDEN)

# Widget-options keys that get their own typed field on :class:`InputOptions`.
# Anything outside this set is preserved in ``InputOptions.extra``.
_KNOWN_OPT_KEYS = frozenset(
    {
        "default",
        "min",
        "max",
        "step",
        "round",
        "tooltip",
        "options",
        "advanced",
        "multiselect",
        "multiline",
        "display",
        "control_after_generate",
        "display_name",
        "dynamicPrompts",
        "template",
        "forceInput",
        "socketless",
        "image_upload",
        "widgetType",
        "file_upload",
        "lazy",
    }
)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class InputOptions:
    """Widget-options dict attached to a required/optional input.

    Every field is ``None`` when the node didn't set that key.  Only the keys ComfyUI emits are modelled here; anything
    else (custom-node extension keys, future keys) is kept verbatim in :attr:`extra`.

    Attributes:
        default (float | int | str | bool | list | dict | None): Default value, heterogeneous across node types.
        min (float | int | None): Numeric lower bound for INT/FLOAT widgets.
        max (float | int | None): Numeric upper bound for INT/FLOAT widgets.
        step (float | int | None): Drag/keyboard step for numeric widgets.
        round (float | int | bool | None): Quantisation increment for FLOAT widgets; a float (e.g. ``0.01``) rounds the
            value to that step and ``False`` disables rounding.  ``None`` when unset.
        tooltip (str | None): Human-readable description shown on hover.
        options (tuple[str, ...] | None): Menu choices for a ``COMBO`` widget whose type token is ``"COMBO"`` (separate
            from :attr:`InputDef.choices`, which holds the choices when the spec used an inline list).
        advanced (bool | None): Whether the widget is folded behind the node's advanced/extra-options UI.
        multiselect (bool | None): Whether a ``COMBO`` may hold more than one selected value.
        multiline (bool | None): Whether a ``STRING`` widget renders as a multi-line textarea.
        display (str | None): Display-mode hint for the widget (e.g. ``"number"`` vs ``"slider"``).
        control_after_generate (bool | None): Whether the widget re-seeds after each generation; carried by
            ``seed``-style inputs.
        display_name (str | None): Override label shown in the UI.
        dynamicPrompts (bool | None): Whether a ``STRING`` widget's content is parsed through the dynamic-prompts
            wildcard engine.
        template (str | None): Template string for the widget's value; semantics are widget-specific.
        forceInput (bool | None): Whether the slot must be fed by a connection rather than edited in-place as a widget.
        socketless (bool | None): Whether the input exposes no connectable socket (widget-only, never linked).
        image_upload (bool | None): Whether the widget shows an image-upload control.
        widgetType (str | None): Override for the frontend widget kind.
        file_upload (bool | None): Whether the widget shows a generic file-upload control.
        lazy (bool | None): Whether the node evaluates this input lazily (deferred until needed).
        extra (Mapping[str, object]): Widget-option keys not modelled above (custom-node extensions, etc.).  An
            immutable :class:`types.MappingProxyType`; empty when every key was recognised.
    """

    default: float | int | str | bool | list | dict | None = None
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    round: float | int | bool | None = None
    tooltip: str | None = None
    options: tuple[str, ...] | None = None
    advanced: bool | None = None
    multiselect: bool | None = None
    multiline: bool | None = None
    display: str | None = None
    control_after_generate: bool | None = None
    display_name: str | None = None
    dynamicPrompts: bool | None = None
    template: str | None = None
    forceInput: bool | None = None
    socketless: bool | None = None
    image_upload: bool | None = None
    widgetType: str | None = None
    file_upload: bool | None = None
    lazy: bool | None = None
    extra: Mapping[str, object] = field(default=MappingProxyType({}))


@dataclass(frozen=True, kw_only=True)
class InputDef:
    """A single named input on a node, from one section of the ``input`` block.

    The raw spec appears in two shapes:

    - ``[type_token, opts]`` -- e.g. ``["INT", {"default": 0, "min": 0}]``.  Sets :attr:`type` and :attr:`options`;
      :attr:`choices` is ``None``.
    - ``[choice_list, opts]`` -- e.g. ``[["euler", "dpm_2"], {}]``.  Sets :attr:`choices` and :attr:`options`;
      :attr:`type` is ``None``.

    ``hidden`` inputs may also be a bare type string (``"PROMPT"``), in which case only :attr:`type` is set and
    :attr:`options` is ``None``.

    A combo menu can be expressed either way: an inline list sets :attr:`choices`, while a ``"COMBO"`` type token sets
    :attr:`type` with the real choices in :attr:`InputOptions.options`.  Treat both uniformly as a combo when
    ``choices is not None`` **or** ``(type == "COMBO" and options.options is not None)``.

    Attributes:
        name (str): Input name as declared by the node; used as the prompt-API key.
        section (str): Source section -- one of :data:`REQUIRED`, :data:`OPTIONAL`, :data:`HIDDEN` (or a custom name).
        type (str | None): Widget/type token (e.g. ``"STRING"``, ``"INT"``, ``"MODEL"``, ``"COMBO"``); ``None`` when the
            spec used an inline choice list.
        choices (tuple[str, ...] | None): Inline menu choices when the spec used an inline list; ``None`` otherwise.
        options (InputOptions | None): Widget options; ``None`` for bare-string ``hidden`` specs.
        raw (object | None): Verbatim spec, set only when its shape matched no known form; lets a custom node's unusual
            input survive parsing for caller inspection.
    """

    name: str
    section: str
    type: str | None = None
    choices: tuple[str, ...] | None = None
    options: InputOptions | None = None
    raw: object | None = None


@dataclass(frozen=True, kw_only=True)
class NodeInfo:
    """Metadata for one registered node type -- a single ``/object_info`` entry.

    Attributes:
        name (str): Registered node-class identifier (the key in ``NODE_CLASS_MAPPINGS``).
        display_name (str): Human-readable label shown in the UI.
        description (str): Free-text node description; may be an empty string.
        category (str): Slash-delimited menu path, e.g. ``"model/sampling"``.
        python_module (str): Module the node class lives in (e.g. ``"nodes"``, ``"comfy_extras/nodes_custom_sample"``).
        input (Mapping[str, Mapping[str, InputDef]]): Section name -> (input name -> :class:`InputDef`); both levels are
            ordered, immutable :class:`types.MappingProxyType`.  Stock sections :data:`REQUIRED`, :data:`OPTIONAL`,
            :data:`HIDDEN` are always present (empty mappings if omitted); any non-standard section names a custom
            node emits appear as additional outer keys.  Section input order is recoverable via
            ``tuple(node.input["required"])``, subsuming the server's parallel ``input_order`` field.
        output (tuple[str, ...]): Return-type tokens, one per output socket.
        output_name (tuple[str, ...]): Per-output display labels (defaults to the type tokens when the node sets none).
        output_is_list (tuple[bool, ...]): Whether each output is a list-valued slot.
        is_input_list (bool): Whether the node consumes its required/optional inputs as lists.
        output_node (bool): Whether the node is a terminal/output node (ends graph output).
        has_intermediate_output (bool): Whether the node emits intermediate outputs mid-run.
        search_aliases (tuple[str, ...]): Extra search terms for the node picker; empty (not ``None``) when unset.
        output_tooltips (tuple[str, ...] | None): Hover tooltips, per output socket; ``None`` when the node sets none.
        essentials_category (str | None): Essentials-library grouping shown in the UI (e.g. ``"Basics"``).
        experimental (bool | None): When ``True`` the node is flagged experimental; ``None`` when the flag is unset.
        deprecated (bool | None): When ``True`` the node is flagged deprecated.
        dev_only (bool | None): When ``True`` the node is gated to developer mode.
        api_node (bool | None): Marks API-backed nodes (e.g. external service nodes).
        output_matchtypes (tuple[str, ...] | None): Input names whose type constrains output socket typing.
        price_badge (dict | None): Structured "price" expression for paid API nodes; shape is node-defined.
    """

    name: str
    display_name: str
    description: str
    category: str
    python_module: str

    input: Mapping[str, Mapping[str, InputDef]]
    output: tuple[str, ...]
    output_name: tuple[str, ...]
    output_is_list: tuple[bool, ...]

    is_input_list: bool
    output_node: bool
    has_intermediate_output: bool
    search_aliases: tuple[str, ...]

    output_tooltips: tuple[str, ...] | None = None
    essentials_category: str | None = None

    experimental: bool | None = None
    deprecated: bool | None = None
    dev_only: bool | None = None
    api_node: bool | None = None
    output_matchtypes: tuple[str, ...] | None = None
    price_badge: dict | None = None


# ------------------------------------------------------------------
# Public parsers
# ------------------------------------------------------------------


def parse_object_info(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, NodeInfo]:
    """Parse the full ``/object_info`` response into typed nodes.

    Args:
        raw (Mapping[str, Mapping[str, Any]]): JSON returned by ``GET /object_info`` (or
            ``GET /object_info/{node_class}``) -- a mapping of node-class name to that node's raw info dict.

    Returns:
        dict[str, NodeInfo]: Mapping of node-class name to a typed :class:`NodeInfo`, one entry per node in *raw*.
    """
    return {name: parse_node(info) for name, info in raw.items()}


def parse_node(raw: Mapping[str, Any]) -> NodeInfo:
    """Parse a single node's raw info dict.

    Args:
        raw (Mapping[str, Any]): One node's raw info dict -- the value side of the ``/object_info`` mapping.

    Returns:
        NodeInfo: The typed node metadata.
    """
    return NodeInfo(
        name=raw["name"],
        display_name=raw["display_name"],
        description=raw["description"],
        category=raw["category"],
        python_module=raw["python_module"],
        input=_parse_inputs(raw.get("input", {})),
        output=tuple(raw["output"]),
        output_name=tuple(raw["output_name"]),
        output_is_list=tuple(raw["output_is_list"]),
        is_input_list=bool(raw["is_input_list"]),
        output_node=bool(raw["output_node"]),
        has_intermediate_output=bool(raw["has_intermediate_output"]),
        search_aliases=_to_tuple(raw.get("search_aliases")),
        output_tooltips=_opt_tuple(raw.get("output_tooltips")),
        essentials_category=raw.get("essentials_category"),
        experimental=raw.get("experimental"),
        deprecated=raw.get("deprecated"),
        dev_only=raw.get("dev_only"),
        api_node=raw.get("api_node"),
        output_matchtypes=_opt_tuple(raw.get("output_matchtypes")),
        price_badge=raw.get("price_badge"),
    )


# ------------------------------------------------------------------
# Internal parsers
# ------------------------------------------------------------------


def _parse_inputs(raw: Mapping[str, Any]) -> Mapping[str, Mapping[str, InputDef]]:
    """Build the section->(name->InputDef) mapping, immutable on both levels.

    Always materialises :data:`REQUIRED`, :data:`OPTIONAL`, :data:`HIDDEN` (as empty mappings if absent), then folds in
    any non-standard section the node emits.  Iterating the returned mapping yields sections in canonical order followed
    by custom sections in their server-declared order.
    """
    out: dict[str, Mapping[str, InputDef]] = {}
    for section in _KNOWN_SECTIONS:
        out[section] = _parse_section(section, raw.get(section, {}))
    for section, entries in raw.items():
        if section in out:
            continue
        out[section] = _parse_section(section, entries)
    return MappingProxyType(out)


def _parse_section(section: str, entries: Mapping[str, Any]) -> Mapping[str, InputDef]:
    out: dict[str, InputDef] = {}
    for name, spec in entries.items():
        out[name] = _parse_input_def(name, section, spec)
    return MappingProxyType(out)


def _parse_input_def(name: str, section: str, spec: Any) -> InputDef:
    # Bare string -- a hidden input type token, e.g. "PROMPT".
    if isinstance(spec, str):
        return InputDef(name=name, section=section, type=spec)

    # Two-element list: [type_token | choice_list, options_dict].
    if isinstance(spec, list | tuple) and len(spec) == 2:
        first, opts = spec
        if isinstance(first, str):
            type_name: str | None = first
            choices: tuple[str, ...] | None = None
        else:
            type_name = None
            choices = tuple(first)
        options = _parse_options(opts) if isinstance(opts, dict) else None
        return InputDef(
            name=name,
            section=section,
            type=type_name,
            choices=choices,
            options=options,
        )

    # Unrecognised shape: preserve verbatim so callers can inspect `raw`.
    return InputDef(name=name, section=section, raw=spec)


def _parse_options(opts: Mapping[str, Any]) -> InputOptions:
    extra = {k: v for k, v in opts.items() if k not in _KNOWN_OPT_KEYS}
    return InputOptions(
        default=opts.get("default"),
        min=opts.get("min"),
        max=opts.get("max"),
        step=opts.get("step"),
        round=opts.get("round"),
        tooltip=opts.get("tooltip"),
        options=_opt_tuple(opts.get("options")),
        advanced=opts.get("advanced"),
        multiselect=opts.get("multiselect"),
        multiline=opts.get("multiline"),
        display=opts.get("display"),
        control_after_generate=opts.get("control_after_generate"),
        display_name=opts.get("display_name"),
        dynamicPrompts=opts.get("dynamicPrompts"),
        template=opts.get("template"),
        forceInput=opts.get("forceInput"),
        socketless=opts.get("socketless"),
        image_upload=opts.get("image_upload"),
        widgetType=opts.get("widgetType"),
        file_upload=opts.get("file_upload"),
        lazy=opts.get("lazy"),
        extra=MappingProxyType(extra),
    )


def _opt_tuple(value: Any | None) -> tuple[str, ...] | None:
    """Convert a list-or-None to a tuple, passing None through.

    Args:
        value (Any | None): A list of strings, or None.

    Returns:
        tuple[str, ...] | None: The values as a tuple, or None if *value* was None.
    """
    return tuple(value) if value is not None else None


def _to_tuple(value: Any | None) -> tuple[str, ...]:
    """Coerce a list-or-None to a tuple (None becomes empty).

    Args:
        value (Any | None): A list of strings, or None.

    Returns:
        tuple[str, ...]: The values as a tuple; empty when *value* was None.
    """
    return tuple(value) if value is not None else ()
