#!/usr/bin/env python3
"""Free-model aggregator entry for make.fun free-model-aggregator-01.

One long-running server. It listens on 0.0.0.0 at a port it chooses, registers
with the trial env, and answers the env client's 20 requests through the ten
simulated free-tier providers behind the env. For every request it picks the
strongest free model that meets the request's requirements, speaks that
provider's wire (OpenAI / Anthropic / Gemini), and fails over to another
provider when one refuses (402 / 429 / 5xx). It returns the provider's answer
in the wire the client used, with the provider's text whole, its usage counts
as that wire reports them, and its stop reason mapped onto the client's wire.

Python 3.13 standard library only. No wall clock, no unseeded randomness:
every route is a function of the requests and the providers' answers seen so
far, so a replay leaves identical state.

Run with --list: print the catalog as one JSON document and exit 0.
With neither MAKEFUN_PROMPT_FILE nor MAKEFUN_ENV_URL set, --list prints
{"providers": []} and exits 0.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlparse

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600            # fixed epoch for every listed model
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
CLIENT_TIMEOUT = 25.0                # seconds for one upstream provider call
REGISTER_RETRIES = 12
WORK_POLL_SECONDS = 0.1
WORK_TIMEOUT_SECONDS = 170.0
MAX_UPSTREAM_ATTEMPTS = 10

# Routing penalties. A provider's calls all spend its free-tier budget, so
# the router spreads load across near-best providers instead of hammering
# the single best one until it is exhausted. A failure costs more than a
# call, so a provider that just refused is not the first pick for the next
# request, but the penalty stays small enough that one scheduled failure
# does not hand the request to a much weaker model: the provider usually
# still has budget after a single refusal.
CALL_PENALTY = 2
FAILURE_PENALTY = 15

# Stop-reason families, shared by every wire. A provider only ever answers
# with the family-0 stop in this simulation, but the mapping is general.
STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
CLIENT_STOP = {"openai": "stop", "anthropic": "end_turn"}
PROVIDER_STOP = {"openai": "stop", "anthropic": "end_turn", "gemini": "STOP"}

# ---------------------------------------------------------------- helpers


def is_record(value):
    return isinstance(value, dict)


def text_of_blocks(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def number_or_none(value):
    return value if isinstance(value, (int, float)) and value == value else None


def log(message):
    sys.stderr.write("free-model-aggregator-01 entry: %s\n" % message)
    sys.stderr.flush()


def json_bytes(value):
    return (json.dumps(value) + "\n").encode("utf-8")


def sse_frame(event, data):
    """One server-sent event frame: `event: <name>` when named, then data."""
    head = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (head, json.dumps(data))


def split_text(text, width=13):
    """Split text into fixed-width pieces, as the simulated providers do."""
    if text == "":
        return [""]
    return [text[i:i + width] for i in range(0, len(text), width)]


# ------------------------------------------------------------ brief parsing

def find_json_document(text, schema):
    """Find and parse the JSON document whose schema field equals `schema`.

    The brief carries three documents (provider directory, quality table,
    workload manifest) as plain text after a heading line. The document is
    located by its schema string and parsed with a raw decoder so nothing
    after it matters.
    """
    marker = '"schema": "%s"' % schema
    at = text.find(marker)
    if at < 0:
        marker = '"schema":"%s"' % schema
        at = text.find(marker)
    if at < 0:
        return None
    start = text.rfind("{", 0, at)
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(text[start:])
    except (ValueError, TypeError):
        return None
    return value if is_record(value) else None


def load_brief(prompt_path):
    """Read the brief file and return (providers, ratings, manifest)."""
    if not prompt_path or not os.path.isfile(prompt_path):
        return None, None, None
    with open(prompt_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    providers = find_json_document(text, "free-model-aggregator-01/providers/v1")
    ratings = find_json_document(text, "free-model-aggregator-01/ratings/v1")
    workload = find_json_document(text, "free-model-aggregator-01/workload/v1")
    manifest = None
    if is_record(workload) and isinstance(workload.get("manifest"), list):
        manifest = workload["manifest"]
    return providers, ratings, manifest


def http_json(method, url, headers=None, body=None, timeout=10.0):
    """One HTTP call returning (status, parsed JSON or None, raw bytes)."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read()
        status = err.code
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        raise ConnectionError("cannot reach %s: %s" % (url, err))
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = None
    return status, parsed, raw


def fetch_instance(env_url):
    """GET the env's /instance: provider directory (with keys) + manifest."""
    status, parsed, _raw = http_json("GET", env_url.rstrip("/") + "/instance", timeout=10.0)
    if status != 200 or not is_record(parsed):
        raise ConnectionError("GET /instance answered HTTP %s" % status)
    providers = parsed.get("providers")
    manifest = None
    workload = parsed.get("workload")
    if is_record(workload) and isinstance(workload.get("manifest"), list):
        manifest = workload["manifest"]
    if not isinstance(providers, list):
        raise ConnectionError("GET /instance holds no provider directory")
    return providers, manifest


# ------------------------------------------------------------ provider access


def auth_headers(provider):
    """The auth header for one provider, from its auth scheme and key."""
    auth = provider.get("auth")
    key = provider.get("key") or ""
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_base(env_url, provider):
    """The provider's own base: the env URL joined to its path."""
    path = provider.get("path") or provider.get("base_url") or ""
    if path.startswith("http://") or path.startswith("https://"):
        return path.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return env_url.rstrip("/") + path


def catalog_url(env_url, provider):
    """The provider's catalog route for its wire."""
    base = provider_base(env_url, provider)
    if provider.get("wire") == "gemini":
        return base + "/v1beta/models"
    return base + "/v1/models"


def probe_catalog(env_url, provider):
    """GET one provider's catalog; return its model rows in listed order."""
    headers = dict(auth_headers(provider))
    if provider.get("wire") == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, parsed, _raw = http_json("GET", catalog_url(env_url, provider),
                                     headers=headers, timeout=10.0)
    if status != 200 or not is_record(parsed):
        raise ConnectionError("catalog probe for %s answered HTTP %s"
                              % (provider.get("id"), status))
    rows = []
    if isinstance(parsed.get("data"), list):
        items = parsed["data"]
    elif isinstance(parsed.get("models"), list):
        items = parsed["models"]
    else:
        raise ConnectionError("catalog probe for %s holds no model list"
                              % provider.get("id"))
    for item in items:
        if not is_record(item):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or model_id == "":
            # Gemini lists models as `models/<id>` in `name`, with no `id`.
            name = item.get("name")
            model_id = name.split("/")[-1] if isinstance(name, str) and name else None
        if not isinstance(model_id, str) or model_id == "":
            continue
        rows.append({
            "id": model_id,
            "free": item.get("free") is True,
            "context_length": item.get("context_length"),
            "tools": item.get("tools") is True,
            "vision": item.get("vision") is True,
            "json": item.get("json") is True,
        })
    return rows


def build_catalog(env_url, directory, quality):
    """Build the catalog document to register and to print with --list.

    Providers stay in directory order and every provider's models stay in
    the order its own catalog lists them, because the catalog score compares
    each provider's free-model list in order. Ratings come from the brief's
    quality table; free flags and context lengths come from the probe, with
    the quality table as fallback when a probe cannot be read.
    """
    ratings = {}
    if is_record(quality) and isinstance(quality.get("ratings"), list):
        for row in quality["ratings"]:
            if is_record(row) and isinstance(row.get("model"), str):
                ratings[(row.get("provider"), row["model"])] = row
    providers_out = []
    probed = {}
    for provider in directory:
        pid = provider.get("id")
        if not isinstance(pid, str):
            continue
        rows = None
        if env_url:
            try:
                rows = probe_catalog(env_url, provider)
            except ConnectionError as err:
                log("catalog probe for %s failed: %s" % (pid, err))
        if rows is None:
            rows = []
            for (prov, model_id), row in ratings.items():
                if prov == pid:
                    rows.append({
                        "id": model_id,
                        "free": row.get("free") is True,
                        "context_length": row.get("context_length"),
                        "tools": row.get("tools") is True,
                        "vision": row.get("vision") is True,
                        "json": row.get("json") is True,
                    })
        probed[pid] = rows
        base_url = provider.get("path") or provider.get("base_url") or ""
        if base_url.startswith("http://") or base_url.startswith("https://"):
            # The catalog reports the provider's own path, never joined.
            base_url = urlparse(base_url).path or base_url
        models = []
        for row in rows:
            entry = {"id": row["id"], "free": bool(row["free"])}
            context = row.get("context_length")
            entry["context_length"] = context if isinstance(context, int) else None
            models.append(entry)
        providers_out.append({
            "id": pid,
            "base_url": base_url,
            "key_env": provider.get("key_env") or "",
            "wire": provider.get("wire") or "",
            "models": models,
        })
    return {"providers": providers_out}, probed


def free_model_ids(catalog):
    """Every free model id the catalog registers, in catalog order."""
    ids = []
    for provider in catalog.get("providers", []):
        for model in provider.get("models", []):
            if is_record(model) and model.get("free") is True:
                ids.append(model["id"])
    return ids


# ------------------------------------------------------- wire: request build

def openai_system_text(body):
    parts = []
    for message in body.get("messages") or []:
        if is_record(message) and message.get("role") == "system":
            text = text_of_blocks(message.get("content"))
            if text:
                parts.append(text)
    return "\n".join(parts)


def openai_user_parts(body):
    """The user message's parts as (kind, value): text or image data URL."""
    parts = []
    for message in body.get("messages") or []:
        if not is_record(message) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content:
                parts.append(("text", content))
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(("text", block["text"]))
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                if isinstance(url, str) and url:
                    parts.append(("image_url", url))
    return parts


def openai_tool_defs(body):
    """OpenAI tools as (name, description, parameters) triples."""
    defs = []
    for tool in body.get("tools") or []:
        if not is_record(tool):
            continue
        fn = tool.get("function")
        if not is_record(fn) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        defs.append((fn["name"],
                     fn.get("description") if isinstance(fn.get("description"), str) else "",
                     params if is_record(params) else {"type": "object", "properties": {}}))
    return defs


def anthropic_tool_defs(body):
    defs = []
    for tool in body.get("tools") or []:
        if not is_record(tool) or not isinstance(tool.get("name"), str):
            continue
        schema = tool.get("input_schema")
        defs.append((tool["name"],
                     tool.get("description") if isinstance(tool.get("description"), str) else "",
                     schema if is_record(schema) else {"type": "object", "properties": {}}))
    return defs


def image_data_url_parts(url):
    """Split a data URL into (mime_type, base64) or (None, url) for a plain URL."""
    if url.startswith("data:"):
        head, _, data = url[5:].partition(",")
        mime = head.split(";")[0] or "image/png"
        return mime, data
    return None, url


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of_blocks(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            calls = []
            for block in content if isinstance(content, list) else []:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of_blocks(content)
            message = {"role": "assistant", "content": text if text else None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        if not isinstance(content, list):
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url",
                                  "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
            elif block.get("type") == "tool_result":
                result = block.get("content")
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": text_of_blocks(result) if result is not None else "",
                })
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = [s for s in stops if isinstance(s, str)]
    defs = anthropic_tool_defs(body)
    if defs:
        out["tools"] = [{"type": "function", "function": {
            "name": name, "description": desc, "parameters": params}}
            for name, desc, params in defs]
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request.

    The Anthropic wire carries no response_format, so the router never sends
    a JSON-mode request down this path.
    """
    system = openai_system_text(body)
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            continue
        if role == "assistant":
            blocks = []
            text = text_of_blocks(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call):
                    continue
                fn = call.get("function") if is_record(call.get("function")) else {}
                args = fn.get("arguments")
                parsed = {}
                if isinstance(args, str):
                    try:
                        value = json.loads(args)
                        parsed = value if is_record(value) else {}
                    except ValueError:
                        parsed = {}
                elif is_record(args):
                    parsed = args
                blocks.append({"type": "tool_use",
                               "id": call.get("id") if isinstance(call.get("id"), str) else "",
                               "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                               "input": parsed})
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": content if isinstance(content, str) else text_of_blocks(content),
            }]})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        blocks = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                if not isinstance(url, str):
                    continue
                mime, data = image_data_url_parts(url)
                if mime is None:
                    blocks.append({"type": "image",
                                   "source": {"type": "url", "url": data}})
                else:
                    blocks.append({"type": "image",
                                   "source": {"type": "base64", "media_type": mime, "data": data}})
        if blocks:
            messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system:
        out["system"] = system
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    if isinstance(stops, list) and stops:
        out["stop_sequences"] = [s for s in stops if isinstance(s, str)]
    defs = openai_tool_defs(body)
    if defs:
        out["tools"] = [{"name": name, "description": desc, "input_schema": params}
                        for name, desc, params in defs]
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def gemini_contents_from_openai(body):
    """OpenAI messages as Gemini contents (user parts only carry the echo)."""
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system" or role == "tool":
            continue
        parts = []
        content = raw.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    if not isinstance(url, str):
                        continue
                    mime, data = image_data_url_parts(url)
                    if mime is None:
                        parts.append({"fileData": {"mimeType": "image/png", "fileUri": data}})
                    else:
                        parts.append({"inlineData": {"mimeType": mime, "data": data}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return contents


def gemini_contents_from_anthropic(body):
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                        parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"mimeType": "image/png", "fileUri": source["url"]}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return contents


def openai_to_gemini(body, model):
    """An OpenAI Chat Completions request as a Gemini generateContent body."""
    out = {}
    system = openai_system_text(body)
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    contents = gemini_contents_from_openai(body)
    if contents:
        out["contents"] = contents
    defs = openai_tool_defs(body)
    if defs:
        out["tools"] = [{"functionDeclarations": [
            {"name": name, "description": desc, "parameters": params}
            for name, desc, params in defs]}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    if isinstance(stops, list) and stops:
        config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out, body.get("stream") is True


def anthropic_to_gemini(body, model):
    """An Anthropic Messages request as a Gemini generateContent body."""
    out = {}
    system = text_of_blocks(body.get("system"))
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    contents = gemini_contents_from_anthropic(body)
    if contents:
        out["contents"] = contents
    defs = anthropic_tool_defs(body)
    if defs:
        out["tools"] = [{"functionDeclarations": [
            {"name": name, "description": desc, "parameters": params}
            for name, desc, params in defs]}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    if config:
        out["generationConfig"] = config
    return out, body.get("stream") is True


# ------------------------------------------------------ wire: response parse


def parse_sse_frames(raw_text):
    """Split a raw SSE body into (event, data) frames, in order."""
    frames = []
    normalized = raw_text.replace("\r\n", "\n")
    for chunk in normalized.split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip(" "))
        if data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


def stop_family(reason):
    if reason is None:
        return None
    return STOP_FAMILIES.get(reason)


def parse_openai_response(raw, is_stream):
    """An OpenAI provider answer as (text, input_tokens, output_tokens, family)."""
    text = ""
    in_tokens = None
    out_tokens = None
    family = None
    if not is_stream:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return text, in_tokens, out_tokens, family
        if not is_record(value):
            return text, in_tokens, out_tokens, family
        choices = value.get("choices")
        if isinstance(choices, list) and choices and is_record(choices[0]):
            choice = choices[0]
            message = choice.get("message") if is_record(choice.get("message")) else {}
            content = message.get("content")
            text = content if isinstance(content, str) else text_of_blocks(content)
            family = stop_family(choice.get("finish_reason"))
        usage = value.get("usage") if is_record(value.get("usage")) else {}
        in_tokens = number_or_none(usage.get("prompt_tokens"))
        out_tokens = number_or_none(usage.get("completion_tokens"))
        return text, in_tokens, out_tokens, family
    frames = parse_sse_frames(raw.decode("utf-8", "replace"))
    for _event, data in frames:
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        usage = chunk.get("usage") if is_record(chunk.get("usage")) else None
        if usage is not None:
            in_tokens = number_or_none(usage.get("prompt_tokens"))
            out_tokens = number_or_none(usage.get("completion_tokens"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not is_record(choice):
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        content = delta.get("content")
        if isinstance(content, str):
            text += content
        reason = choice.get("finish_reason")
        if isinstance(reason, str):
            family = stop_family(reason)
    return text, in_tokens, out_tokens, family


def parse_anthropic_response(raw, is_stream):
    """An Anthropic provider answer as (text, input_tokens, output_tokens, family)."""
    text = ""
    in_tokens = None
    out_tokens = None
    family = None
    if not is_stream:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return text, in_tokens, out_tokens, family
        if not is_record(value):
            return text, in_tokens, out_tokens, family
        text = text_of_blocks(value.get("content"))
        usage = value.get("usage") if is_record(value.get("usage")) else {}
        in_tokens = number_or_none(usage.get("input_tokens"))
        out_tokens = number_or_none(usage.get("output_tokens"))
        family = stop_family(value.get("stop_reason"))
        return text, in_tokens, out_tokens, family
    frames = parse_sse_frames(raw.decode("utf-8", "replace"))
    for event, data in frames:
        try:
            value = json.loads(data)
        except ValueError:
            continue
        if not is_record(value):
            continue
        if event == "message_start":
            message = value.get("message") if is_record(value.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            in_tokens = number_or_none(usage.get("input_tokens"))
        elif event == "content_block_delta":
            delta = value.get("delta") if is_record(value.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
        elif event == "message_delta":
            delta = value.get("delta") if is_record(value.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                family = stop_family(delta["stop_reason"])
            usage = value.get("usage") if is_record(value.get("usage")) else {}
            out_tokens = number_or_none(usage.get("output_tokens"))
    return text, in_tokens, out_tokens, family


def gemini_parts_text(parts):
    if not isinstance(parts, list):
        return ""
    return "".join(p.get("text", "") for p in parts
                   if is_record(p) and isinstance(p.get("text"), str))


def parse_gemini_response(raw, is_stream):
    """A Gemini provider answer as (text, input_tokens, output_tokens, family)."""
    text = ""
    in_tokens = None
    out_tokens = None
    family = None
    if not is_stream:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return text, in_tokens, out_tokens, family
        if not is_record(value):
            return text, in_tokens, out_tokens, family
        candidates = value.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            text = gemini_parts_text(content.get("parts"))
            if isinstance(candidate.get("finishReason"), str):
                family = stop_family(candidate["finishReason"])
        meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
        in_tokens = number_or_none(meta.get("promptTokenCount"))
        out_tokens = number_or_none(meta.get("candidatesTokenCount"))
        return text, in_tokens, out_tokens, family
    frames = parse_sse_frames(raw.decode("utf-8", "replace"))
    for _event, data in frames:
        try:
            value = json.loads(data)
        except ValueError:
            continue
        if not is_record(value):
            continue
        candidates = value.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            text += gemini_parts_text(content.get("parts"))
            if isinstance(candidate.get("finishReason"), str):
                family = stop_family(candidate["finishReason"])
        meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else None
        if meta is not None:
            in_tokens = number_or_none(meta.get("promptTokenCount"))
            out_tokens = number_or_none(meta.get("candidatesTokenCount"))
    return text, in_tokens, out_tokens, family


def parse_provider_response(wire, raw, is_stream):
    if wire == "openai":
        return parse_openai_response(raw, is_stream)
    if wire == "anthropic":
        return parse_anthropic_response(raw, is_stream)
    return parse_gemini_response(raw, is_stream)


# -------------------------------------------------------- wire: response emit


def stop_for_client(family, client_api):
    """The client's stop reason for a provider stop family."""
    if family == 1:
        return "length" if client_api == "openai" else "max_tokens"
    if family == 2:
        return "tool_calls" if client_api == "openai" else "tool_use"
    if family == 3:
        return "content_filter" if client_api == "openai" else "refusal"
    return CLIENT_STOP[client_api]


def emit_openai_json(model, text, in_tokens, out_tokens, family):
    return json_bytes({
        "id": "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": stop_for_client(family, "openai"),
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": in_tokens if in_tokens is not None else 0,
            "completion_tokens": out_tokens if out_tokens is not None else 0,
            "total_tokens": (in_tokens or 0) + (out_tokens or 0),
        },
    })


def emit_openai_sse(model, text, in_tokens, out_tokens, family):
    """OpenAI event stream: role chunk, content pieces, finish, usage, [DONE]."""
    stop = stop_for_client(family, "openai")
    frames = []

    def chunk(delta, finish):
        return {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames.append(sse_frame(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in split_text(text):
        frames.append(sse_frame(None, chunk({"content": piece}, None)))
    frames.append(sse_frame(None, chunk({}, stop)))
    frames.append(sse_frame(None, {
        "id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model, "choices": [],
        "usage": {
            "prompt_tokens": in_tokens if in_tokens is not None else 0,
            "completion_tokens": out_tokens if out_tokens is not None else 0,
            "total_tokens": (in_tokens or 0) + (out_tokens or 0),
        },
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def emit_anthropic_json(model, text, in_tokens, out_tokens, family):
    return json_bytes({
        "id": "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_for_client(family, "anthropic"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": in_tokens if in_tokens is not None else 0,
            "output_tokens": out_tokens if out_tokens is not None else 0,
        },
    })


def emit_anthropic_sse(model, text, in_tokens, out_tokens, family):
    """Anthropic event order: start, block start, deltas, stop, delta, stop."""
    stop = stop_for_client(family, "anthropic")
    frames = []
    frames.append(sse_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_aggregate", "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tokens if in_tokens is not None else 0,
                      "output_tokens": 0},
        },
    }))
    frames.append(sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in split_text(text):
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(sse_frame("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    }))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": out_tokens if out_tokens is not None else 0},
    }))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames).encode("utf-8")


def emit_client_response(client_api, is_stream, model, text, in_tokens, out_tokens, family):
    if client_api == "openai":
        if is_stream:
            return ("text/event-stream; charset=utf-8",
                    emit_openai_sse(model, text, in_tokens, out_tokens, family))
        return ("application/json",
                emit_openai_json(model, text, in_tokens, out_tokens, family))
    if is_stream:
        return ("text/event-stream; charset=utf-8",
                emit_anthropic_sse(model, text, in_tokens, out_tokens, family))
    return ("application/json",
            emit_anthropic_json(model, text, in_tokens, out_tokens, family))


# ------------------------------------------------------------------- router


def body_requirements(client_api, body):
    """The requirements a client body itself reveals: tools, vision, json."""
    req = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        req["tools"] = True
    if client_api == "openai":
        if any(kind == "image_url" for kind, _v in openai_user_parts(body)):
            req["vision"] = True
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            req["json"] = True
    else:
        for message in body.get("messages") or []:
            if not is_record(message):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if is_record(block) and block.get("type") == "image":
                    req["vision"] = True
    return req


def model_meets(row, req):
    if req.get("tools") is True and not row.get("tools"):
        return False
    if req.get("vision") is True and not row.get("vision"):
        return False
    if req.get("json") is True and not row.get("json"):
        return False
    min_context = req.get("min_context")
    if isinstance(min_context, int) and min_context > 0:
        context = row.get("context_length")
        if not isinstance(context, int) or context < min_context:
            return False
    return True


class Router:
    """Picks the serving model per request and fails over on refusal."""

    def __init__(self, directory, quality, probed, manifest):
        self.directory = {p["id"]: p for p in directory if isinstance(p.get("id"), str)}
        self.manifest = manifest or []
        self.calls = {pid: 0 for pid in self.directory}
        self.failures = {pid: 0 for pid in self.directory}
        self.request_index = 0
        self.lock = threading.Lock()
        # The free-model table: one row per (provider, model) that is free.
        self.models = []
        ratings = {}
        if is_record(quality) and isinstance(quality.get("ratings"), list):
            for row in quality["ratings"]:
                if is_record(row) and isinstance(row.get("model"), str):
                    ratings[(row.get("provider"), row["model"])] = row
        for provider in directory:
            pid = provider.get("id")
            if not isinstance(pid, str):
                continue
            for row in probed.get(pid, []):
                if not row.get("free"):
                    continue
                rated = ratings.get((pid, row["id"])) or {}
                self.models.append({
                    "provider": pid,
                    "wire": provider.get("wire") or "",
                    "model": row["id"],
                    "rating": rated.get("rating_milli") if isinstance(rated.get("rating_milli"), int) else 0,
                    "context_length": rated.get("context_length") if isinstance(rated.get("context_length"), int) else row.get("context_length"),
                    "tools": rated.get("tools") is True if "tools" in rated else bool(row.get("tools")),
                    "vision": rated.get("vision") is True if "vision" in rated else bool(row.get("vision")),
                    "json": rated.get("json") is True if "json" in rated else bool(row.get("json")),
                })

    def requirements(self, client_api, body, index):
        """The request's requirements: the manifest entry by arrival order,
        unioned with what the body itself reveals (tools, vision, json)."""
        req = {}
        if 0 <= index < len(self.manifest):
            entry = self.manifest[index]
            if is_record(entry):
                manifest_req = entry.get("requires")
                if is_record(manifest_req):
                    req.update(manifest_req)
                if entry.get("api") != client_api or entry.get("stream") != (body.get("stream") is True):
                    log("request %d: manifest says %s/stream=%s, body says %s/stream=%s"
                        % (index + 1, entry.get("api"), entry.get("stream"),
                           client_api, body.get("stream") is True))
        req.update(body_requirements(client_api, body))
        return req

    def rank(self, req, client_api):
        """Candidate (provider, model) rows, best first, load-balanced."""
        json_needs_wire = client_api == "openai" and req.get("json") is True
        pool = []
        for row in self.models:
            if not model_meets(row, req):
                continue
            if json_needs_wire and row["wire"] == "anthropic":
                # The Anthropic wire carries no response_format.
                continue
            pool.append(row)
        if not pool:
            return []

        def key(row):
            pid = row["provider"]
            score = row["rating"] - CALL_PENALTY * self.calls.get(pid, 0) \
                - FAILURE_PENALTY * self.failures.get(pid, 0)
            return (-score, self.calls.get(pid, 0), -row["rating"], pid, row["model"])

        return sorted(pool, key=key)

    def upstream(self, provider, model, client_api, body):
        """One provider call. Returns (status, text, in, out, family) or None."""
        wire = provider.get("wire")
        stream = body.get("stream") is True
        base = provider_base(STATE.env_url, provider)
        headers = dict(auth_headers(provider))
        if wire == "openai":
            url = base + "/v1/chat/completions"
            if client_api == "anthropic":
                # The client speaks Anthropic; the provider speaks OpenAI.
                upstream_body = anthropic_to_openai(body, model)
            else:
                upstream_body = dict(body)
                upstream_body["model"] = model
                if stream:
                    upstream_body["stream"] = True
                    upstream_body["stream_options"] = {"include_usage": True}
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = ANTHROPIC_VERSION
            if client_api == "anthropic":
                upstream_body = dict(body)
                upstream_body["model"] = model
            else:
                upstream_body = openai_to_anthropic(body, model)
        else:  # gemini
            action = "streamGenerateContent" if stream else "generateContent"
            url = "%s/v1beta/models/%s:%s" % (base, quote(model, safe=""), action)
            if client_api == "anthropic":
                upstream_body, _s = anthropic_to_gemini(body, model)
            else:
                upstream_body, _s = openai_to_gemini(body, model)
        data = json.dumps(upstream_body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=CLIENT_TIMEOUT) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as err:
            raw = err.read()
            status = err.code
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            raise ConnectionError("upstream %s unreachable: %s" % (provider.get("id"), err))
        if status < 200 or status >= 300:
            return status, None
        text, in_tokens, out_tokens, family = parse_provider_response(wire, raw, stream)
        return status, (text, in_tokens, out_tokens, family)

    def serve(self, client_api, body):
        """Serve one client request; returns (content_type, body_bytes)."""
        with self.lock:
            index = self.request_index
            self.request_index += 1
            req = self.requirements(client_api, body, index)
            ranked = self.rank(req, client_api)
        if not ranked:
            raise RuntimeError("no free model meets the request's requirements")
        tried = set()
        last_error = None
        for attempt, row in enumerate(ranked):
            if attempt >= MAX_UPSTREAM_ATTEMPTS:
                break
            pid = row["provider"]
            if pid in tried:
                continue
            tried.add(pid)
            provider = self.directory[pid]
            try:
                result = self.upstream(provider, row["model"], client_api, body)
            except ConnectionError as err:
                result = None
                last_error = str(err)
            if result is None:
                with self.lock:
                    self.calls[pid] = self.calls.get(pid, 0) + 1
                    self.failures[pid] = self.failures.get(pid, 0) + 1
                log("request %d: %s/%s unreachable (%s); failing over"
                    % (index + 1, pid, row["model"], last_error))
                continue
            status, parsed = result
            if status < 200 or status >= 300:
                with self.lock:
                    if status in (402, 429) or status >= 500:
                        self.calls[pid] = self.calls.get(pid, 0) + 1
                    self.failures[pid] = self.failures.get(pid, 0) + 1
                log("request %d: %s/%s refused HTTP %s; failing over"
                    % (index + 1, pid, row["model"], status))
                continue
            with self.lock:
                self.calls[pid] = self.calls.get(pid, 0) + 1
            text, in_tokens, out_tokens, family = parsed
            log("request %d: served by %s/%s (rating %d)"
                % (index + 1, pid, row["model"], row["rating"]))
            return emit_client_response(client_api, body.get("stream") is True,
                                         row["model"], text, in_tokens, out_tokens, family)
        raise RuntimeError("every provider refused the request: %s" % (last_error or "no candidate left"))


# ------------------------------------------------------------------ state

class State:
    def __init__(self):
        self.env_url = os.environ.get("MAKEFUN_ENV_URL", "")
        self.prompt_path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
        self.router = None
        self.catalog = {"providers": []}
        self.served_models = []  # (id, provider) for GET /v1/models


STATE = State()


# ------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status, value):
        self._send(status, "application/json", json_bytes(value))

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/v1/models":
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error", "code": None}})
            return
        anthropic = self.headers.get("x-api-key") is not None
        if anthropic:
            data = [{"type": "model", "id": model_id, "display_name": model_id,
                     "created_at": FIXED_CREATED_AT, "free": True}
                    for model_id, _pid in STATE.served_models]
            self._send_json(200, {"data": data, "has_more": False})
        else:
            data = [{"id": model_id, "object": "model", "created": FIXED_CREATED,
                     "owned_by": pid, "free": True}
                    for model_id, pid in STATE.served_models]
            self._send_json(200, {"object": "list", "data": data})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error", "code": None}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        if not is_record(body):
            self._send_json(400, {"error": {"message": "the request body is not one JSON object",
                                            "type": "invalid_request_error", "code": None}})
            return
        try:
            content_type, payload = STATE.router.serve(client_api, body)
        except RuntimeError as err:
            log("serving %s failed: %s" % (client_api, err))
            self._send_json(502, {"error": {"message": str(err),
                                            "type": "api_error", "code": None}})
            return
        self._send(200, content_type, payload)


# --------------------------------------------------------------------- main


def register(port):
    body = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": STATE.catalog}
    last_error = None
    for attempt in range(REGISTER_RETRIES):
        try:
            status, parsed, _raw = http_json("POST", STATE.env_url.rstrip("/") + "/register",
                                             body=body, timeout=10.0)
        except ConnectionError as err:
            last_error = str(err)
            time.sleep(0.5)
            continue
        if status == 200 and is_record(parsed) and parsed.get("ok") is True:
            log("registered on port %d; %s requests to serve" % (port, parsed.get("requests")))
            return True
        last_error = "HTTP %s" % status
        time.sleep(0.5)
    log("register failed: %s" % last_error)
    return False


def poll_work_until_done():
    deadline = time.time() + WORK_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            status, parsed, _raw = http_json("GET", STATE.env_url.rstrip("/") + "/work", timeout=10.0)
        except ConnectionError:
            time.sleep(WORK_POLL_SECONDS)
            continue
        if status == 200 and is_record(parsed):
            if parsed.get("done") is True:
                log("the client is done; exiting 0")
                return True
        time.sleep(WORK_POLL_SECONDS)
    log("work polling timed out after %ds" % WORK_TIMEOUT_SECONDS)
    return False


def load_world():
    """Load the brief and the env instance; return (directory, quality, manifest)."""
    brief_providers, quality, brief_manifest = load_brief(STATE.prompt_path)
    directory = brief_providers.get("providers") if is_record(brief_providers) else None
    manifest = brief_manifest
    if STATE.env_url:
        try:
            instance_providers, instance_manifest = fetch_instance(STATE.env_url)
            directory = instance_providers
            if instance_manifest is not None:
                manifest = instance_manifest
        except ConnectionError as err:
            log("GET /instance failed (%s); using the brief's directory" % err)
    if not isinstance(directory, list) or not directory:
        raise RuntimeError("no provider directory: set MAKEFUN_PROMPT_FILE or MAKEFUN_ENV_URL")
    return directory, quality, manifest


def main(argv):
    list_only = "--list" in argv
    if not STATE.env_url and not STATE.prompt_path:
        sys.stdout.write(json.dumps({"providers": []}) + "\n")
        return 0
    try:
        directory, quality, manifest = load_world()
    except RuntimeError as err:
        log(str(err))
        if list_only:
            sys.stdout.write(json.dumps({"providers": []}) + "\n")
            return 0
        return 1
    catalog, probed = build_catalog(STATE.env_url, directory, quality)
    STATE.catalog = catalog
    STATE.served_models = []
    for provider in catalog.get("providers", []):
        for model in provider.get("models", []):
            if is_record(model) and model.get("free") is True:
                STATE.served_models.append((model["id"], provider["id"]))
    if list_only:
        sys.stdout.write(json.dumps(catalog) + "\n")
        return 0
    STATE.router = Router(directory, quality, probed, manifest)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    log("listening on 0.0.0.0:%d" % port)
    try:
        if not register(port):
            server.shutdown()
            return 1
        done = poll_work_until_done()
    finally:
        server.shutdown()
        server.server_close()
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
