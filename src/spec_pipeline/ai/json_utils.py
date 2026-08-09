from __future__ import annotations

import json


def extract_json_payload(text: str) -> dict[str, object]:
    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    fenced = _extract_fenced_json(stripped)
    if fenced is not None:
        return fenced

    candidate = _extract_first_json_object(stripped)
    if candidate is not None:
        return candidate
    raise ValueError("No JSON object found in AI response")


def _extract_fenced_json(text: str) -> dict[str, object] | None:
    marker = "```"
    start = text.find(marker)
    while start != -1:
        content_start = text.find("\n", start + len(marker))
        if content_start == -1:
            return None
        end = text.find(marker, content_start + 1)
        if end == -1:
            return None
        block = text[content_start + 1 : end].strip()
        try:
            loaded = json.loads(block)
        except json.JSONDecodeError:
            start = text.find(marker, end + len(marker))
            continue
        return loaded if isinstance(loaded, dict) else None
    return None


def _extract_first_json_object(text: str) -> dict[str, object] | None:
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        loaded = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return loaded if isinstance(loaded, dict) else None
        search_from = start + 1
    return None
