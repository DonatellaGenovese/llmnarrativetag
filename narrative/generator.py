"""Prompt assembly and narrative generation.

The prompt is built from three versioned pieces — the template, the glossary and
the artefact — and the hash of the rendered text is recorded with the narrative,
so a Part 4 number can always be traced to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from .llm_client import Completion, LLMClient


class GlossaryError(RuntimeError):
    """Raised when the glossary is not fit to be put in front of a reader."""


def load_glossary(path: Path, require_reviewed: bool = True) -> Dict[str, dict]:
    doc = yaml.safe_load(path.read_text())
    entries = doc["observables"]

    missing = [k for k, v in entries.items() if not (v.get("meaning") or "").strip()]
    if missing:
        raise GlossaryError(f"glossary entries have no `meaning`: {', '.join(sorted(missing))}")

    if require_reviewed:
        draft = [k for k, v in entries.items() if v.get("status") != "reviewed"]
        if draft:
            raise GlossaryError(
                "these glossary entries are still `status: draft`, and their text goes "
                "verbatim into the narrative:\n  "
                + ", ".join(sorted(draft))
                + f"\n\nReview them in {path}, set `status: reviewed`, or set "
                "`glossary.require_reviewed: false` in the config to proceed anyway."
            )
    return entries


def render_glossary(entries: Dict[str, dict], names: Optional[list] = None) -> str:
    """The glossary block, restricted to the observables the jet reports."""
    lines = []
    for name in names or list(entries):
        e = entries[name]
        units = f", {e['units']}" if e.get("units") else ""
        meaning = " ".join((e.get("meaning") or "").split())
        definition = " ".join((e.get("definition") or "").split())
        lines.append(f"- `{name}` ({e.get('label', name)}{units}): {definition} {meaning}".rstrip())
    return "\n".join(lines)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class NarrativeGenerator:
    """Renders prompts for artefacts and asks the model to narrate them."""

    def __init__(self, prompt_path: Path, glossary: Dict[str, dict], client: LLMClient):
        doc = yaml.safe_load(prompt_path.read_text())
        self.prompt_id = doc["id"]
        self.system = doc["system"]
        self.template = doc["task"]
        self.caveat = doc.get("caveat", "")
        self.remainder = doc.get("remainder", "")
        self.glossary = glossary
        self.client = client

    def render(self, artefact: dict, caveat: bool = False) -> str:
        names = [f["name"] for f in artefact["features"]]
        artefact_json = json.dumps(artefact, indent=2)
        # replace(), not format(): the template is free to contain braces.
        text = self.template.replace("{glossary}", render_glossary(self.glossary, names))
        text = text.replace("{artefact_json}", artefact_json)
        if "n_other" in artefact and self.remainder:
            text = text + self.remainder
        if caveat and self.caveat:
            text = text + self.caveat
        return text

    def generate(
        self,
        artefact: dict,
        caveat: bool = False,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[Completion, str]:
        prompt = self.render(artefact, caveat=caveat)
        completion = self.client.generate(
            self.system, prompt, seed=seed, temperature=temperature
        )
        return completion, prompt

    def prompt_hash(self, artefact: dict, caveat: bool = False) -> str:
        return _sha(self.system + self.render(artefact, caveat=caveat))
