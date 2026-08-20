"""Thin wrapper over google-genai, for Google AI Studio and Vertex AI.

Both backends are the same SDK and the same models; they differ only in how the
client authenticates:

    aistudio  genai.Client(api_key=...)                    GOOGLE_API_KEY
    vertex    genai.Client(vertexai=True, project=...,      GOOGLE_CLOUD_PROJECT
                           location=...)                    GOOGLE_CLOUD_LOCATION
                                                            (+ application default
                                                             credentials)

`seed` is passed through, but the SDK is explicit that it buys a "best effort to
provide the same response for repeated requests" and "not a guaranteed absolute
deterministic behavior". Part 4 therefore records every completion rather than
assuming it can be regenerated: see `Completion`, which carries everything
needed to attribute a narrative to the call that produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

BACKEND_AISTUDIO = "aistudio"
BACKEND_VERTEX = "vertex"

# Providers that speak the OpenAI chat-completions protocol. One code path
# serves all of them; only the host, the key and the way reasoning is switched
# off differ. Called over urllib rather than the openai SDK on purpose: the
# conda environment here also carries the fitted surrogate stack, and is not
# worth perturbing for thirty lines of HTTP.
OPENAI_BACKENDS = {
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama": ("https://ollama.com/v1", "OLLAMA_API_KEY"),
}


@dataclass
class Completion:
    """One model response, with enough provenance to audit it later."""

    text: str
    model: str
    backend: str
    seed: Optional[int]
    temperature: float
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    # What was actually applied, which is not always what was asked: models that
    # reject the field fall back to their default and say so here.
    thinking_budget: Optional[int] = None
    thinking_control: str = "default"  # "default" | "applied" | "unsupported"

    def to_dict(self) -> dict:
        return asdict(self)


class LLMClient:
    """Generates narratives from a system instruction and a rendered prompt."""

    def __init__(
        self,
        backend: str = BACKEND_AISTUDIO,
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_output_tokens: int = 8192,
        project: Optional[str] = None,
        location: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        if backend not in (BACKEND_AISTUDIO, BACKEND_VERTEX) and backend not in OPENAI_BACKENDS:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.project = project
        self.location = location
        self.thinking_budget = thinking_budget
        # Overrides the effort derived from thinking_budget. Needed because some
        # models cannot switch reasoning off at all: gpt-oss has only low, medium
        # and high internally, and answers with a `reasoning` field even when
        # sent `none`, so the honest setting for it is the floor rather than a
        # request it cannot honour.
        self.reasoning_effort = reasoning_effort
        # Set once a model has refused thinking_config, so the whole run does not
        # pay for the same rejected request on every jet.
        self._thinking_unsupported = False
        self._client = None  # built lazily so importing needs no credentials

    # -- client ------------------------------------------------------------

    def _build_client(self):
        from google import genai

        if self.backend == BACKEND_AISTUDIO:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. Create a key at "
                    "https://aistudio.google.com/apikey, or switch "
                    "llm.backend to 'vertex' in the config."
                )
            return genai.Client(api_key=api_key)

        # Environment overrides the config, so the same checkout can point at a
        # different project without an edit.
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or self.project
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or self.location or "us-central1"
        if not project:
            raise RuntimeError(
                "No Vertex AI project: set llm.project in the config or export "
                "GOOGLE_CLOUD_PROJECT."
            )
        try:
            return genai.Client(vertexai=True, project=project, location=location)
        except Exception as exc:
            raise RuntimeError(
                f"could not open a Vertex AI client for project {project!r} in "
                f"{location!r}: {exc}\n\nApplication default credentials are needed. "
                "Either run `gcloud auth application-default login`, or point "
                "GOOGLE_APPLICATION_CREDENTIALS at a service account key file."
            ) from exc

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # -- generation --------------------------------------------------------

    def _generate_openai(
        self, system: str, prompt: str, seed: Optional[int], temperature: float
    ) -> Completion:
        """One call against an OpenAI-compatible chat-completions endpoint."""
        import json
        import urllib.request

        base, env_var = OPENAI_BACKENDS[self.backend]
        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(f"{env_var} is not set, required by backend {self.backend!r}")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed

        # `reasoning_effort: "none"` switches reasoning off on Gemini and
        # DeepSeek. Note that omitting it is NOT equivalent: DeepSeek's docs read
        # as though non-thinking were the default, but a request without the
        # field comes back with reasoning tokens spent. Measured, not assumed.
        effort = self.reasoning_effort or ("none" if self.thinking_budget == 0 else None)
        control = "default"
        if effort is not None:
            body["reasoning_effort"] = effort
            control = f"effort={effort}"

        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.load(resp)

        choice = payload["choices"][0]
        raw = payload.get("usage", {}) or {}
        details = raw.get("completion_tokens_details", {}) or {}
        # Normalised to the Gemini field names so metrics.py needs no branch.
        usage = {
            "prompt_token_count": raw.get("prompt_tokens"),
            "candidates_token_count": raw.get("completion_tokens"),
            "total_token_count": raw.get("total_tokens"),
        }
        if details.get("reasoning_tokens") is not None:
            usage["thoughts_token_count"] = details["reasoning_tokens"]

        # Some providers return the chain of thought in a separate field and
        # count it inside `completion_tokens`. Recorded by length so the
        # asymmetry is quantified rather than merely declared; never appended to
        # the narrative.
        reasoning = choice["message"].get("reasoning") or ""
        if reasoning:
            usage["reasoning_chars"] = len(reasoning)

        return Completion(
            text=choice["message"].get("content") or "",
            model=self.model,
            backend=self.backend,
            seed=seed,
            temperature=temperature,
            finish_reason=choice.get("finish_reason"),
            usage={k: v for k, v in usage.items() if v is not None},
            thinking_budget=self.thinking_budget if control == "applied" else None,
            thinking_control=control,
        )

    def generate(
        self,
        system: str,
        prompt: str,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        if self.backend in OPENAI_BACKENDS:
            return self._generate_openai(
                system,
                prompt,
                self.seed if seed is None else seed,
                self.temperature if temperature is None else temperature,
            )

        from google.genai import types

        used_seed = self.seed if seed is None else seed
        used_temp = self.temperature if temperature is None else temperature

        def build_config(thinking: Optional[int]):
            return types.GenerateContentConfig(
                system_instruction=system,
                temperature=used_temp,
                max_output_tokens=self.max_output_tokens,
                seed=used_seed,
                thinking_config=(
                    None if thinking is None else types.ThinkingConfig(thinking_budget=thinking)
                ),
            )

        want = None if self._thinking_unsupported else self.thinking_budget
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=build_config(want)
            )
            control = "applied" if want is not None else "default"
            applied = want
        except Exception as exc:
            # gemini-2.5-pro refuses thinking_config with a 400. Retry without it
            # rather than abort: the run continues, and the record shows that the
            # request was not honoured.
            if want is None or "thinking" not in str(exc).lower():
                raise
            self._thinking_unsupported = True
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=build_config(None)
            )
            control, applied = "unsupported", None

        finish_reason = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = getattr(reason, "name", None) or (str(reason) if reason else None)

        usage: Dict[str, Any] = {}
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            for k in (
                "prompt_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                v = getattr(meta, k, None)
                if v is not None:
                    usage[k] = v

        return Completion(
            text=response.text or "",
            model=self.model,
            backend=self.backend,
            seed=used_seed,
            temperature=used_temp,
            finish_reason=finish_reason,
            usage=usage,
            thinking_budget=applied,
            thinking_control=control,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMClient":
        llm = cfg["llm"]
        return cls(
            backend=llm["backend"],
            model=llm["model"],
            temperature=float(llm["temperature"]),
            seed=llm.get("seed"),
            max_output_tokens=int(llm["max_output_tokens"]),
            project=llm.get("project"),
            location=llm.get("location"),
            thinking_budget=llm.get("thinking_budget"),
            reasoning_effort=llm.get("reasoning_effort"),
        )
