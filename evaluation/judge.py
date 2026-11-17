"""Run the LLM-as-a-judge rubric over an annotation set.

The judges (2026-11-13) are `zai-glm-5-3` and `mistral-large-4`, through the Mistral API (credential
MISTRAL_API_KEY or `~/.mistral_key`), and `muse-spark-1.3`, through Meta's Model
API (credential META_API_KEY or `~/.meta_key`; endpoint META_BASE_URL, default
https://api.meta.ai/v1). `claude-sonnet-5`, the judge of the earlier runs, is
no longer used: its Anthropic credit ran out and its verdicts saturated. It
stays registered so those runs can still be read and resumed. Also available:
`zai-glm-5-3-flash`, `mistral-large-4`, and `muse-spark-1.3-contributor`, the
cheaper Muse tier whose prompts and completions Meta may train on.

The judge is not a generator of the narratives being scored, which matters: a
model grading its own output is the one configuration a judge study cannot use.

The judge sees the narrative alone — no artefact, no glossary — because that is
what the human annotators see. Their agreement is the whole point of the
exercise, and it only measures the judge if both were shown the same thing.

Sonnet 5 rejects sampling parameters outright, so its verdicts carry whatever
variation the model has; running it twice on the same narrative is the way to
find out how much, and `--repeat` does that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from narrative.blocks_report import pick_attempt
from narrative.generator import load_glossary

DIMENSIONS = ("linguistic_realization", "internal_consistency", "overall_satisfaction")

# `reasoning` is what the judge is pinned to, and it is part of its identity
# rather than a knob. Reasoning was measured and it did not help: it costs
# thinking tokens and latency without buying sensitivity, and on the one probe
# with an unambiguous answer it made the verdict worse.
#
# The Mistral-API judges cannot switch reasoning off: GLM 5.3 and Mistral Large 4
# accept only low, high or max (measured, 2026-11-06). They are pinned to the
# floor, `low`, and every verdict records the reasoning tokens it spent, so the
# asymmetry with Sonnet is quantified rather than hidden. None of them shares a
# family with a generator; Mistral's own DeepSeek is left out for that reason.
JUDGES = {
    "claude-sonnet-5": {"backend": "anthropic", "model": "claude-sonnet-5", "reasoning": "off"},
    "zai-glm-5-3": {"backend": "mistral", "model": "zai-glm-5-3", "reasoning": "low",
                    "reasoning_effort": "low", "min_max_tokens": 8000},
    "zai-glm-5-3-flash": {"backend": "mistral", "model": "zai-glm-5-3-flash", "reasoning": "low",
                          "reasoning_effort": "low", "min_max_tokens": 8000},
    "mistral-large-4": {"backend": "mistral", "model": "mistral-large-4", "reasoning": "low",
                        "reasoning_effort": "low", "min_max_tokens": 8000},
    # Meta's Model API, OpenAI-style. Reasoning cannot be switched off either;
    # `minimal` is its floor (minimal/low/medium/high/xhigh). Not a generator's
    # family. The contributor tier is Meta's to train on: not the default.
    "muse-spark-1.3": {"backend": "meta", "model": "muse-spark-1.3", "reasoning": "minimal",
                       "reasoning_effort": "minimal", "min_max_tokens": 8000},
    "muse-spark-1.3-contributor": {"backend": "meta", "model": "muse-spark-1.3-contributor",
                                   "reasoning": "minimal", "reasoning_effort": "minimal",
                                   "min_max_tokens": 8000},
}
DEFAULT_JUDGES = ["zai-glm-5-3", "muse-spark-1.3"]


def read_key(env: str, path: str) -> str:
    key = os.environ.get(env, "")
    if not key and Path(path).expanduser().exists():
        key = Path(path).expanduser().read_text().strip()
    if not key:
        raise RuntimeError(f"no credential: set {env} or write it to {path}")
    return key


def parse_verdict(text: str, dimensions=DIMENSIONS) -> dict:
    """Pull the JSON object out of a reply, tolerating a fenced block around it.

    Raises rather than guessing: a verdict that cannot be read is a datum about
    the prompt, and silently coercing it would hide exactly what a pilot run is
    for.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    v = json.loads(m.group(0))
    for d in dimensions:
        if d not in v:
            raise ValueError(f"missing dimension {d!r}")
        if not isinstance(v[d].get("score"), int) or not 0 <= v[d]["score"] <= 3:
            raise ValueError(f"{d}: score {v[d].get('score')!r} is not an integer in 0..3")
    return v


def call_anthropic(system: str, prompt: str, model: str, max_tokens: int = 2000) -> dict:
    import anthropic

    # An identity-linked API key (one issued to a person rather than to an
    # organisation) is refused with a 400 unless the request names the workspace
    # it acts in. Plain organisation keys ignore the header, so sending it when
    # it is available is safe either way.
    headers = {}
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "")
    if not ws and Path("~/.anthropic_workspace").expanduser().exists():
        ws = Path("~/.anthropic_workspace").expanduser().read_text().strip()
    if ws:
        headers["anthropic-workspace-id"] = ws

    client = anthropic.Anthropic(
        api_key=read_key("ANTHROPIC_API_KEY", "~/.anthropic_key"),
        default_headers=headers or None,
    )
    # No temperature: Sonnet 5 rejects sampling parameters outright.
    #
    # Thinking off, and explicitly so: it is on by default here, and left alone
    # the reasoning tokens are drawn from max_tokens and truncated the reply on
    # the one probe that needed the most deliberation, at 2000 tokens with a
    # verdict half written.
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return {"text": text, "usage": {"in": r.usage.input_tokens, "out": r.usage.output_tokens},
            "stop": r.stop_reason, "thinking": "disabled"}


def call_mistral(system: str, prompt: str, model: str, max_tokens: int = 2000,
                 effort: Optional[str] = None,
                 url: str = "https://api.mistral.ai/v1/chat/completions",
                 key_env: str = "MISTRAL_API_KEY", key_path: str = "~/.mistral_key",
                 provider: str = "mistral") -> dict:
    """One call to an OpenAI-style chat-completions API: Mistral's (which also
    serves the Z.ai GLMs) by default, Meta's through `call_meta`.

    Over urllib rather than an SDK, as llm_client does for the OpenAI-style
    generators. Greedy decoding: temperature 0, and top_p 1 with it, since the
    API refuses greedy sampling with any other top_p.

    Reasoning models answer with a list of parts, `thinking` and `text`. Only the
    text is the verdict; the thinking is never parsed, and is recorded by size.
    """
    import urllib.error
    import urllib.request

    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {read_key(key_env, key_path)}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{provider} {model}: HTTP {e.code}: {e.read().decode()[:300]}") from e

    choice = d["choices"][0]
    content = choice["message"].get("content") or ""
    thinking_chars = 0
    if isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        thinking_chars = sum(len(json.dumps(p.get("thinking", ""))) for p in content
                             if p.get("type") == "thinking")
    else:
        text = content
    u = d.get("usage", {}) or {}
    usage = {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if rt is not None:
        usage["reasoning"] = rt
    if thinking_chars:
        usage["thinking_chars"] = thinking_chars
    return {"text": text, "usage": usage, "stop": choice.get("finish_reason"),
            "thinking": f"effort={effort}" if effort else "default"}


def call_meta(system: str, prompt: str, model: str, max_tokens: int = 2000,
              effort: Optional[str] = None) -> dict:
    """Meta's Model API (Muse), OpenAI-style. The base URL is taken from the
    secondary sources available on 2026-11-13; override it with META_BASE_URL
    if Meta's own docs say otherwise."""
    base = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
    return call_mistral(system, prompt, model, max_tokens, effort,
                        url=f"{base}/chat/completions", key_env="META_API_KEY",
                        key_path="~/.meta_key", provider="meta")


def judge_one(narrative: str, judge: str, prompt: dict, max_tokens: int) -> dict:
    spec = JUDGES[judge]
    task = prompt["task"].replace("{narrative}", narrative)
    # Reasoning tokens come out of the same budget as the verdict, so a judge
    # that cannot switch reasoning off is given a floor high enough that the
    # JSON is never cut short.
    max_tokens = max(max_tokens, spec.get("min_max_tokens", 0))
    t0 = time.time()
    if spec["backend"] == "anthropic":
        raw = call_anthropic(prompt["system"].strip(), task, spec["model"], max_tokens)
    elif spec["backend"] == "mistral":
        raw = call_mistral(prompt["system"].strip(), task, spec["model"], max_tokens,
                           spec.get("reasoning_effort"))
    elif spec["backend"] == "meta":
        raw = call_meta(prompt["system"].strip(), task, spec["model"], max_tokens,
                        spec.get("reasoning_effort"))
    else:
        raise ValueError(f"unknown judge backend {spec['backend']!r}")
    raw["seconds"] = round(time.time() - t0, 1)
    return raw


def load_from_runs(runs: Path, block: str, models: List[str], accepted_only: bool,
                   glossary: Optional[dict] = None, which: str = "final",
                   banned_terms: tuple = ()) -> Dict[str, dict]:
    """Narratives straight out of a sweep, keyed `<model-short>-<jet_id>`.

    `which` picks the attempt read from each jet. `final` is the delivered
    narrative, the last attempt of the retry loop, which is the one that passed
    when any did. `first` is what the model wrote unaided, before any feedback;
    it is returned whether or not it passed, so that the same jets can be
    judged before and after refinement and the two compared pairwise. The keys
    are the same under both, which is what pairs them.

    Scoring what the pipeline would actually deliver means taking only the
    narratives that passed verification; `accepted_only=False` also returns the
    rejected ones, which is how one asks whether passing the deterministic
    checks has anything to do with reading well.

    Acceptance is decided by re-verifying the text, never by reading the
    `status` the run recorded. That field was written by whatever verifier was
    on disk during the sweep, and it has already drifted once: a narrative
    writing `tau2/1` for the observable `tau21` was charged a bare number until
    the parser learnt to match names on their spelling, so the file still calls
    it failed while the current checker passes it. Reading `status` would leave
    the judged set one narrative short of the reported pass rate, which is
    exactly the sort of gap that is noticed in a table and not in the code.
    """
    from narrative.verifier import verify

    out: Dict[str, dict] = {}
    for m in models:
        f = runs / f"{block}_{m}.jsonl"
        if not f.exists():
            print(f"  warning: {f.name} not found")
            continue
        short = m.replace("gemini-3.5-", "g").replace("deepseek-v4-", "d").replace("gpt-oss-", "o")
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "abstained" or not r["attempts"]:
                continue
            # final: under the retry loop the last attempt, which is the one that
            # passed when any did; with one attempt it is the only one, so the
            # published v16 selection is unchanged. first: attempt 0.
            att = pick_attempt(r, which)
            if att is None:
                continue
            passes = verify(att["completion"]["text"], r["artefact"], glossary=glossary,
                            banned_terms=banned_terms).ok
            if accepted_only and which == "final" and not passes:
                continue
            text = re.sub(r"[ \t]{2,}", " ",
                          re.sub(r"\[/?[A-Za-z][A-Za-z0-9_]*\]", "", att["completion"]["text"])).strip()
            out[f"{short}-{r['jet_id']}"] = {
                "text": text, "model": m, "jet_id": r["jet_id"],
                "status": "ok" if passes else "failed_verification",
                "status_recorded": r["status"], "regime": r["regime"],
                "view": which, "attempt": att["attempt"], "n_attempts": len(r["attempts"]),
                "delivered": r["status"] == "ok",
                "decision": r["artefact"]["decision"],
            }
    return out


def load_items(d: Path) -> Dict[str, str]:
    """id -> the explanation text, taken from the item files as handed out."""
    out = {}
    for f in sorted(d.glob("*.txt")):
        body = f.read_text()
        # Items may or may not carry the record header; take what follows the
        # last separator either way, so the same loader serves both layouts.
        out[f.stem] = body.split("---\n\n")[-1].strip() if "---" in body else body.strip()
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", type=Path, default=repo / "evaluation/annotation_set")
    p.add_argument("--items", default="items", help="Subdirectory of --set to score")
    # GLM 5.3 and Muse Spark 1.3 by default (2026-11-16); Sonnet and Mistral only to read or
    # resume the earlier runs.
    p.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES, choices=list(JUDGES))
    p.add_argument("--only", nargs="+", default=None, help="Score just these ids, e.g. N01 N02")
    p.add_argument("--repeat", type=int, default=1, help="Score each item this many times")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--prompt", type=Path, default=repo / "evaluation/judge_prompt.yaml")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--from-runs", metavar="BLOCK", default=None,
                   help="Score a sweep block directly, e.g. v16, instead of an item directory")
    p.add_argument("--models", nargs="+", default=None,
                   help="With --from-runs, the run names to score (default: `generators` "
                        "in the narrative config)")
    p.add_argument("--attempt", choices=["first", "final"], default="final",
                   help="With --from-runs: final, the delivered narratives (default); first, "
                        "every jet's unaided first attempt, passed or not, for a paired "
                        "before/after-refinement comparison")
    p.add_argument("--include-rejected", action="store_true",
                   help="With --from-runs, also score narratives that failed verification")
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml",
                   help="With --from-runs, the narrative config of the use case "
                        "(narrative_anomaly.yaml for anomaly detection)")
    p.add_argument("--jets", type=int, nargs="+", default=None,
                   help="With --from-runs, score only these jet ids (every model's narrative of each)")
    args = p.parse_args()

    prompt = yaml.safe_load(args.prompt.read_text())
    # judge-v2 has no `dimensions` field: its three are the module default.
    dims = tuple(prompt.get("dimensions") or DIMENSIONS)
    meta: Dict[str, dict] = {}
    if args.from_runs:
        repo_cfg = yaml.safe_load(args.config.read_text())
        runs = repo / repo_cfg["outputs"]["narratives"]
        gens = args.models or [g["name"] for g in repo_cfg["generators"]]
        glossary = load_glossary(repo / repo_cfg["glossary"]["path"], require_reviewed=True)
        banned = tuple((repo_cfg.get("verifier") or {}).get("banned_terms") or ())
        meta = load_from_runs(runs, args.from_runs, gens, not args.include_rejected, glossary,
                              which=args.attempt, banned_terms=banned)
        if args.jets:
            keep = set(args.jets)
            meta = {k: v for k, v in meta.items() if v["jet_id"] in keep}
        items = {k: v["text"] for k, v in meta.items()}
        if args.attempt == "first":
            kind = "first attempts, passed or not"
        else:
            kind = "all narrated" if args.include_rejected else "delivered only"
        print(f"block {args.from_runs}: {len(items)} narratives ({kind})")
    else:
        items = load_items(args.set / args.items)
    ids = args.only or sorted(items)
    missing = [i for i in ids if i not in items]
    if missing:
        sys.exit(f"unknown ids: {' '.join(missing)}")

    tag = f"_{args.from_runs}_{args.attempt}" if args.from_runs else ""
    out_path = args.out or (repo / f"evaluation/results/judgments{tag}_{prompt['id']}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Everything already scored, so an interrupted run resumes instead of paying
    # for its own history again. Only verdicts count as done: a record carrying
    # an error is a call that never landed, and re-running is how it is retried.
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # a line torn in half by a kill
                continue
            # A verdict from another prompt version was given by a different
            # instrument, so it does not count as done for this one.
            if "verdict" in r and r.get("prompt_id") == prompt["id"]:
                done.add((r["id"], r["judge"], r.get("rep", 0)))
    todo = sum(1 for n in ids for j in args.judges for k in range(args.repeat)
               if (n, j, k) not in done)
    print(f"{prompt['id']} | {len(ids)} items x {len(args.judges)} judges x {args.repeat}"
          f" = {len(ids) * len(args.judges) * args.repeat}"
          + (f", {len(done)} already done, {todo} to do" if done else ""))
    n_bad = 0
    with open(out_path, "a") as fh:
        for nid in ids:
            for judge in args.judges:
                for rep in range(args.repeat):
                    if (nid, judge, rep) in done:
                        continue
                    rec = {"id": nid, "judge": judge, "prompt_id": prompt["id"], "rep": rep}
                    if nid in meta:
                        rec.update({k: v for k, v in meta[nid].items() if k != "text"})
                    try:
                        raw = judge_one(items[nid], judge, prompt, args.max_tokens)
                        rec.update(raw)
                        rec["verdict"] = parse_verdict(raw["text"], dims)
                        scores = " ".join(f"{d[:4]}={rec['verdict'][d]['score']}" for d in dims)
                        print(f"  {nid} {judge:<16} rep{rep}  {scores}  ({raw['seconds']}s)")
                    except Exception as exc:
                        n_bad += 1
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"  {nid} {judge:<16} rep{rep}  FAILED: {rec['error'][:110]}")
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
    print(f"\nwrote -> {out_path}" + (f"   ({n_bad} failed)" if n_bad else ""))


if __name__ == "__main__":
    main()
