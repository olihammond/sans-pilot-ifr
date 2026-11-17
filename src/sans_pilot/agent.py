"""``sans-pilot-agent``: run the fixed analysis procedure with a local LLM.

The runner owns the procedure; the language model only makes the decisions
inside it.  For one data file it

1. runs the guided analysis (load, Guess, fit the top suggestions, rank)
   deterministically through the MCP server's guided tools;
2. optionally hands the result to a model served on any OpenAI-compatible
   endpoint (Ollama, LM Studio, llama.cpp, vLLM, ...), which may switch
   candidate, refine the fit with the small session tools, and write an
   interpretation, within a fixed step budget;
3. always writes ``report.md``, ``analysis.json`` and the fit plot, whether or
   not the model behaved.

Without ``--model`` only steps 1 and 3 run and no language model is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastmcp import Client

DEFAULT_BASE_URL = "http://localhost:11434/v1"
MAX_TOOL_RESULT_CHARS = 9_000
# Tools the model may call.  Loading data, running the guided procedure and
# exporting stay with the runner so the outcome never depends on the model.
MODEL_TOOLS = (
  "use-sans-candidate",
  "fit-sans-suggestions",
  "set-sans-model",
  "set-sans-parameter",
  "set-sans-polydispersity",
  "run-sans-fit",
  "get-sans-analysis",
)

SYSTEM_PROMPT = """
You are refining a small-angle neutron scattering (SANS) model fit.
The guided analysis has ALREADY been run; its result is in the user message.
Work only with the provided tools and the session_id you are given.

Do this, in order:
1. Compare the candidates with what is known about the sample. If a different
   fitted candidate is more plausible, switch with use-sans-candidate. If the
   sample suggests a model that was not tried, fit it with
   fit-sans-suggestions and extra_models.
2. If the selected fit's diagnostics list issues, act on ONE suggestion with
   set-sans-parameter or set-sans-polydispersity, then call
   run-sans-fit. Make at most three refinements. Stop early when the verdict is
   good or acceptable and no issue remains that you can act on. Every change
   must be followed by run-sans-fit: unfitted changes are not reported.
3. Reply WITHOUT calling a tool to finish. Write a short interpretation: the
   model chosen and why, each fitted value with its uncertainty and unit
   (lengths are in angstrom), the reduced chi-squared and verdict, and any
   remaining issues or doubts.

Rules:
- Quote numbers exactly as the tools return them. Never invent values.
- A low chi-squared means a model CAN describe the data, not that it is right.
- Prefer the simpler model when candidates are marked close_to_best.
- Keep SLD parameters fixed unless the user gave their values.
- The Q range is the user's choice. You cannot change it, and a model that
  only fits part of the data is not a good model.
- Never fix a parameter at its own fitted value (vary=false without a new
  value). That leaves the fit unchanged and only makes the other uncertainties
  look smaller. Fix a parameter only at an independently known value.
- Strong correlations that cannot be resolved are a limitation to report, not
  something to hide.
""".strip()


@dataclass(slots=True)
class AgentConfig:
  """Everything one runner invocation needs."""

  out: Path
  data: Path | None = None
  series: list[Path] = field(default_factory=list)
  per_file_parameters: str | None = None
  example: str | None = None
  model: str | None = None
  base_url: str = DEFAULT_BASE_URL
  api_key: str = "local"
  sample: str = ""
  top: int = 3
  extra_models: str | None = None
  fixed_parameters: str | None = None
  q_min: float | None = None
  q_max: float | None = None
  max_steps: int = 12
  temperature: float = 0.0
  request_timeout: float = 600.0
  plot: bool = True


@dataclass(slots=True)
class AgentResult:
  """What the runner produced."""

  session_id: str
  status: str
  model: str | None
  verdict: str | None
  reduced_chisq: float | None
  report: Path
  files: list[Path] = field(default_factory=list)
  interpretation: str | None = None
  llm_note: str | None = None
  llm_steps: int = 0


def _tool_text(result: Any) -> str:
  """Reduce an MCP tool result to the text a language model can read."""
  structured = getattr(result, "structured_content", None)
  if structured is not None:
    return json.dumps(structured, ensure_ascii=False, default=str)
  parts = [
    item.text
    for item in getattr(result, "content", []) or []
    if getattr(item, "text", None)
  ]
  return "\n".join(parts)


def _clip(text: str) -> str:
  if len(text) <= MAX_TOOL_RESULT_CHARS:
    return text
  return text[:MAX_TOOL_RESULT_CHARS] + " ... [truncated; call get-sans-analysis]"


async def _call_tool(client: Client, name: str, arguments: dict[str, Any]) -> str:
  """Call one MCP tool; errors come back as text the model can act on."""
  try:
    result = await client.call_tool(name, arguments)
  except Exception as exc:
    return json.dumps({"error": str(exc)[:600]})
  return _tool_text(result)


def openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
  """Convert MCP tool definitions to OpenAI function-calling definitions."""
  converted = []
  for tool in tools:
    if tool.name not in MODEL_TOOLS:
      continue
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", {})
    converted.append(
      {
        "type": "function",
        "function": {
          "name": tool.name,
          "description": tool.description or "",
          "parameters": schema or {"type": "object", "properties": {}},
        },
      }
    )
  return converted


_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
  """Return normalized tool calls, tolerating common local-model formats.

  Handles the OpenAI ``tool_calls`` field, argument objects that were not
  serialized to a string, and models that print a call as JSON text.
  """
  calls: list[dict[str, Any]] = []
  for index, call in enumerate(message.get("tool_calls") or []):
    function = call.get("function") or {}
    calls.append(
      {
        "id": call.get("id") or f"call_{index}",
        "name": str(function.get("name") or ""),
        "arguments": function.get("arguments"),
      }
    )
  if calls:
    return calls

  content = message.get("content")
  if not isinstance(content, str):
    return []
  candidates = _TEXT_TOOL_CALL.findall(content)
  stripped = content.strip()
  if not candidates and stripped.startswith("{") and stripped.endswith("}"):
    candidates = [stripped]
  for index, text in enumerate(candidates):
    try:
      payload = json.loads(text)
    except json.JSONDecodeError:
      continue
    name = payload.get("name")
    arguments = payload.get("arguments", payload.get("parameters"))
    if isinstance(name, str) and name in MODEL_TOOLS and arguments is not None:
      calls.append({"id": f"text_call_{index}", "name": name, "arguments": arguments})
  return calls


def decode_arguments(arguments: Any) -> dict[str, Any] | str:
  """Decode tool arguments, returning an error string when they are unusable."""
  if arguments is None or arguments == "":
    return {}
  if isinstance(arguments, dict):
    return arguments
  if isinstance(arguments, str):
    try:
      decoded = json.loads(arguments)
    except json.JSONDecodeError as exc:
      return f"Tool arguments were not valid JSON ({exc.msg}). Send a JSON object."
    if isinstance(decoded, dict):
      return decoded
  return "Tool arguments must be a JSON object."


async def _chat(
  http: httpx.AsyncClient,
  config: AgentConfig,
  messages: list[dict[str, Any]],
  tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "model": config.model,
    "messages": messages,
    "temperature": config.temperature,
    "stream": False,
  }
  if tools:
    payload["tools"] = tools
  response = await http.post("/chat/completions", json=payload)
  response.raise_for_status()
  body = response.json()
  try:
    return dict(body["choices"][0]["message"])
  except (KeyError, IndexError, TypeError) as exc:
    raise ValueError(f"Unexpected chat response: {str(body)[:300]}") from exc


async def refine_with_model(
  client: Client,
  config: AgentConfig,
  guided: dict[str, Any],
  *,
  transport: httpx.AsyncBaseTransport | None,
  log: Callable[[str], None],
) -> tuple[str | None, str | None, int, list[dict[str, Any]]]:
  """Let the model refine the analysis within the step budget.

  Returns (interpretation, note, model turns used, transcript).
  """
  tools = openai_tools(await client.list_tools())
  brief = {
    key: guided.get(key)
    for key in (
      "session_id",
      "data",
      "features",
      "candidates",
      "ranking",
      "fixed_parameters",
      "selected_model",
      "selected_fit",
      "caution",
    )
  }
  messages: list[dict[str, Any]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
      "role": "user",
      "content": (
        "What is known about the sample: "
        + (config.sample.strip() or "nothing was provided.")
        + "\n\nGuided analysis result:\n"
        + json.dumps(brief, ensure_ascii=False, default=str)
      ),
    },
  ]
  note: str | None = None
  steps = 0
  reminded: set[str] = set()
  async with httpx.AsyncClient(
    base_url=config.base_url.rstrip("/"),
    headers={"Authorization": f"Bearer {config.api_key}"},
    timeout=config.request_timeout,
    transport=transport,
  ) as http:
    use_tools: list[dict[str, Any]] | None = tools
    while steps < config.max_steps:
      steps += 1
      final_turn = steps == config.max_steps
      if final_turn and use_tools:
        messages.append(
          {
            "role": "user",
            "content": (
              "The step budget is used up. Do not call tools. "
              "Write the final interpretation now."
            ),
          }
        )
      try:
        message = await _chat(http, config, messages, None if final_turn else use_tools)
      except httpx.HTTPStatusError as exc:
        if use_tools and exc.response.status_code in (400, 404, 422, 501):
          # The endpoint or model rejected tool definitions: ask for an
          # interpretation of the deterministic result only.
          note = (
            f"The model endpoint rejected tool calling "
            f"(HTTP {exc.response.status_code}); it only interpreted the "
            "guided result and made no refinements."
          )
          log(note)
          use_tools = None
          steps -= 1
          continue
        return None, f"Model request failed: {exc}", steps, messages
      except (httpx.HTTPError, ValueError) as exc:
        return None, f"Model request failed: {exc}", steps, messages

      calls = [] if final_turn or not use_tools else parse_tool_calls(message)
      assistant: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
      }
      if message.get("tool_calls"):
        assistant["tool_calls"] = message["tool_calls"]
      messages.append(assistant)
      if not calls:
        text = (message.get("content") or "").strip()
        reminder = None
        if not final_turn and "unfitted" not in reminded:
          status = json.loads(
            await _call_tool(
              client, "get-sans-analysis", {"session_id": guided["session_id"]}
            )
          )
          if status.get("unfitted_changes"):
            reminded.add("unfitted")
            reminder = (
              "You changed settings without refitting: "
              + "; ".join(status["unfitted_changes"])
              + ". Call run-sans-fit now, or these changes will be discarded "
              "and the previous fit reported."
            )
        if reminder is None and not text and "empty" not in reminded:
          reminded.add("empty")
          reminder = "Your reply was empty. Write the final interpretation now."
        if reminder is None or final_turn:
          return text or None, note, steps, messages
        messages.append({"role": "user", "content": reminder})
        continue

      native = bool(message.get("tool_calls"))
      for call in calls:
        arguments = decode_arguments(call["arguments"])
        if call["name"] not in MODEL_TOOLS:
          output = json.dumps(
            {
              "error": f"Unknown tool '{call['name']}'. "
              "Available: " + ", ".join(MODEL_TOOLS)
            }
          )
        elif isinstance(arguments, str):
          output = json.dumps({"error": arguments})
        else:
          arguments.setdefault("session_id", guided["session_id"])
          log(f"  model -> {call['name']} {json.dumps(arguments, default=str)[:160]}")
          output = await _call_tool(client, call["name"], arguments)
        output = _clip(output)
        if native:
          messages.append(
            {"role": "tool", "tool_call_id": call["id"], "content": output}
          )
        else:
          messages.append(
            {"role": "user", "content": f"Result of {call['name']}:\n{output}"}
          )
  return None, "The model did not finish within the step budget.", steps, messages


def _prepare_workspace(config: AgentConfig) -> str | None:
  """Create an isolated upload/runs area under the output folder."""
  work = config.out / ".sans-pilot"
  uploads, runs = work / "uploads", work / "runs"
  uploads.mkdir(parents=True, exist_ok=True)
  runs.mkdir(parents=True, exist_ok=True)
  os.environ["UPLOAD_DIR"] = str(uploads.resolve())
  os.environ["SANS_PILOT_RUNS_DIR"] = str(runs.resolve())
  os.environ["SANS_PILOT_ARTIFACT_TTL_SECONDS"] = "0"
  os.environ.pop("API_TOKEN", None)
  if config.data is None:
    return None
  if not config.data.is_file():
    raise FileNotFoundError(f"Data file not found: {config.data}")
  target = uploads / config.data.name
  shutil.copyfile(config.data, target)
  return target.name


def _comparison(
  guided: dict[str, Any], result: AgentResult, last_fit: dict[str, Any]
) -> list[str]:
  """Describe honestly what the language model's turn changed."""
  before = (guided.get("selected_fit") or {}).get("statistics") or {}
  after = last_fit.get("statistics") or {}
  same_model = guided.get("selected_model") == result.model
  chisq_before, chisq_after = before.get("chisq"), after.get("chisq")
  lines: list[str] = []
  if before.get("n_points") != after.get("n_points"):
    lines.append(
      f"The Q range changed during the language model's turn "
      f"({before.get('n_points')} points before, {after.get('n_points')} after), "
      f"so χ² and BIC before and after are not comparable. Before: "
      f"`{guided.get('selected_model')}`, χ²/dof {before.get('reduced_chisq')}; "
      f"after: `{result.model}`, χ²/dof {after.get('reduced_chisq')}."
    )
    lines.append("The pre-refinement state is saved as `analysis_guided.json`.")
    return lines
  unchanged = (
    same_model
    and isinstance(chisq_before, (int, float))
    and isinstance(chisq_after, (int, float))
    and abs(chisq_after - chisq_before) <= 1e-6 * max(abs(chisq_before), 1.0)
  )
  if unchanged:
    lines.append(
      f"The language model's turn did not change the fit: the model is still "
      f"`{result.model}` and χ² is still {chisq_after}."
    )
    if after.get("n_free") != before.get("n_free"):
      lines.append(
        f"It changed the number of free parameters from {before.get('n_free')} "
        f"to {after.get('n_free')}, which alters χ²/dof and the reported "
        "uncertainties without altering the fit."
      )
  else:
    lines.append(
      f"Before the language model's turn the selected model was "
      f"`{guided.get('selected_model')}` (χ² {chisq_before}, χ²/dof "
      f"{before.get('reduced_chisq')}, {before.get('n_free')} free); afterwards "
      f"it is `{result.model}` (χ² {chisq_after}, χ²/dof "
      f"{after.get('reduced_chisq')}, {after.get('n_free')} free)."
    )
    if (
      same_model
      and isinstance(chisq_before, (int, float))
      and isinstance(chisq_after, (int, float))
      and chisq_after > 1.5 * chisq_before
    ):
      lines.append(
        "**The refinement made the fit worse.** Prefer `analysis_guided.json` "
        "unless the change is justified by knowledge of the sample."
      )
  conditional = last_fit.get("conditional_on") or []
  if conditional:
    lines.append(
      "**Uncertainties are conditional:** "
      + ", ".join(conditional)
      + " were fixed at their own fitted values. The standard errors in the "
      "final fit understate the real uncertainty; `analysis_guided.json` "
      "holds the fit with those parameters free."
    )
  lines.append("The pre-refinement state is saved as `analysis_guided.json`.")
  return lines


def _append_report(
  report: Path,
  config: AgentConfig,
  result: AgentResult,
  guided: dict[str, Any],
  last_fit: dict[str, Any],
) -> None:
  """Add the runner's own sections to report.md and report.html."""
  from sans_pilot.report_html import EXTRA_MARKER, markdown_to_html

  produced = [
    "The data loading, model suggestions (Sassafras curve library), candidate "
    "fits (IQuana) and diagnostics were computed deterministically by "
    "sans-pilot."
  ]
  if guided.get("guess_unavailable"):
    produced.append(
      f"Model suggestions were unavailable: {guided['guess_unavailable']}"
    )
  if config.sample.strip():
    produced.append(f"Sample description supplied by the user: {config.sample}")
  if config.model and guided.get("selected_fit"):
    produced += _comparison(guided, result, last_fit)

  disclaimer = (
    "Written by a language model from the results above. It has not been "
    "checked; verify numbers against the tables in this report."
  )
  interpretation = result.interpretation or "_The model produced no interpretation._"

  lines = ["", "## How this report was produced", "", *produced]
  page = ["<h2>How this report was produced</h2>"] + [
    markdown_to_html(item) for item in produced
  ]
  if config.model:
    lines += [
      "",
      f"## Interpretation by `{config.model}`",
      "",
      f"_{disclaimer}_",
      "",
      interpretation,
    ]
    page += [
      f"<h2>Interpretation by {html.escape(config.model)}</h2>",
      f"<div class=note>{html.escape(disclaimer)}</div>",
      markdown_to_html(interpretation),
    ]
    if result.llm_note:
      lines += ["", f"Note: {result.llm_note}"]
      page.append(f"<p class=small>Note: {html.escape(result.llm_note)}</p>")
  with report.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

  html_report = report.with_suffix(".html")
  if html_report.is_file():
    text = html_report.read_text(encoding="utf-8")
    html_report.write_text(
      text.replace(EXTRA_MARKER, "\n".join(page)), encoding="utf-8"
    )


async def run_agent(
  config: AgentConfig,
  *,
  transport: httpx.AsyncBaseTransport | None = None,
  log: Callable[[str], None] = print,
) -> AgentResult:
  """Run the fixed procedure for one dataset and write the results."""
  config.out.mkdir(parents=True, exist_ok=True)
  stored_name = _prepare_workspace(config)

  # Imported after the workspace environment is set.
  from sans_pilot import sessions
  from sans_pilot.main import build_server

  server = build_server("guided")
  async with Client(server) as client:
    log("Running guided analysis (load, suggest models, fit, rank)...")
    arguments: dict[str, Any] = {"top": config.top}
    if stored_name is not None:
      arguments["file"] = stored_name
    else:
      arguments["example"] = config.example
    for key in ("extra_models", "fixed_parameters", "q_min", "q_max"):
      if getattr(config, key) is not None:
        arguments[key] = getattr(config, key)
    try:
      raw = await client.call_tool("run-sans-guided-analysis", arguments)
    except Exception as exc:
      raise ValueError(f"Guided analysis failed: {exc}") from exc
    guided = json.loads(_tool_text(raw))
    session_id = guided["session_id"]
    for candidate in guided.get("candidates") or []:
      log(
        "  {rank}. {model}: {detail}".format(
          rank=candidate.get("rank", "-"),
          model=candidate.get("model"),
          detail=candidate.get("error")
          or f"chi2/dof {candidate.get('reduced_chisq')} ({candidate.get('verdict')})",
        )
      )
    if guided.get("status") != "fitted":
      log(f"Guided analysis status: {guided.get('status')}. {guided.get('reason', '')}")

    interpretation = note = None
    steps = 0
    transcript: list[dict[str, Any]] = []
    if config.model and guided.get("status") == "fitted":
      # Keep the deterministic result: a model's refinements can make a fit worse.
      guided_analysis = sessions.sessions_root(None) / session_id / "analysis.json"
      shutil.copyfile(guided_analysis, config.out / "analysis_guided.json")
      log(f"Refining with {config.model} at {config.base_url} ...")
      interpretation, note, steps, transcript = await refine_with_model(
        client, config, guided, transport=transport, log=log
      )
      if note:
        log(f"  note: {note}")
    elif config.model:
      note = "The model was not consulted because no candidate fit succeeded."

  # Written first so a failure later still leaves the conversation to inspect.
  (config.out / "guided_result.json").write_text(
    json.dumps(guided, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
  )
  if transcript:
    (config.out / "transcript.json").write_text(
      json.dumps(transcript, indent=2, ensure_ascii=False, default=str),
      encoding="utf-8",
    )
  exported = await anyio.to_thread.run_sync(
    lambda: sessions.export_session(session_id, None, config.out, config.plot)
  )
  summary = exported["summary"]
  last_fit = summary.get("last_fit") or {}
  result = AgentResult(
    session_id=session_id,
    status=str(guided.get("status")),
    model=summary.get("model"),
    verdict=(last_fit.get("diagnostics") or {}).get("verdict"),
    reduced_chisq=(last_fit.get("statistics") or {}).get("reduced_chisq"),
    report=config.out / "report.md",
    files=sorted(exported["artifacts"].values()),
    interpretation=interpretation,
    llm_note=note,
    llm_steps=steps,
  )
  _append_report(result.report, config, result, guided, last_fit)
  if summary.get("unfitted_changes"):
    log(
      "  note: the model changed settings without refitting; the report shows "
      "the last completed fit and lists the unfitted changes."
    )
  log(f"Model: {result.model}  verdict: {result.verdict}")
  html_report = result.report.with_suffix(".html")
  if html_report.is_file():
    log(f"Open:   {html_report}  (figures and tables)")
  log(f"Report: {result.report}")
  return result


SERIES_PROMPT = """
You are summarising a joint fit of several small-angle neutron scattering
curves measured on the same particles. The analysis is finished; its result is
in the user message. Do not ask for tools.

Write a short interpretation: the model chosen and why it beat the others,
each shared fitted value with its uncertainty and unit (lengths in angstrom,
SLDs in 1e-6 per square angstrom), the joint reduced chi-squared and verdict,
how evenly the curves are fitted, and any issues or doubts.

Rules:
- Quote numbers exactly as given. Never invent values.
- A low chi-squared means a model CAN describe the data, not that it is right.
- Mention candidates marked close_to_best: the data does not separate them.
""".strip()


async def run_series(
  config: AgentConfig,
  *,
  transport: httpx.AsyncBaseTransport | None = None,
  log: Callable[[str], None] = print,
) -> AgentResult:
  """Run the joint series procedure and write the results."""
  config.out.mkdir(parents=True, exist_ok=True)
  _prepare_workspace(config)
  uploads = Path(os.environ["UPLOAD_DIR"])
  names: list[str] = []
  for path in config.series:
    if not path.is_file():
      raise FileNotFoundError(f"Data file not found: {path}")
    shutil.copyfile(path, uploads / path.name)
    names.append(path.name)
  if len(set(names)) != len(names):
    raise ValueError("Series files must have distinct names.")

  from sans_pilot.report_html import EXTRA_MARKER, markdown_to_html
  from sans_pilot.series import series_analysis
  from sans_pilot.workers import run_cancellable_worker

  log(f"Running joint analysis of {len(names)} curves (suggest, fit jointly, rank)...")
  try:
    outcome = await run_cancellable_worker(
      series_analysis,
      names,
      None,
      config.per_file_parameters,
      config.fixed_parameters,
      config.top,
      config.extra_models,
      config.out,
      config.plot,
      operation_name="series-analysis",
    )
  except Exception as exc:
    raise ValueError(f"Series analysis failed: {exc}") from exc
  summary = outcome["summary"]
  fit = summary["selected_fit"]
  for candidate in summary["candidates"]:
    log(
      "  {rank}. {model}: {detail}".format(
        rank=candidate.get("rank", "-"),
        model=candidate["model"],
        detail=candidate.get("error")
        or f"joint chi2/dof {candidate.get('reduced_chisq')} ({candidate.get('verdict')})",
      )
    )
  for name, row in fit["shared"].items():
    log(f"  {name} = {row['value']} ± {row['stderr']}")

  interpretation = note = None
  steps = 0
  if config.model:
    log(f"Asking {config.model} for an interpretation ...")
    brief = {
      key: summary.get(key)
      for key in (
        "files",
        "per_file_parameters",
        "fixed_parameters",
        "candidates",
        "ranking",
        "selected_model",
        "selected_fit",
        "note",
      )
    }
    messages = [
      {"role": "system", "content": SERIES_PROMPT},
      {
        "role": "user",
        "content": (
          "What is known about the sample: "
          + (config.sample.strip() or "nothing was provided.")
          + "\n\nJoint analysis result:\n"
          + json.dumps(brief, ensure_ascii=False, default=str)
        ),
      },
    ]
    try:
      async with httpx.AsyncClient(
        base_url=config.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=config.request_timeout,
        transport=transport,
      ) as http:
        message = await _chat(http, config, messages, None)
      interpretation = (message.get("content") or "").strip() or None
      steps = 1
    except (httpx.HTTPError, ValueError) as exc:
      note = f"Model request failed: {exc}"
      log(f"  note: {note}")

  report = config.out / "report.md"
  lines = ["", "## How this report was produced", ""]
  produced = (
    "Model suggestions (Sassafras curve library, from the reference curve), the "
    "joint fits (IQuana MultiFitter, several starting points each) and the "
    "diagnostics were computed deterministically by sans-pilot."
  )
  lines.append(produced)
  page = ["<h2>How this report was produced</h2>", f"<p>{html.escape(produced)}</p>"]
  if config.sample.strip():
    lines.append(f"Sample description supplied by the user: {config.sample}")
    page.append(
      f"<p>Sample description supplied by the user: {html.escape(config.sample)}</p>"
    )
  if config.model:
    disclaimer = (
      "Written by a language model from the results above. It has not been "
      "checked; verify numbers against the tables in this report."
    )
    text = interpretation or "_The model produced no interpretation._"
    lines += [
      "",
      f"## Interpretation by `{config.model}`",
      "",
      f"_{disclaimer}_",
      "",
      text,
    ]
    page += [
      f"<h2>Interpretation by {html.escape(config.model)}</h2>",
      f"<div class=note>{html.escape(disclaimer)}</div>",
      markdown_to_html(text),
    ]
    if note:
      lines += ["", f"Note: {note}"]
  with report.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
  html_report = report.with_suffix(".html")
  if html_report.is_file():
    html_report.write_text(
      html_report.read_text(encoding="utf-8").replace(EXTRA_MARKER, "\n".join(page)),
      encoding="utf-8",
    )
    log(f"Open:   {html_report}  (figures and tables)")
  log(f"Model: {summary['selected_model']}  verdict: {fit['verdict']}")
  log(f"Report: {report}")
  return AgentResult(
    session_id="series",
    status="fitted",
    model=summary["selected_model"],
    verdict=fit["verdict"],
    reduced_chisq=fit["statistics"]["reduced_chisq"],
    report=report,
    files=sorted(outcome["artifacts"].values()),
    interpretation=interpretation,
    llm_note=note,
    llm_steps=steps,
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="sans-pilot-agent",
    description=(
      "Analyse one reduced 1D SANS file with a fixed procedure, optionally "
      "refined by a local LLM on an OpenAI-compatible endpoint."
    ),
  )
  parser.add_argument(
    "data",
    nargs="*",
    type=Path,
    help="reduced 1D data file; give several files to fit them jointly as a series",
  )
  parser.add_argument(
    "--per-file",
    action="append",
    default=[],
    metavar="NAME=V1,V2,...",
    help=(
      "series only: a known value that differs between files, one per file in "
      "the order given, e.g. 'sld_solvent=-0.56,2.9,6.36' (repeatable)"
    ),
  )
  parser.add_argument("--example", help="use a bundled IQuana example instead")
  parser.add_argument(
    "--out", type=Path, help="output folder (default: <data>_analysis)"
  )
  parser.add_argument(
    "--model",
    default=os.environ.get("SANS_PILOT_LLM_MODEL"),
    help="model name on the endpoint; omit to run without a language model",
  )
  parser.add_argument(
    "--base-url",
    default=os.environ.get("SANS_PILOT_LLM_BASE_URL", DEFAULT_BASE_URL),
    help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})",
  )
  parser.add_argument(
    "--api-key", default=os.environ.get("SANS_PILOT_LLM_API_KEY", "local")
  )
  parser.add_argument("--sample", default="", help="what is known about the sample")
  parser.add_argument("--top", type=int, default=3, help="suggestions to fit (1-6)")
  parser.add_argument("--extra-models", help="also fit these, e.g. 'sphere,cylinder'")
  parser.add_argument("--sld", type=float, help="known particle SLD (1e-6/Å²)")
  parser.add_argument("--sld-solvent", type=float, help="known solvent SLD (1e-6/Å²)")
  parser.add_argument(
    "--fixed",
    help="other known values held fixed, e.g. 'sld_core=0.4,temperature=298'",
  )
  parser.add_argument("--q-min", type=float)
  parser.add_argument("--q-max", type=float)
  parser.add_argument("--max-steps", type=int, default=12, help="model turn budget")
  parser.add_argument("--no-plot", action="store_true", help="skip the PNG fit plot")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if bool(args.data) == (args.example is not None):
    parser.error("give data file(s) or --example NAME (not both, not neither)")
  series = list(args.data) if len(args.data) > 1 else []
  single = args.data[0] if len(args.data) == 1 else None
  if args.per_file and not series:
    parser.error("--per-file needs two or more data files")
  per_file = []
  for item in args.per_file:
    name, separator, values = item.partition("=")
    if not separator:
      parser.error("--per-file must look like NAME=V1,V2,...")
    per_file.append(f"{name.strip()}={values.replace(',', '/').replace(' ', '')}")
  if args.out:
    out = args.out
  elif series:
    out = series[0].with_name(f"{series[0].stem}_series_analysis")
  elif single is not None:
    out = single.with_name(f"{single.stem}_analysis")
  else:
    out = Path(f"{args.example}_analysis")
  known = [
    item
    for item in (
      f"sld={args.sld}" if args.sld is not None else "",
      f"sld_solvent={args.sld_solvent}" if args.sld_solvent is not None else "",
      args.fixed or "",
    )
    if item
  ]
  config = AgentConfig(
    fixed_parameters=",".join(known) or None,
    out=out,
    data=single,
    series=series,
    per_file_parameters=",".join(per_file) or None,
    example=args.example,
    model=args.model,
    base_url=args.base_url,
    api_key=args.api_key,
    sample=args.sample,
    top=max(1, min(args.top, 6)),
    extra_models=args.extra_models,
    q_min=args.q_min,
    q_max=args.q_max,
    max_steps=max(1, args.max_steps),
    plot=not args.no_plot,
  )
  try:
    result = asyncio.run(run_series(config) if series else run_agent(config))
  except (FileNotFoundError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  return 0 if result.status == "fitted" else 1


if __name__ == "__main__":
  raise SystemExit(main())
