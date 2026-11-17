"""Tests for the local-LLM runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from sans_pilot.agent import (
  AgentConfig,
  decode_arguments,
  parse_tool_calls,
  run_agent,
)

DATA = Path(__file__).with_name("simulated_sans_data.csv")


def _reply(**message) -> httpx.Response:
  return httpx.Response(
    200, json={"choices": [{"message": {"role": "assistant", **message}}]}
  )


def test_tool_calls_are_parsed_from_native_and_text_formats():
  native = parse_tool_calls(
    {
      "tool_calls": [
        {"id": "a", "function": {"name": "run-sans-fit", "arguments": "{}"}}
      ]
    }
  )
  assert native == [{"id": "a", "name": "run-sans-fit", "arguments": "{}"}]

  text = parse_tool_calls(
    {
      "content": 'ok <tool_call>{"name": "run-sans-fit", "arguments": '
      '{"method": "lm"}}</tool_call>'
    }
  )
  assert text[0]["name"] == "run-sans-fit"
  assert text[0]["arguments"] == {"method": "lm"}

  assert parse_tool_calls({"content": "The radius is 50 Å."}) == []
  assert parse_tool_calls({"content": '{"name": "rm-rf", "arguments": {}}'}) == []
  assert decode_arguments('{"a": 1}') == {"a": 1}
  assert decode_arguments({"a": 1}) == {"a": 1}
  assert isinstance(decode_arguments("{broken"), str)
  assert isinstance(decode_arguments("[1]"), str)


def test_runner_without_a_model_writes_a_deterministic_report(tmp_path, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  config = AgentConfig(
    out=tmp_path / "out", data=DATA, extra_models="sphere", plot=False
  )
  result = asyncio.run(run_agent(config, log=lambda _line: None))

  assert result.status == "fitted"
  assert result.model == "sphere"
  assert result.llm_steps == 0
  report = result.report.read_text(encoding="utf-8")
  assert "## Models tried" in report
  assert "computed deterministically" in report
  assert "Interpretation by" not in report
  assert (tmp_path / "out" / "analysis.json").is_file()
  assert not (tmp_path / "out" / "transcript.json").exists()


def test_runner_lets_a_model_refine_within_the_framework(tmp_path, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  requests: list[dict] = []

  def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    requests.append(body)
    turn = len(requests)
    if turn == 1:
      return _reply(
        content="",
        tool_calls=[
          {
            "id": "c1",
            "type": "function",
            "function": {
              "name": "set-sans-parameter",
              "arguments": json.dumps({"name": "radius", "min": 5, "max": 500}),
            },
          },
          {
            "id": "c2",
            "type": "function",
            "function": {"name": "set-sans-parameter", "arguments": "{broken"},
          },
          {
            "id": "c3",
            "type": "function",
            "function": {"name": "delete-everything", "arguments": "{}"},
          },
        ],
      )
    if turn == 2:
      return _reply(
        content="",
        tool_calls=[
          {
            "id": "c4",
            "type": "function",
            "function": {"name": "run-sans-fit", "arguments": {}},
          }
        ],
      )
    return _reply(content="Spheres describe the data.")

  config = AgentConfig(
    out=tmp_path / "out",
    data=DATA,
    model="test-model",
    base_url="http://llm.test/v1",
    sample="dilute silica spheres",
    extra_models="sphere",
    plot=False,
  )
  result = asyncio.run(
    run_agent(config, transport=httpx.MockTransport(handler), log=lambda _line: None)
  )

  assert result.llm_steps == 3
  assert result.interpretation == "Spheres describe the data."
  # The model only sees the refinement tools, never load/export.
  offered = {tool["function"]["name"] for tool in requests[0]["tools"]}
  assert "run-sans-fit" in offered
  assert not offered & {"run-sans-guided-analysis", "export-sans-analysis"}
  assert "dilute silica spheres" in requests[0]["messages"][1]["content"]

  tool_messages = [m for m in requests[1]["messages"] if m["role"] == "tool"]
  assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2", "c3"]
  assert '"max": 500' in tool_messages[0]["content"]
  assert "not valid JSON" in tool_messages[1]["content"]
  assert "Unknown tool" in tool_messages[2]["content"]
  # session_id is filled in when the model omits it.
  assert '"statistics"' in requests[2]["messages"][-1]["content"]

  report = result.report.read_text(encoding="utf-8")
  assert "## Interpretation by `test-model`" in report
  assert "has not been" in report
  assert (tmp_path / "out" / "analysis_guided.json").is_file()
  assert (tmp_path / "out" / "transcript.json").is_file()


def test_runner_survives_an_endpoint_without_tool_calling(tmp_path, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  seen: list[bool] = []

  def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    seen.append("tools" in body)
    if "tools" in body:
      return httpx.Response(400, json={"error": "model does not support tools"})
    return _reply(content="Interpretation only.")

  config = AgentConfig(
    out=tmp_path / "out",
    data=DATA,
    model="no-tools",
    extra_models="sphere",
    plot=False,
  )
  result = asyncio.run(
    run_agent(config, transport=httpx.MockTransport(handler), log=lambda _line: None)
  )

  assert seen == [True, False]
  assert result.interpretation == "Interpretation only."
  assert "rejected tool calling" in (result.llm_note or "")


def test_runner_still_reports_when_the_endpoint_is_unreachable(tmp_path, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")

  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)

  config = AgentConfig(
    out=tmp_path / "out",
    data=DATA,
    model="offline",
    extra_models="sphere",
    plot=False,
  )
  result = asyncio.run(
    run_agent(config, transport=httpx.MockTransport(handler), log=lambda _line: None)
  )

  assert result.status == "fitted"
  assert result.interpretation is None
  assert "Model request failed" in (result.llm_note or "")
  assert "produced no interpretation" in result.report.read_text(encoding="utf-8")


def test_runner_reminds_a_model_that_stops_without_refitting(tmp_path, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  requests: list[dict] = []

  def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    requests.append(body)
    turn = len(requests)
    if turn == 1:
      return _reply(
        content="",
        tool_calls=[
          {
            "id": "c1",
            "type": "function",
            "function": {
              "name": "set-sans-parameter",
              "arguments": json.dumps({"name": "scale", "vary": False}),
            },
          }
        ],
      )
    if turn == 2:
      return _reply(content="")  # stops early with nothing to say
    if turn == 3:
      return _reply(content="")  # ignores the reminder as well
    return _reply(content="Done.")

  config = AgentConfig(
    out=tmp_path / "out",
    data=DATA,
    model="quiet",
    extra_models="sphere",
    plot=False,
  )
  result = asyncio.run(
    run_agent(config, transport=httpx.MockTransport(handler), log=lambda _line: None)
  )

  reminders = [m["content"] for m in requests[-1]["messages"] if m["role"] == "user"][
    1:
  ]
  assert "without refitting: scale: vary=False" in reminders[0]
  assert reminders[1].startswith("Your reply was empty")
  assert result.interpretation == "Done."
  # The export still succeeds and reports the last completed fit.
  report = result.report.read_text(encoding="utf-8")
  assert "Settings changed after this fit and not fitted" in report
  assert (tmp_path / "out" / "analysis.json").is_file()
  assert (tmp_path / "out" / "transcript.json").is_file()


def test_report_says_when_the_models_turn_changed_nothing(tmp_path, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  requests: list[dict] = []

  def call(identifier: str, name: str, arguments: dict) -> dict:
    return {
      "id": identifier,
      "type": "function",
      "function": {"name": name, "arguments": json.dumps(arguments)},
    }

  def handler(request: httpx.Request) -> httpx.Response:
    requests.append(json.loads(request.content))
    turn = len(requests)
    if turn == 1:
      return _reply(
        content="",
        tool_calls=[call("a", "set-sans-parameter", {"name": "scale", "vary": False})],
      )
    if turn == 2:
      return _reply(content="", tool_calls=[call("b", "run-sans-fit", {})])
    return _reply(content="| Parameter | Value |\n|---|---|\n| radius | 1 |")

  config = AgentConfig(
    out=tmp_path / "out",
    data=DATA,
    model="fixer",
    extra_models="sphere",
    fixed_parameters="sld_solvent=6.3",
    plot=False,
  )
  result = asyncio.run(
    run_agent(config, transport=httpx.MockTransport(handler), log=lambda _line: None)
  )

  warned = next(m for m in requests[1]["messages"] if m["role"] == "tool")
  assert "fixed at its own fitted value" in warned["content"]
  report = result.report.read_text(encoding="utf-8")
  assert "did not change the fit" in report
  assert "Uncertainties are conditional:** scale" in report
  assert result.verdict != "good"

  page = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
  assert "SANS-PILOT-EXTRA" not in page
  assert "Interpretation by fixer" in page and "<th>Parameter</th>" in page
  assert "did not change the fit" in page
  assert "sld_solvent = 6.3" in page
