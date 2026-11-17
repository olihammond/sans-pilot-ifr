"""Tests for guided analysis sessions and the fixed procedure."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from fastmcp import Client
from iquana import IQuana, examples

from sans_pilot import procedure, sessions
from sans_pilot.main import build_server

DATA = Path(__file__).with_name("simulated_sans_data.csv")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
  uploads = tmp_path / "uploads"
  uploads.mkdir()
  shutil.copyfile(DATA, uploads / DATA.name)
  monkeypatch.setenv("UPLOAD_DIR", str(uploads))
  monkeypatch.setenv("SANS_PILOT_RUNS_DIR", str(tmp_path / "runs"))
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  return tmp_path


def test_session_persists_model_parameters_and_fit_between_calls(workspace):
  started = sessions.create_session(file=None, example="sphere", user_id=None)
  session_id = started["session_id"]
  assert started["features"]["hints"]
  assert started["next_steps"][0].startswith("suggest-sans-models")

  with pytest.raises(sessions.SessionError, match="No model is set"):
    sessions.run_fit(session_id, method="lm", user_id=None)

  described = sessions.set_model(
    session_id, model="sphere", structure_factor=None, user_id=None
  )
  assert "description" in described["parameters"]["radius"]

  with pytest.raises(sessions.SessionError, match="No parameter is set to vary"):
    sessions.run_fit(session_id, method="lm", user_id=None)
  with pytest.raises(sessions.SessionError, match="not a parameter of sphere"):
    sessions.set_parameter(
      session_id,
      name="length",
      value=1.0,
      minimum=None,
      maximum=None,
      vary=None,
      user_id=None,
    )

  radius = examples.get_example("sphere").params["radius"]["value"]
  for name in ("radius", "scale", "background"):
    changed = sessions.set_parameter(
      session_id,
      name=name,
      value=radius * 0.9 if name == "radius" else None,
      minimum=None,
      maximum=None,
      vary=True,
      user_id=None,
    )
  assert set(changed["varying"]) == {"radius", "scale", "background"}

  fitted = sessions.run_fit(session_id, method="lm", user_id=None)
  assert fitted["statistics"]["n_free"] == 3
  assert fitted["diagnostics"]["verdict"] in {"good", "acceptable", "poor"}
  assert fitted["fitted"]["radius"]["value"] == pytest.approx(radius, rel=0.1)
  assert "radius" in fitted["fitted"]
  assert fitted["residuals"]["points"] > 5

  # State was written to disk, not kept in memory.
  status = sessions.session_status(session_id, user_id=None)
  assert status["model"] == "sphere"
  assert status["last_fit"]["statistics"] == fitted["statistics"]
  assert status["parameters"]["radius"]["vary"] is True

  exported = sessions.export_session(
    session_id, None, workspace / "export", include_plot=False
  )
  assert set(exported["artifacts"]) == {
    "report.md",
    "analysis.json",
    "sasview_parameter_values.txt",
  }
  report = (workspace / "export" / "report.md").read_text(encoding="utf-8")
  assert "## Diagnostics" in report and "## Steps taken" in report
  reloaded = IQuana.load_analysis(str(workspace / "export" / "analysis.json"))
  assert reloaded.params["radius"]["value"] == pytest.approx(
    fitted["fitted"]["radius"]["value"], rel=1e-5
  )


def test_sessions_are_scoped_to_the_user_and_validate_names(workspace):
  started = sessions.create_session(file=None, example="sphere", user_id="alice")
  with pytest.raises(sessions.SessionError, match="was not found"):
    sessions.load_session(started["session_id"], user_id="bob")
  with pytest.raises(sessions.SessionError, match="must look like"):
    sessions.load_session("../../etc", user_id="alice")
  with pytest.raises(sessions.SessionError, match="Unknown model 'spher'"):
    sessions.set_model(
      started["session_id"], model="spher", structure_factor=None, user_id="alice"
    )
  with pytest.raises(sessions.SessionError, match="exactly one"):
    sessions.create_session(file=None, example=None, user_id="alice")


def test_q_range_and_polydispersity_are_applied_to_the_saved_analysis(workspace):
  session_id = sessions.create_session(file=DATA.name, example=None, user_id=None)[
    "session_id"
  ]
  sessions.set_model(session_id, model="sphere", structure_factor=None, user_id=None)
  narrowed = sessions.set_q_range(session_id, q_min=0.02, q_max=0.3, user_id=None)
  assert narrowed["data"]["actual_q_range"]["min"] >= 0.02
  assert narrowed["data"]["actual_q_range"]["max"] <= 0.3
  spread = sessions.set_polydispersity(
    session_id,
    parameter="radius",
    width=0.15,
    distribution="schulz",
    vary=False,
    user_id=None,
  )
  assert spread["polydispersity"]["radius"] == {
    "width": 0.15,
    "distribution": "schulz",
    "vary": False,
  }
  fitter = sessions.open_fitter(sessions.load_session(session_id, user_id=None))
  q_range = fitter.get_q_range()
  assert q_range is not None
  assert (float(q_range[0]), float(q_range[1])) == (0.02, 0.3)


def test_without_guess_the_procedure_asks_for_models_then_fits_them(workspace):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  unavailable = procedure.suggest(session_id, None, 5)
  assert unavailable["available"] is False

  needs_model = procedure.fit_suggestions(session_id, None, 3, None)
  assert needs_model["status"] == "needs_model"

  with pytest.raises(sessions.SessionError, match="Unknown model"):
    procedure.fit_suggestions(session_id, None, 3, "sphere,not_a_model")

  fitted = procedure.fit_suggestions(session_id, None, 3, "sphere, ellipsoid")
  assert fitted["status"] == "fitted"
  assert {item["model"] for item in fitted["candidates"]} == {"sphere", "ellipsoid"}
  assert [item["rank"] for item in fitted["candidates"]] == [1, 2]
  assert fitted["selected_model"] == fitted["candidates"][0]["model"]
  assert all(item["started_from"] == "model defaults" for item in fitted["candidates"])

  other = fitted["candidates"][1]["model"]
  switched = procedure.use_candidate(session_id, None, other)
  assert switched["model"] == other
  with pytest.raises(sessions.SessionError, match="No fitted candidate"):
    procedure.use_candidate(session_id, None, "cylinder")


def test_library_suggestions_supply_starting_values_and_ranking(workspace, monkeypatch):
  radius = examples.get_example("sphere").params["radius"]["value"]
  calls: dict[str, int] = {}

  def fake_suggest(data, *, top, starting_values, also):
    calls.update(top=top, starting_values=starting_values, also=also)
    return {
      "requested": [
        {"model": model, "structure_factor": sf, "expression": model}
        for model, sf in also
      ],
      "library": "fake library",
      "suggestions": [
        {
          "model": "ellipsoid",
          "structure_factor": None,
          "expression": "ellipsoid",
          "library_chisq_n": 1.0,
          "loadable": True,
          "start": {
            "parameters": {
              "radius_polar": radius,
              "radius_equatorial": radius,
            },
            "polydispersity": {},
          },
        },
        {
          "model": "sphere",
          "structure_factor": None,
          "expression": "sphere",
          "library_chisq_n": 2.0,
          "loadable": True,
          "start": {
            "parameters": {"radius": radius * 0.95},
            "polydispersity": {},
          },
        },
        {
          "model": "gone",
          "structure_factor": None,
          "expression": "gone",
          "library_chisq_n": 3.0,
          "loadable": False,
        },
      ],
    }

  monkeypatch.setattr(procedure, "guess_status", lambda: {"available": True})
  monkeypatch.setattr(procedure, "suggest_models", fake_suggest)

  result = procedure.guided_analysis(
    None, "sphere", None, 2, None, None, None, include_baselines=False
  )

  assert calls == {"top": 5, "starting_values": 2, "also": []}
  assert result["status"] == "fitted"
  models = [item["model"] for item in result["candidates"]]
  assert sorted(models) == ["ellipsoid", "sphere"]
  assert all(item["started_from"] == "curve library" for item in result["candidates"])
  sphere = next(item for item in result["candidates"] if item["model"] == "sphere")
  assert sphere["fitted"]["radius"] == pytest.approx(radius, rel=0.1)
  # BIC ranks the model with fewer free parameters first when both fit equally.
  ellipsoid = next(
    item for item in result["candidates"] if item["model"] == "ellipsoid"
  )
  assert sphere["n_free"] < ellipsoid["n_free"]
  assert result["selected_model"] == result["candidates"][0]["model"]
  assert "BIC" in result["ranking"]


def test_tool_profiles_expose_the_expected_tools():
  async def names(profile: str) -> set[str]:
    async with Client(build_server(profile)) as client:
      return {tool.name for tool in await client.list_tools()}

  full = asyncio.run(names("full"))
  guided = asyncio.run(names("guided"))
  both = asyncio.run(names("all"))

  assert "fit-sans-model" in full and "run-sans-fit" not in full
  assert "run-sans-guided-analysis" in guided and "fit-sans-model" not in guided
  assert {"list-uploaded-sans-files", "read-sans-artifact"} <= guided
  assert both == full | guided
  with pytest.raises(ValueError):
    build_server("tiny")


def test_guided_tool_schemas_stay_small_and_flat():
  async def schema_size() -> tuple[int, int]:
    import json

    async with Client(build_server("guided")) as client:
      tools = await client.list_tools()
    text = json.dumps([tool.input_schema for tool in tools])
    return len(text), text.count("$ref") + text.count("$defs")

  size, references = asyncio.run(schema_size())
  assert size < 8_000
  assert references == 0


def test_simple_baselines_are_fitted_alongside_library_suggestions(
  workspace, monkeypatch
):
  radius = examples.get_example("sphere").params["radius"]["value"]
  seen: dict[str, list] = {}

  def fake_suggest(data, *, top, starting_values, also):
    seen["also"] = also
    return {
      "suggestions": [
        {
          "model": "ellipsoid",
          "structure_factor": None,
          "expression": "ellipsoid",
          "library_chisq_n": 1.0,
          "loadable": True,
          "start": {
            "parameters": {"radius_polar": radius, "radius_equatorial": radius},
            "polydispersity": {},
          },
        }
      ],
      "requested": [
        {
          "model": model,
          "structure_factor": sf,
          "expression": model,
          "start": {"parameters": {"radius": radius}, "polydispersity": {}},
        }
        for model, sf in also
        if model == "sphere"
      ],
    }

  monkeypatch.setattr(procedure, "guess_status", lambda: {"available": True})
  monkeypatch.setattr(procedure, "suggest_models", fake_suggest)
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]

  result = procedure.fit_suggestions(session_id, None, 1, None)

  assert ("sphere", None) in seen["also"]
  origins = {item["model"]: item["origin"] for item in result["candidates"]}
  assert origins == {"ellipsoid": "curve library", "sphere": "simple baseline"}
  ellipsoid = next(c for c in result["candidates"] if c["model"] == "ellipsoid")
  # Orientation angles do not affect 1D data and are never fitted.
  assert not {"theta", "phi"} & set(ellipsoid["fitted"])


def test_time_limit_interrupts_a_runaway_fit():
  import time

  started = time.perf_counter()
  with pytest.raises(sessions.FitTimeout), sessions.time_limit(0.2):
    while True:
      sum(range(1000))
  assert time.perf_counter() - started < 5
  # Outside the block the alarm is cleared.
  with sessions.time_limit(5):
    pass


def test_export_reports_the_last_fit_when_settings_changed_afterwards(workspace):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  fitted = procedure.fit_suggestions(session_id, None, 1, "sphere")
  radius = fitted["selected_fit"]["fitted"]["radius"]["value"]

  changed = sessions.set_parameter(
    session_id,
    name="scale",
    value=None,
    minimum=None,
    maximum=None,
    vary=False,
    user_id=None,
  )
  assert changed["unfitted_changes"] == ["scale: vary=False"]
  assert "run-sans-fit" in changed["note"]
  assert sessions.session_status(session_id, user_id=None)["unfitted_changes"]

  exported = sessions.export_session(
    session_id, None, workspace / "export", include_plot=False
  )
  assert exported["summary"]["unfitted_changes"] == ["scale: vary=False"]
  assert "sasview_parameter_values.txt" in exported["artifacts"]
  report = (workspace / "export" / "report.md").read_text(encoding="utf-8")
  assert "Settings changed after this fit and not fitted" in report
  assert "χ²/dof" in report
  # The exported analysis is the completed fit, with scale still free.
  reloaded = IQuana.load_analysis(str(workspace / "export" / "analysis.json"))
  assert reloaded.params["scale"]["vary"] is True
  assert reloaded.params["radius"]["value"] == pytest.approx(radius, rel=1e-5)

  # Refitting clears the pending list.
  sessions.run_fit(session_id, method="lm", user_id=None)
  assert "unfitted_changes" not in sessions.session_status(session_id, user_id=None)
