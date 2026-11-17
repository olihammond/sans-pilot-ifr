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
    "report.html",
  }
  page = (workspace / "export" / "report.html").read_text(encoding="utf-8")
  assert "Plotly.newPlot" in page and "Fitted parameters" in page
  assert "<td>radius</td>" in page
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

  def fake_suggest(data, *, top, starting_values, also, usable=None):
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

  assert calls == {"top": 8, "starting_values": 2, "also": []}
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

  def fake_suggest(data, *, top, starting_values, also, usable=None):
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


def test_fixing_a_parameter_at_its_fitted_value_is_flagged(workspace):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  procedure.fit_suggestions(session_id, None, 1, "sphere")

  def fix(name, value=None, vary=False):
    return sessions.set_parameter(
      session_id,
      name=name,
      value=value,
      minimum=None,
      maximum=None,
      vary=vary,
      user_id=None,
    )

  flagged = fix("scale")
  assert "fixed at the value the last fit gave it" in flagged["warning"]
  refit = sessions.run_fit(session_id, method="lm", user_id=None)
  assert refit["conditional_on"] == ["scale"]
  assert refit["diagnostics"]["verdict"] != "good"
  assert "Uncertainties are conditional: scale" in refit["diagnostics"]["issues"][0]

  sessions.export_session(session_id, None, workspace / "export", include_plot=False)
  page = (workspace / "export" / "report.html").read_text(encoding="utf-8")
  assert "Uncertainties are conditional." in page

  # A rounded copy of the fitted value is still the fitted value.
  fitted_scale = refit["fixed"]["scale"]
  fix("scale", vary=True)
  again = sessions.run_fit(session_id, method="lm", user_id=None)
  rounded = float(f"{again['fitted']['scale']['value']:.3g}")
  assert "warning" in fix("scale", value=rounded)
  assert fitted_scale > 0

  # A genuinely different value, or freeing the parameter, clears the flag.
  known = fix("scale", value=1.0)
  assert "warning" not in known and "widened" in known["bounds_note"]
  assert "warning" in fix("background")
  assert "warning" not in fix("background", vary=True)
  cleared = sessions.run_fit(session_id, method="lm", user_id=None)
  assert cleared["conditional_on"] == []


def test_known_values_are_held_fixed_in_every_candidate(workspace):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  with pytest.raises(sessions.SessionError, match="name=number"):
    procedure.fit_suggestions(session_id, None, 1, "sphere", True, "sld")

  free = procedure.fit_suggestions(session_id, None, 1, "sphere")
  result = procedure.fit_suggestions(
    session_id, None, 1, "sphere", True, "sld=2.5, sld_solvent=6.36"
  )

  assert result["fixed_parameters"] == {"sld": 2.5, "sld_solvent": 6.36}
  assert result["newly_fitted"] == ["sphere"]  # new known values: refit
  fixed = result["selected_fit"]["fixed"]
  assert (fixed["sld"], fixed["sld_solvent"]) == (2.5, 6.36)
  # Same curve, different contrast: the scale absorbs (5 / 3.86)^2 and the
  # fit quality is unchanged.
  before = free["candidates"][0]
  after = result["candidates"][0]
  assert after["reduced_chisq"] == pytest.approx(before["reduced_chisq"], rel=1e-3)
  assert after["fitted"]["scale"] / before["fitted"]["scale"] == pytest.approx(
    (5 / 3.86) ** 2, rel=0.02
  )


def test_extra_models_are_fitted_without_refitting_existing_candidates(
  workspace, monkeypatch
):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  procedure.fit_suggestions(session_id, None, 1, "sphere")
  sessions.set_parameter(
    session_id,
    name="radius",
    value=None,
    minimum=1.0,
    maximum=None,
    vary=None,
    user_id=None,
  )
  refined = sessions.run_fit(session_id, method="lm", user_id=None)

  fitted: list[str] = []
  original = procedure._fit_candidate

  def counting(session, model, *args, **kwargs):
    fitted.append(model)
    return original(session, model, *args, **kwargs)

  monkeypatch.setattr(procedure, "_fit_candidate", counting)
  again = procedure.fit_suggestions(session_id, None, 1, "sphere,cylinder")

  assert fitted == ["cylinder"]
  assert again["newly_fitted"] == ["cylinder"]
  assert {item["model"] for item in again["candidates"]} == {"sphere", "cylinder"}
  # The refined sphere stays selected unless the new candidate ranks first.
  if again["candidates"][0]["model"] == "sphere":
    assert again["selected_model"] == "sphere"
    assert again["selected_fit"]["statistics"] == refined["statistics"]

  # A changed Q range invalidates the kept candidates.
  fitted.clear()
  sessions.set_q_range(session_id, q_min=0.005, q_max=None, user_id=None)
  procedure.fit_suggestions(session_id, None, 1, "sphere")
  assert fitted == ["sphere"]


def test_fixing_an_undetermined_parameter_is_not_treated_as_hiding_anything(workspace):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  session = sessions.load_session(session_id, user_id=None)
  session.meta["last_fit"] = {
    "fitted": {
      "thick_solvent": {"value": 6.09, "stderr": 1e8},
      "radius": {"value": 245.0, "stderr": 0.5},
    }
  }

  quiet = sessions._track_conditional(session, "thick_solvent", value=0.0, vary=False)
  loud = sessions._track_conditional(session, "radius", value=245.2, vary=False)

  assert quiet is None
  assert session.meta["fixed_after_fit"]["thick_solvent"]["undetermined"] is True
  assert "fixed at the value the last fit gave it" in (loud or "")
  assert session.meta["conditional_on"] == ["radius"]


def test_parameters_that_only_rescale_the_intensity_are_not_fitted(workspace):
  import numpy as np
  from iquana import examples as iquana_examples

  data = iquana_examples.simulate(
    "vesicle", seed=5, npoints=120, qmin=0.004, qmax=0.3, radius=200.0, thickness=30.0
  )
  path = workspace / "uploads" / "vesicle.csv"
  np.savetxt(
    path, np.c_[data.x, data.y, data.dy], delimiter=",", header="Q,I,dI", comments=""
  )
  session_id = sessions.create_session(file="vesicle.csv", example=None, user_id=None)[
    "session_id"
  ]

  result = procedure.fit_suggestions(session_id, None, 1, "vesicle")
  candidate = result["candidates"][0]

  assert "volfraction" not in candidate["fitted"]
  assert "scale" in candidate["fitted"]
  assert any(
    "only rescales the intensity" in note for note in candidate["procedure_notes"]
  )
  issues = " ".join(result["selected_fit"]["diagnostics"]["issues"])
  assert "scale & volfraction" not in issues
  exported = sessions.export_session(
    session_id, None, workspace / "export", include_plot=False
  )
  report = exported["artifacts"]["report.md"].read_text(encoding="utf-8")
  assert "Notes from the automatic procedure" in report


def test_scale_degeneracy_is_read_from_the_covariance():
  import numpy as np

  cov = np.array([[1.0, -0.9999, 0.2], [-0.9999, 1.0, 0.1], [0.2, 0.1, 1.0]])
  labels = ["scale", "volfraction", "radius"]
  assert procedure._degenerate_with_scale({"cov": cov, "cov_labels": labels}) == [
    "volfraction"
  ]
  assert procedure._degenerate_with_scale({"cov": cov, "cov_labels": labels[1:]}) == []
  assert procedure._degenerate_with_scale({}) == []


def test_single_shell_counterpart_and_engine_check(workspace, monkeypatch):
  calls: list[dict] = []

  def fake_suggest(data, *, top, starting_values, also, usable=None):
    calls.append({"also": also, "usable": usable})
    if usable is None:  # the follow-up request for an unusable model's counterpart
      return {
        "suggestions": [],
        "requested": [
          {"model": model, "structure_factor": sf, "expression": model}
          for model, sf in also
        ],
      }
    return {
      "suggestions": [
        {
          "model": "multilayer_vesicle",
          "structure_factor": None,
          "expression": "multilayer_vesicle",
          "library_chisq_n": 1.0,
          "loadable": True,
          "start": {
            "parameters": {"n_shells": 1.2, "radius": 400.0, "thick_shell": 30.0},
            "polydispersity": {"thick_shell": {"width": 0.1, "distribution": "schulz"}},
          },
        },
        {
          "model": "onion",
          "structure_factor": None,
          "expression": "onion",
          "library_chisq_n": 2.0,
          "loadable": False,
        },
      ],
      "requested": [],
    }

  monkeypatch.setattr(procedure, "guess_status", lambda: {"available": True})
  monkeypatch.setattr(procedure, "suggest_models", fake_suggest)
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]

  result = procedure.fit_suggestions(session_id, None, 1, None, include_baselines=False)

  # The unusable onion's counterpart is requested from the library; the
  # vesicle starts from the multi-shell values with names translated.
  assert calls[1]["also"] == [("core_shell_sphere", None)]
  origins = {item["model"]: item["origin"] for item in result["candidates"]}
  assert origins == {
    "multilayer_vesicle": "curve library",
    "vesicle": "single-shell counterpart",
    "core_shell_sphere": "single-shell counterpart",
  }
  vesicle = next(c for c in result["candidates"] if c["model"] == "vesicle")
  assert vesicle["started_from"] == "curve library"
  usable = calls[0]["usable"]
  assert usable("sphere") is True
  assert usable("not_a_model") is False


def _upload_sphere(workspace, scale):
  import numpy as np

  data = examples.simulate(
    "sphere", seed=3, npoints=100, qmin=0.005, qmax=0.3, radius=50, scale=scale
  )
  name = f"sphere_scale_{scale:g}.csv"
  np.savetxt(
    workspace / "uploads" / name,
    np.c_[data.x, data.y, data.dy],
    delimiter=",",
    header="Q,I,dI",
    comments="",
  )
  return name


def test_scale_never_exceeds_one_and_the_cap_is_explained(workspace):
  name = _upload_sphere(workspace, scale=5.0)
  session_id = sessions.create_session(file=name, example=None, user_id=None)[
    "session_id"
  ]
  result = procedure.fit_suggestions(session_id, None, 1, "sphere")
  sphere = next(c for c in result["candidates"] if c["model"] == "sphere")
  assert sphere["fitted"]["scale"] <= 1.0
  issues = " ".join(result["selected_fit"]["diagnostics"]["issues"])
  assert "scale is at its upper limit of 1" in issues

  with pytest.raises(sessions.SessionError, match="scale cannot exceed 1"):
    sessions.set_parameter(
      session_id,
      name="scale",
      value=2.0,
      minimum=None,
      maximum=None,
      vary=None,
      user_id=None,
    )
  clipped = sessions.set_parameter(
    session_id,
    name="scale",
    value=None,
    minimum=None,
    maximum=10.0,
    vary=True,
    user_id=None,
  )
  assert clipped["parameter"]["scale"]["max"] == 1.0


def test_scale_limit_can_be_raised_for_arbitrary_units(workspace, monkeypatch):
  monkeypatch.setenv("SANS_PILOT_SCALE_MAX", "100")
  name = _upload_sphere(workspace, scale=5.0)
  session_id = sessions.create_session(file=name, example=None, user_id=None)[
    "session_id"
  ]
  result = procedure.fit_suggestions(session_id, None, 1, "sphere")
  sphere = next(c for c in result["candidates"] if c["model"] == "sphere")
  assert sphere["fitted"]["scale"] == pytest.approx(5.0, rel=0.1)
  assert sphere["fitted"]["radius"] == pytest.approx(50.0, rel=0.05)


def test_polymer_models_keep_an_unlimited_scale(workspace):
  session_id = sessions.create_session(file=None, example="sphere", user_id=None)[
    "session_id"
  ]
  sessions.set_model(
    session_id, model="poly_gauss_coil", structure_factor=None, user_id=None
  )
  changed = sessions.set_parameter(
    session_id,
    name="scale",
    value=25.0,
    minimum=None,
    maximum=1000.0,
    vary=True,
    user_id=None,
  )
  assert changed["parameter"]["scale"]["value"] == 25.0
  assert changed["parameter"]["scale"]["max"] == 1000.0
