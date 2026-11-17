"""Tests for joint series analysis and multi-start fitting."""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import numpy as np
import pytest
from iquana import examples

from sans_pilot import procedure
from sans_pilot.agent import AgentConfig, run_series
from sans_pilot.series import parse_per_file, series_analysis
from sans_pilot.sessions import SessionError

SOLVENTS = (0.0, 4.5, 6.3)
TRUTH = {"radius": 60.0, "sld": 2.0, "scale": 0.02}


@pytest.fixture
def contrast_series(tmp_path, monkeypatch):
  """Three simulated curves of the same spheres in different solvents."""
  uploads = tmp_path / "uploads"
  uploads.mkdir()
  names = []
  for index, solvent in enumerate(SOLVENTS):
    data = examples.simulate(
      "sphere",
      seed=10 + index,
      npoints=90,
      qmin=0.008,
      qmax=0.3,
      sld_solvent=solvent,
      background=0.05,
      radius=TRUTH["radius"],
      sld=TRUTH["sld"],
      scale=TRUTH["scale"],
    )
    name = f"contrast_{index}.csv"
    np.savetxt(
      uploads / name,
      np.c_[data.x, data.y, data.dy],
      delimiter=",",
      header="Q,I,dI",
      comments="",
    )
    names.append(name)
  monkeypatch.setenv("UPLOAD_DIR", str(uploads))
  monkeypatch.setenv("SANS_PILOT_RUNS_DIR", str(tmp_path / "runs"))
  monkeypatch.setenv("SANS_PILOT_GUESS", "off")
  return tmp_path, names


def test_per_file_values_are_parsed_and_validated():
  assert parse_per_file("sld_solvent=-0.5/2/6.3, temperature=20/30/40", 3) == {
    "sld_solvent": [-0.5, 2.0, 6.3],
    "temperature": [20.0, 30.0, 40.0],
  }
  assert parse_per_file(None, 3) == {}
  with pytest.raises(SessionError, match="one value per file"):
    parse_per_file("sld_solvent=1/2", 3)
  with pytest.raises(SessionError, match="must look like"):
    parse_per_file("sld_solvent", 3)


def test_alternative_starts_mirror_sizes_and_stay_inside_bounds():
  parameters = {
    "radius_polar": {"value": 150.0, "min": 40.0, "max": 600.0, "vary": True},
    "radius_equatorial": {"value": 130.0, "min": 30.0, "max": 500.0, "vary": True},
    "scale": {"value": 1.0, "min": 0.1, "max": 10.0, "vary": True},
    "sld": {"value": 1.0, "min": -1.0, "max": 10.0, "vary": True},
    "length": {"value": 10.0, "min": 1.0, "max": 100.0, "vary": False},
  }
  starts = procedure.alternative_starts(parameters, "ellipsoid", 5)

  assert len(starts) == 5
  assert all(set(start) == {"radius_polar", "radius_equatorial"} for start in starts)
  mirrored = starts[0]
  assert mirrored["radius_polar"] < mirrored["radius_equatorial"]
  assert math.isclose(
    mirrored["radius_polar"] * 150.0, mirrored["radius_equatorial"] * 130.0
  )
  for start in starts:
    assert 40.0 < start["radius_polar"] < 600.0
    assert 30.0 < start["radius_equatorial"] < 500.0
  assert starts == procedure.alternative_starts(parameters, "ellipsoid", 5)
  assert procedure.alternative_starts({"scale": parameters["scale"]}, "x", 3) == []


def test_joint_fit_recovers_shared_structure_and_the_particle_sld(contrast_series):
  tmp_path, names = contrast_series

  outcome = series_analysis(
    names,
    None,
    "sld_solvent=" + "/".join(str(value) for value in SOLVENTS),
    None,
    3,
    "sphere,ellipsoid",
    tmp_path / "out",
    include_plot=False,
  )
  summary = outcome["summary"]
  fit = summary["selected_fit"]

  assert summary["selected_model"] == "sphere"
  assert fit["verdict"] == "good"
  assert fit["statistics"]["n_datasets"] == 3
  shared = fit["shared"]
  assert set(shared) == {"radius", "scale", "sld"}
  assert shared["radius"]["value"] == pytest.approx(TRUTH["radius"], rel=0.02)
  assert shared["sld"]["value"] == pytest.approx(TRUTH["sld"], abs=0.1)
  assert shared["scale"]["value"] == pytest.approx(TRUTH["scale"], rel=0.1)
  assert fit["free_slds"] == ["sld"]
  for index, curve in enumerate(fit["datasets"]):
    assert curve["known"] == {"sld_solvent": SOLVENTS[index]}
    assert curve["background"] == pytest.approx(0.05, abs=0.02)
    assert curve["reduced_chisq"] < 2

  ranks = {item["model"]: item["rank"] for item in summary["candidates"]}
  assert ranks["sphere"] == 1 and ranks["ellipsoid"] == 2
  written = set(outcome["artifacts"])
  assert {"report.html", "report.md", "series_result.json"} <= written
  assert {f"analysis_contrast_{index}.json" for index in range(3)} <= written
  page = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
  assert "Shared parameters" in page and "Plotly.newPlot" in page
  assert "sld_solvent = 4.5" in page


def test_without_known_contrasts_each_curve_keeps_its_own_scale(contrast_series):
  tmp_path, names = contrast_series

  summary = series_analysis(
    names, None, None, None, 1, "sphere", tmp_path / "out", include_plot=False
  )["summary"]

  fit = summary["selected_fit"]
  assert set(fit["shared"]) == {"radius"}
  assert fit["shared"]["radius"]["value"] == pytest.approx(TRUTH["radius"], rel=0.02)
  assert all("scale" in curve for curve in fit["datasets"])
  assert "own scale" in summary["note"]


def test_series_needs_several_files_models_and_uncertainties(contrast_series):
  tmp_path, names = contrast_series
  with pytest.raises(SessionError, match="at least two files"):
    series_analysis(names[:1], None, None, None, 1, "sphere", tmp_path / "o")
  with pytest.raises(SessionError, match="Name models to fit"):
    series_analysis(names, None, None, None, 1, None, tmp_path / "o")
  with pytest.raises(SessionError, match="3 files"):
    series_analysis(names, None, "sld_solvent=1/2", None, 1, "sphere", tmp_path / "o")


def test_runner_series_mode_writes_reports_and_an_interpretation(contrast_series):
  tmp_path, names = contrast_series
  uploads = tmp_path / "uploads"
  requests: list[dict] = []

  def handler(request: httpx.Request) -> httpx.Response:
    requests.append(json.loads(request.content))
    return httpx.Response(
      200,
      json={"choices": [{"message": {"role": "assistant", "content": "Spheres."}}]},
    )

  config = AgentConfig(
    out=tmp_path / "series_out",
    series=[uploads / name for name in names],
    per_file_parameters="sld_solvent=" + "/".join(str(v) for v in SOLVENTS),
    extra_models="sphere",
    model="summariser",
    sample="silica-like spheres, contrast series",
    plot=False,
  )
  result = asyncio.run(
    run_series(config, transport=httpx.MockTransport(handler), log=lambda _line: None)
  )

  assert result.model == "sphere" and result.verdict == "good"
  assert result.interpretation == "Spheres."
  assert "tools" not in requests[0]
  assert "contrast series" in requests[0]["messages"][1]["content"]
  report = result.report.read_text(encoding="utf-8")
  assert "## Shared parameters" in report
  assert "## Interpretation by `summariser`" in report
  page = (tmp_path / "series_out" / "report.html").read_text(encoding="utf-8")
  assert "Interpretation by summariser" in page and "SANS-PILOT-EXTRA" not in page
