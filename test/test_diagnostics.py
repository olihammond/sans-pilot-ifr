"""Tests for text-only data features and fit diagnostics."""

from __future__ import annotations

import types

import numpy as np

from sans_pilot.diagnostics import (
  assess_fit,
  describe_features,
  residual_diagnostics,
)


def _data(q, intensity, dy=None):
  return types.SimpleNamespace(
    x=q, y=intensity, dy=dy, dx=None, mask=None, qmin=None, qmax=None
  )


def test_features_describe_a_guinier_plateau_and_background():
  q = np.geomspace(0.005, 0.4, 120)
  rg = 30.0
  intensity = 10.0 * np.exp(-((q * rg) ** 2) / 3.0) + 0.01
  features = describe_features(_data(q, intensity, 0.02 * intensity))

  assert abs(features["guinier"]["rg_angstrom"] - rg) < 2
  assert abs(features["high_q_background_estimate"] - 0.01) < 1e-3
  assert features["peak"] is None
  assert any("plateau" in hint for hint in features["hints"])
  assert not any("uncertainties" in hint for hint in features["hints"])


def test_features_report_an_interference_peak_and_missing_uncertainties():
  q = np.linspace(0.01, 0.3, 150)
  intensity = 1.0 + 4.0 * np.exp(-((q - 0.1) ** 2) / (2 * 0.01**2))
  features = describe_features(_data(q, intensity))

  assert abs(features["peak"]["q_peak"] - 0.1) < 0.01
  assert any("structure factor" in hint for hint in features["hints"])
  assert any("No intensity uncertainties" in hint for hint in features["hints"])


def test_systematic_residuals_downgrade_the_verdict_and_locate_the_misfit():
  q = np.linspace(0.01, 0.3, 90)
  theory = np.ones_like(q)
  sigma = np.full_like(q, 0.1)
  data = theory + np.where(q < 0.1, 0.25, -0.02)
  residuals = residual_diagnostics(q, data, theory, sigma)
  verdict = assess_fit(
    {"reduced_chisq": 1.8, "parameters": {}}, residuals, has_uncertainties=True
  )

  assert residuals["runs_z"] < -3
  assert verdict["verdict"] == "acceptable"
  assert any("below the data at low Q" in issue for issue in verdict["issues"])
  assert any("Low-Q misfit" in item for item in verdict["suggestions"])


def test_assessment_reports_bounds_correlations_and_weak_parameters():
  verdict = assess_fit(
    {
      "reduced_chisq": 9.0,
      "on_bounds": [("radius", "max")],
      "strong_correlations": [
        {"parameters": ["radius", "scale"], "correlation": -0.99}
      ],
      "parameters": {
        "length": {"value": 100.0, "stderr": 80.0, "fixed": False},
        "sld": {"value": 1.0, "stderr": 0.0, "fixed": True},
      },
    },
    None,
    has_uncertainties=True,
  )

  assert verdict["verdict"] == "poor"
  text = " ".join(verdict["issues"])
  assert "radius (at its max)" in text
  assert "radius & scale (-0.99)" in text
  assert "Poorly determined parameters: length" in text


def test_missing_uncertainties_give_an_uncalibrated_verdict():
  verdict = assess_fit({"reduced_chisq": 0.01}, None, has_uncertainties=False)
  assert verdict["verdict"] == "uncalibrated"


def test_baseline_models_follow_the_low_q_slope():
  from sans_pilot.diagnostics import baseline_models

  assert baseline_models({"low_q_log_slope": -0.05, "peak": None})[0] == "sphere"
  assert baseline_models({"low_q_log_slope": -1.0, "peak": None})[0] == "cylinder"
  assert baseline_models({"low_q_log_slope": -2.0, "peak": None})[0] == "lamellar"
  assert baseline_models({"low_q_log_slope": -3.8, "peak": None}) == ["power_law"]
  assert baseline_models({"low_q_log_slope": -0.1, "peak": {"q_peak": 0.05}})[0] == (
    "sphere@hardsphere"
  )
  assert baseline_models({"low_q_log_slope": None}) == []
