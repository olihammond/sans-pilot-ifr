"""Joint analysis of a series of curves from the same particles.

A contrast series, a concentration series or one sample on several instrument
settings share most of their physics.  Fitting the curves one at a time throws
that constraint away: weak-contrast curves cannot identify the shape and
different curves settle on different models.  ``series_analysis`` instead

1. picks the curve with the best signal as the reference and asks Guess for
   candidate models there (plus simple baselines and any named models);
2. fits every candidate to all curves at once with IQuana's ``MultiFitter``:
   structural parameters are shared, each curve keeps its own background,
   and per-file known values (for example each solvent SLD) are held fixed;
3. tries several starting points per candidate and keeps the best;
4. ranks the joint fits by BIC and reports the winner.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from iquana import IQuana, MultiFitter

from sans_pilot.analyses._fitting_helpers import filter_actionable_warnings
from sans_pilot.artifacts import safe_path_component
from sans_pilot.datasets import inspect_data, load_data_source
from sans_pilot.diagnostics import (
  REDUCED_CHISQ_ACCEPTABLE,
  REDUCED_CHISQ_GOOD,
  active_arrays,
  baseline_models,
  describe_features,
  scale_limit,
)
from sans_pilot.guess import CAUTION, guess_status, suggest_models
from sans_pilot.procedure import (
  CANDIDATE_METHOD_SECONDS,
  DEFAULT_FIT_STEPS,
  EXTRA_STARTS,
  MAX_BASELINES,
  MAX_CANDIDATES,
  _configure_candidate,
  _expression,
  _solve_scale_background,
  _split_models,
  alternative_starts,
  parse_fixed_parameters,
)
from sans_pilot.runtime import render_runtime
from sans_pilot.schemas import UploadDataSource
from sans_pilot.sessions import SessionError, _number, quiet_science, time_limit

MAX_SERIES_FILES = 12
SERIES_FIT_SECONDS = 30.0
SERIES_RESTART_SECONDS = 60.0
STRONG_CORRELATION = 0.95
RANKING_NOTE = (
  "Ranked by BIC = joint chi-squared + n_free * ln(total points); lower is "
  "better. Candidates within 2 of the best BIC are marked close."
)


def _plain(value: Any) -> Any:
  """Convert numpy scalars and other stragglers to plain JSON types."""
  import json

  def fallback(item: Any) -> Any:
    return item.item() if hasattr(item, "item") else str(item)

  return json.loads(json.dumps(value, default=fallback))


def parse_per_file(text: str | None, count: int) -> dict[str, list[float]]:
  """Parse 'sld_solvent=-0.56/1.17/6.36' into one value per file, in file order."""
  result: dict[str, list[float]] = {}
  for item in (text or "").replace(";", ",").split(","):
    if not item.strip():
      continue
    name, separator, values = item.partition("=")
    try:
      if not separator or not name.strip():
        raise ValueError
      numbers = [float(value) for value in values.split("/")]
    except ValueError:
      raise SessionError(
        f"per_file_parameters entry '{item.strip()}' must look like "
        "name=v1/v2/..., one value per file, for example "
        "'sld_solvent=-0.56/2.9/6.36'."
      ) from None
    if len(numbers) != count:
      raise SessionError(
        f"per_file_parameters gives {len(numbers)} values for {name.strip()} "
        f"but there are {count} files. Give one value per file, in file order."
      )
    result[name.strip()] = numbers
  return result


def _signal_to_noise(data: Any) -> float:
  arrays = active_arrays(data)
  if "di" not in arrays:
    return 0.0
  excess = arrays["i"] - float(np.median(arrays["i"][np.argsort(arrays["q"])][-5:]))
  good = arrays["di"] > 0
  return (
    float(np.median(np.abs(excess[good]) / arrays["di"][good])) if good.any() else 0.0
  )


def _match_point(per_file: dict[str, list[float]], datasets: list[Any]) -> float | None:
  """Estimate the particle SLD as the solvent SLD of the weakest curve."""
  solvent = next(
    (values for name, values in per_file.items() if name.startswith("sld")), None
  )
  if solvent is None:
    return None
  signal = [_signal_to_noise(data) for data in datasets]
  return float(solvent[int(np.argmin(signal))])


def _build(
  datasets: list[Any],
  expression: str,
  start: dict[str, Any] | None,
  fixed: dict[str, float],
  per_file: dict[str, list[float]],
  trial: dict[str, float],
  particle_sld: float | None,
) -> tuple[MultiFitter, list[str], list[str]]:
  """Configure one joint fit; returns (fitter, shared names, free SLD names)."""
  model, _, structure_factor = expression.partition("@")
  plain = {**start, "polydispersity": {}} if start is not None else None
  fitters: list[IQuana] = []
  free_slds: list[str] = []
  for index, data in enumerate(datasets):
    fitter = IQuana()
    fitter.set_data(data)
    fitter.set_model(model)
    if structure_factor:
      fitter.set_structure_factor(structure_factor)
    if "dq" not in active_arrays(data):
      fitter.set_resolution("none")
    known = {**fixed, **{name: values[index] for name, values in per_file.items()}}
    _configure_candidate(fitter, expression, plain, known)
    for name, value in trial.items():
      if name in fitter.params:
        fitter.set_param(name, value=value)
    # With the solvent SLD known for every curve, contrast variation
    # identifies the remaining particle SLDs: free them, shared across curves.
    if particle_sld is not None:
      free_slds = [
        name
        for name in fitter.params
        if name.startswith("sld") and name not in known and "solvent" not in name
      ]
      for name in free_slds:
        fitter.set_param(
          name, value=trial.get(name, particle_sld), min=-1.0, max=10.0, vary=True
        )
    _solve_scale_background(fitter, keep_bounds=False)
    fitters.append(fitter)

  cap = scale_limit(expression)
  ceiling = math.inf if cap is None else cap
  share_scale = bool(per_file) and "scale" in fitters[0].params
  if share_scale:
    scales = [float(fitter.params["scale"]["value"]) for fitter in fitters]
    scale = float(np.median([value for value in scales if value > 0] or [1.0]))
    scale = min(scale, ceiling)
    for fitter in fitters:
      fitter.set_param(
        "scale",
        value=scale,
        min=scale / 100.0,
        max=min(scale * 100.0, ceiling),
        vary=True,
      )
      _resolve_background(fitter)
  else:
    for fitter in fitters:
      value = float(fitter.params["scale"]["value"])
      if value > 0:
        value = min(value, ceiling)
        fitter.set_param(
          "scale",
          value=value,
          min=value / 100.0,
          max=min(value * 100.0, ceiling),
          vary=True,
        )
  for fitter in fitters:
    value = float(fitter.params["background"]["value"])
    arrays = active_arrays(fitter.data)
    span = max(abs(value), float(np.median(np.abs(arrays["i"]))) * 1e-2, 1e-6)
    fitter.set_param(
      "background", value=value, min=value - 5 * span, max=value + 5 * span, vary=True
    )

  joint = MultiFitter()
  for index, fitter in enumerate(fitters):
    joint.add_fitter(f"d{index + 1}", fitter)
  shared = [
    name
    for name, row in fitters[0].params.items()
    if row.get("vary") and name != "background" and (name != "scale" or share_scale)
  ]
  if shared:
    joint.share(*shared)
  return joint, shared, free_slds


def _resolve_background(fitter: IQuana) -> None:
  """Re-estimate the background alone, with the (shared) scale held."""
  arrays = active_arrays(fitter.data)
  current = float(fitter.params["background"]["value"])
  theory = np.asarray(fitter.calculate(), dtype=float)
  if theory.shape == arrays["index"].shape:
    theory = theory[arrays["index"]]
  if theory.shape != arrays["q"].shape or "di" not in arrays:
    return
  weight = 1.0 / arrays["di"] ** 2
  finite = np.isfinite(theory) & np.isfinite(weight)
  if not finite.any():
    return
  shift = float(
    np.sum(weight[finite] * (arrays["i"][finite] - theory[finite]))
    / np.sum(weight[finite])
  )
  if math.isfinite(shift):
    row = fitter.params["background"]
    value = current + shift
    low, high = row.get("min"), row.get("max")
    fitter.set_param(
      "background",
      value=value,
      min=min(float(low), value) if low is not None else None,
      max=max(float(high), value) if high is not None else None,
    )


def _strong_correlations(result: dict[str, Any]) -> list[str]:
  labels, corr = result.get("cov_labels") or [], result.get("corr")
  if corr is None or not labels:
    return []
  matrix = np.asarray(corr, dtype=float)
  pairs = []
  for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
      if math.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= STRONG_CORRELATION:
        pairs.append(f"{labels[i]} & {labels[j]} ({matrix[i, j]:+.2f})")
  return pairs


def _fit_joint(
  datasets: list[Any],
  expression: str,
  start: dict[str, Any] | None,
  fixed: dict[str, float],
  per_file: dict[str, list[float]],
  particle_sld: float | None,
) -> dict[str, Any]:
  """Fit one model to all curves from several starts; keep the best."""
  best: dict[str, Any] | None = None
  notes: list[str] = []
  attempts = 0
  deadline = time.perf_counter() + SERIES_RESTART_SECONDS
  trials: list[dict[str, float]] = [{}]
  for trial in trials:
    if attempts and time.perf_counter() > deadline:
      break
    attempts += 1
    try:
      joint, shared, free_slds = _build(
        datasets, expression, start, fixed, per_file, trial, particle_sld
      )
      if attempts == 1:
        # Other starting points for the shared sizes (SLDs are excluded: their
        # lower bound is not positive).
        trials += alternative_starts(joint["d1"].params, expression, EXTRA_STARTS)

      timer = time.perf_counter()
      joint.calculate()
      per_pass = max(time.perf_counter() - timer, 1e-4)
      free = len(shared) + len(datasets)
      steps = int(SERIES_FIT_SECONDS / (per_pass * (free + 2)))
      with time_limit(CANDIDATE_METHOD_SECONDS * 2):
        fitted = joint.fit(
          method="lm", steps=max(2, min(steps, DEFAULT_FIT_STEPS))
        ).to_dict()
    except Exception as exc:
      notes.append(f"{expression}: start {attempts} failed: {str(exc)[:160]}")
      continue
    chisq = float(fitted.get("chisq", math.inf))
    if math.isfinite(chisq) and (best is None or chisq < best["chisq"] * (1 - 1e-4)):
      best = {
        "chisq": chisq,
        "joint": joint,
        "result": fitted,
        "shared": shared,
        "free_slds": free_slds,
        "start_number": attempts,
      }
  if best is None:
    raise RuntimeError("; ".join(notes) or "no joint fit succeeded")
  best["starts_tried"] = attempts
  best["notes"] = notes
  return best


def _verdict(reduced: float | None, issues: list[str], per_dataset: list[dict]) -> str:
  if reduced is None or not math.isfinite(reduced):
    return "unknown"
  worst = max((item.get("reduced_chisq") or 0.0) for item in per_dataset)
  if reduced <= REDUCED_CHISQ_GOOD and worst <= REDUCED_CHISQ_ACCEPTABLE:
    return "good"
  if reduced <= REDUCED_CHISQ_ACCEPTABLE:
    return "acceptable"
  return "poor"


def _summarize(
  best: dict[str, Any],
  labels: list[str],
  per_file: dict[str, list[float]],
) -> dict[str, Any]:
  result = best["result"]
  parameters = result.get("parameters") or {}
  shared: dict[str, Any] = {}
  for name in best["shared"]:
    entry = parameters.get(f"d1.{name}") or {}
    shared[name] = {
      "value": _number(entry.get("value")),
      "stderr": _number(entry.get("stderr")),
    }
  per_dataset = []
  for index, label in enumerate(labels):
    key = f"d{index + 1}"
    info = (result.get("datasets") or {}).get(key) or {}
    own = {
      name: _number((parameters.get(f"{key}.{name}") or {}).get("value"))
      for name in ("background", "scale")
      if name not in best["shared"] and f"{key}.{name}" in parameters
    }
    points = info.get("n_points") or 0
    per_dataset.append(
      {
        "file": label,
        "points": points,
        "chisq": _number(info.get("chisq")),
        "reduced_chisq": _number(info["chisq"] / points) if points else None,
        **own,
        **{
          f"{name}_stderr": _number(
            (parameters.get(f"{key}.{name}") or {}).get("stderr")
          )
          for name in own
        },
        "known": {name: values[index] for name, values in per_file.items()},
      }
    )
  n_points, n_free = result.get("n_points"), result.get("n_free")
  chisq = result.get("chisq")
  issues: list[str] = []
  on_bounds = result.get("on_bounds") or []
  if on_bounds:
    issues.append(
      "Parameters resting on a bound: "
      + ", ".join(
        " ".join(str(part) for part in item)
        if isinstance(item, (list, tuple))
        else str(item)
        for item in on_bounds
      )
      + "."
    )
  correlated = _strong_correlations(result)
  if correlated:
    issues.append(
      "Strongly correlated parameters: "
      + "; ".join(correlated[:4])
      + (f"; and {len(correlated) - 4} more" if len(correlated) > 4 else "")
      + "."
    )
  uneven = [
    item["file"]
    for item in per_dataset
    if (item.get("reduced_chisq") or 0) > REDUCED_CHISQ_ACCEPTABLE
  ]
  if uneven:
    issues.append(
      "The joint model fits these curves poorly: " + ", ".join(uneven) + "."
    )
  reduced = result.get("reduced_chisq")
  return {
    "statistics": {
      "reduced_chisq": _number(reduced),
      "chisq": _number(chisq),
      "dof": result.get("dof"),
      "n_points": n_points,
      "n_free": n_free,
      "n_datasets": result.get("n_datasets"),
      "bic": (
        _number(chisq + n_free * math.log(max(n_points, 1)))
        if isinstance(chisq, (int, float)) and n_free and n_points
        else None
      ),
    },
    "shared": shared,
    "free_slds": best["free_slds"],
    "datasets": per_dataset,
    "issues": issues,
    "verdict": _verdict(reduced, issues, per_dataset),
    "starts_tried": best["starts_tried"],
    "best_start": best["start_number"],
  }


def series_analysis(
  files: list[str],
  user_id: str | None,
  per_file_parameters: str | None,
  fixed_parameters: str | None,
  top: int,
  extra_models: str | None,
  output_dir: Path,
  include_plot: bool = True,
) -> dict[str, Any]:
  """Run the fixed procedure jointly over several curves of the same particles."""
  names = [name.strip() for name in files if name.strip()]
  if len(names) < 2:
    raise SessionError("A series needs at least two files.")
  if len(names) > MAX_SERIES_FILES:
    raise SessionError(f"A series can hold at most {MAX_SERIES_FILES} files.")
  per_file = parse_per_file(per_file_parameters, len(names))
  fixed = parse_fixed_parameters(fixed_parameters)
  extras = _split_models(extra_models)
  top = max(1, min(int(top), MAX_CANDIDATES))
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  warnings_out: list[str] = []
  datasets: list[Any] = []
  labels: list[str] = []
  with quiet_science(warnings_out):
    for name in names:
      loaded = load_data_source(
        UploadDataSource(kind="upload", file=name), user_id=user_id
      )
      if not inspect_data(loaded.data)["has_intensity_errors"]:
        raise SessionError(
          f"'{name}' has no intensity uncertainties; a joint fit needs dI to "
          "weight the curves against each other."
        )
      datasets.append(loaded.data)
      labels.append(str(loaded.provenance.get("file_name") or name))

  signal = [_signal_to_noise(data) for data in datasets]
  reference = int(np.argmax(signal))
  features = describe_features(datasets[reference])
  particle_sld = _match_point(per_file, datasets)

  plan: list[dict[str, Any]] = []
  guess_note: str | None = None
  status = guess_status()
  if status["available"]:
    baselines = _split_models(",".join(baseline_models(features)))[:MAX_BASELINES]
    also = list(dict.fromkeys([*extras, *baselines]))
    with quiet_science(warnings_out):
      guess = suggest_models(
        datasets[reference], top=max(top, 5), starting_values=top, also=also
      )
    for item in guess["suggestions"]:
      if len(plan) >= top:
        break
      if item["loadable"] and "start" in item:
        plan.append(
          {
            "model": item["expression"],
            "start": item["start"],
            "origin": "curve library",
          }
        )
    for item in guess.get("requested") or []:
      if any(item["expression"] == entry["model"] for entry in plan):
        continue
      pair = (item["model"], item["structure_factor"])
      plan.append(
        {
          "model": item["expression"],
          "start": item.get("start"),
          "origin": "requested" if pair in extras else "simple baseline",
        }
      )
  else:
    guess_note = status["reason"]
    plan = [
      {"model": _expression(*pair), "start": None, "origin": "requested"}
      for pair in extras
    ]
  if not plan:
    raise SessionError(
      (guess_note or "No candidate model was found.")
      + " Name models to fit with extra_models, for example 'sphere,ellipsoid'."
    )

  candidates: list[dict[str, Any]] = []
  fits: dict[str, dict[str, Any]] = {}
  for entry in plan[: MAX_CANDIDATES + MAX_BASELINES]:
    started = time.perf_counter()
    candidate: dict[str, Any] = {"model": entry["model"], "origin": entry["origin"]}
    try:
      with quiet_science(warnings_out):
        best = _fit_joint(
          datasets, entry["model"], entry["start"], fixed, per_file, particle_sld
        )
        summary = _summarize(best, labels, per_file)
    except Exception as exc:
      candidate["error"] = f"joint fit failed: {str(exc)[:200]}"
      candidates.append(candidate)
      continue
    fits[entry["model"]] = {**best, "summary": summary}
    warnings_out += best["notes"]
    candidate.update(
      {
        "reduced_chisq": summary["statistics"]["reduced_chisq"],
        "bic": summary["statistics"]["bic"],
        "n_free": summary["statistics"]["n_free"],
        "verdict": summary["verdict"],
        "shared": {name: row["value"] for name, row in summary["shared"].items()},
        "worst_curve_reduced_chisq": max(
          (item.get("reduced_chisq") or 0.0) for item in summary["datasets"]
        ),
        "starts_tried": summary["starts_tried"],
        "seconds": round(time.perf_counter() - started, 1),
      }
    )
    candidates.append(candidate)

  def score(item: dict[str, Any]) -> float:
    value = item.get("bic")
    return float(value) if isinstance(value, (int, float)) else math.inf

  candidates.sort(key=score)
  for rank, item in enumerate(candidates, start=1):
    item["rank"] = rank
    if rank > 1 and math.isfinite(score(item)):
      item["close_to_best"] = bool(score(item) - score(candidates[0]) < 2.0)
  succeeded = [item for item in candidates if "error" not in item]
  if not succeeded:
    raise SessionError(
      "Every joint fit failed: "
      + "; ".join(str(item.get("error")) for item in candidates)
    )

  selected = succeeded[0]["model"]
  chosen = fits[selected]
  summary = {
    "analysis": "series",
    "files": labels,
    "reference_file": labels[reference],
    "per_file_parameters": per_file,
    "fixed_parameters": fixed,
    "features_of_reference": features,
    "candidates": candidates,
    "ranking": RANKING_NOTE,
    "selected_model": selected,
    "selected_fit": chosen["summary"],
    "caution": CAUTION,
    "warnings": filter_actionable_warnings(warnings_out)[:10],
  }
  if guess_note:
    summary["guess_unavailable"] = guess_note
  if not per_file:
    summary["note"] = (
      "No per-file known values were given, so each curve has its own scale "
      "and only the structural parameters are shared."
    )

  artifacts = _write_outputs(output_dir, summary, chosen, labels, include_plot)
  mime_types = {
    ".md": "text/markdown",
    ".html": "text/html",
    ".json": "application/json",
    ".png": "image/png",
    ".csv": "text/csv",
  }
  summary["artifacts"] = [
    {"name": name, "mime_type": mime_types.get(path.suffix, "text/plain")}
    for name, path in artifacts.items()
  ]
  return {"summary": _plain(summary), "artifacts": artifacts}


def _series_figure(joint: MultiFitter, labels: list[str]) -> Any:
  """Draw every curve with its joint-fit line on one plot, residuals below."""
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots

  palette = [
    "#3b6fb6",
    "#c2571a",
    "#2e8b57",
    "#9b3fa3",
    "#b8860b",
    "#c0392b",
    "#1f9aa8",
    "#6b6b6b",
    "#7a5c3e",
    "#d1478f",
    "#4f7f1f",
    "#34495e",
  ]
  theory = joint.calculate()
  figure = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04
  )
  for index, label in enumerate(labels):
    key = f"d{index + 1}"
    arrays = active_arrays(joint[key].data)
    model = np.asarray(theory[key], dtype=float)
    if model.shape == arrays["index"].shape:
      model = model[arrays["index"]]
    order = np.argsort(arrays["q"])
    q, intensity, model = arrays["q"][order], arrays["i"][order], model[order]
    sigma = arrays["di"][order] if "di" in arrays else None
    colour = palette[index % len(palette)]
    name = Path(label).stem
    figure.add_trace(
      go.Scatter(
        x=q,
        y=intensity,
        mode="markers",
        name=name,
        legendgroup=name,
        marker={"size": 4, "color": colour, "opacity": 0.7},
        error_y=(
          {"type": "data", "array": sigma, "width": 0, "thickness": 0.8}
          if sigma is not None
          else None
        ),
      ),
      row=1,
      col=1,
    )
    figure.add_trace(
      go.Scatter(
        x=q,
        y=model,
        mode="lines",
        name=f"{name} fit",
        legendgroup=name,
        showlegend=False,
        line={"width": 2, "color": colour},
      ),
      row=1,
      col=1,
    )
    if sigma is not None:
      figure.add_trace(
        go.Scatter(
          x=q,
          y=(intensity - model) / sigma,
          mode="markers",
          legendgroup=name,
          showlegend=False,
          marker={"size": 3, "color": colour, "opacity": 0.7},
        ),
        row=2,
        col=1,
      )
  figure.add_shape(
    type="line",
    xref="x2 domain",
    yref="y2",
    x0=0,
    x1=1,
    y0=0,
    y1=0,
    line={"width": 1, "color": "#888888"},
  )
  figure.update_xaxes(type="log", row=1, col=1)
  figure.update_xaxes(type="log", title_text="Q (Å⁻¹)", row=2, col=1)
  figure.update_yaxes(type="log", title_text="I(Q)", row=1, col=1)
  figure.update_yaxes(title_text="Residual (σ)", row=2, col=1)
  figure.update_layout(
    height=620, template="plotly_white", legend={"orientation": "h", "y": -0.14}
  )
  return figure


def _write_outputs(
  output_dir: Path,
  summary: dict[str, Any],
  chosen: dict[str, Any],
  labels: list[str],
  include_plot: bool,
) -> dict[str, Path]:
  """Write the joint report, per-curve analysis files and figures."""
  import json

  from sans_pilot.report_html import build_series_html, series_markdown

  artifacts: dict[str, Path] = {}
  joint: MultiFitter = chosen["joint"]

  result_path = output_dir / "series_result.json"
  result_path.write_text(
    json.dumps(_plain(summary), indent=2, ensure_ascii=False),
    encoding="utf-8",
  )
  artifacts[result_path.name] = result_path

  with quiet_science():
    report_text = joint.get_fit_report().to_markdown()
  markdown_path = output_dir / "report.md"
  markdown_path.write_text(series_markdown(summary, report_text), encoding="utf-8")
  artifacts[markdown_path.name] = markdown_path

  try:
    with quiet_science():
      figure = _series_figure(joint, labels)
  except Exception:
    figure = None

  html_path = output_dir / "report.html"
  html_path.write_text(build_series_html(summary, figure), encoding="utf-8")
  artifacts[html_path.name] = html_path

  if include_plot and figure is not None:
    plot_path = output_dir / "fit_plot.png"
    try:
      with render_runtime():
        figure.write_image(plot_path)
      artifacts[plot_path.name] = plot_path
    except Exception:
      pass

  # One IQuana analysis file per curve, holding the joint solution.
  parameters = chosen["result"].get("parameters") or {}
  model, _, structure_factor = summary["selected_model"].partition("@")
  for index, label in enumerate(labels):
    key = f"d{index + 1}"
    try:
      with quiet_science():
        fitter = IQuana()
        fitter.set_data(joint[key].data)
        fitter.set_model(model)
        if structure_factor:
          fitter.set_structure_factor(structure_factor)
        for name in fitter.params:
          entry = parameters.get(f"{key}.{name}")
          if entry is None or not isinstance(entry.get("value"), (int, float)):
            continue
          value = float(entry["value"])
          row = fitter.params[name]
          low, high = row.get("min"), row.get("max")
          fitter.set_param(
            name,
            value=value,
            min=min(float(low), value) if low is not None else None,
            max=max(float(high), value) if high is not None else None,
            vary=entry.get("status") in ("free", "shared"),
          )
        path = output_dir / f"analysis_{safe_path_component(Path(label).stem)}.json"
        fitter.save_analysis(str(path), include_result=False)
      artifacts[path.name] = path
    except Exception:
      continue
  return artifacts
