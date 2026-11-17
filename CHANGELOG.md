# Changelog

## 0.11.0

### Changed

- **IQuana 0.6 replaces sans-fitter 0.3.0** (same project, renamed). `chisq`
  is now the total chi-squared; the reduced value is reported as
  `reduced_chisq`. Fit results also carry `dof`, `n_points`, `n_free`,
  parameters on bounds and strongly correlated pairs.
- `fit-sans-model` accepts optional `resolution` and `weighting` settings and
  can write `analysis.json` and a self-contained `fit_report.html`.

### Added

- **Tool profiles** (`SANS_PILOT_TOOLS=full|guided|all`). `full` is the
  default and unchanged. `guided` exposes small, flat, text-only session tools
  intended for local language models.
- **Guided analysis** (`run-sans-guided-analysis`): loads the data, ranks
  models with Guess from the Sassafras curve library, fits the top
  suggestions plus simple baseline shapes from several starting points, and
  ranks the fits by BIC. Guess is an optional extra (`sans-pilot[guess]`).
- **Series analysis** (`run-sans-series-analysis`): fits each candidate
  jointly to several curves of the same particles (contrast, concentration
  or instrument series) with IQuana's `MultiFitter`, sharing the structure
  and fitting the particle SLD from contrast variation.
- **Text diagnostics** for models that cannot read plots: a verdict, issues
  (bounds, correlations, undetermined or poorly determined parameters,
  systematic residuals located by Q range) and suggested next actions.
- **Reports**: `report.html` (interactive figures, parameter tables,
  diagnostics, candidate comparison) and `report.md` for every analysis.
- **`sans-pilot-agent`**: a command-line runner that executes the procedure
  for one file or a series and optionally lets a model on any
  OpenAI-compatible endpoint (Ollama, LM Studio, ...) refine and interpret the
  result within a fixed step budget. It always writes the deterministic
  report, also when the endpoint is unreachable or the model misbehaves.

### Safeguards

- Fixing a parameter at the value the last fit gave it (exactly or within its
  uncertainty) is flagged: the next fit's uncertainties are marked as
  conditional and cannot be rated `good`. Parameters the data does not
  constrain at all may be fixed freely.
- Parameters that only rescale the intensity (for example `volfraction`) are
  not fitted together with `scale`.
- Changes made after a fit and not refitted are listed; reports show the last
  completed fit.
- Each candidate fit has an iteration budget and a wall-clock limit.
- The language model cannot change the Q range: dropping low-Q points let a
  wrong model win. Diagnostics no longer suggest trimming data, and reports
  state when only part of the data was fitted.
- Without known SLDs, a fit that needs more intensity than the scale limit
  allows fits the particle contrasts instead (scale held at the limit), so
  the correct model is no longer lost to placeholder SLDs. Restarts are
  compared with one fitted size distribution, like the kept fit.
- Switching to a candidate that fits much worse is warned about, and reports
  flag such a switch and interpretations that describe an earlier model.
- Fixed polydispersity attached by the curve library is removed when the fit
  is better without it.
- Warnings from discarded trial fits no longer appear in results.
- Where `scale` is a volume fraction (form factors with volume and SLD
  parameters), guided and series fits keep it at or below 1
  (`SANS_PILOT_SCALE_MAX`) and explain a fit that reaches the limit.
  Polymer, fractal, peak and other shape-independent models are exempt.

### Known limitations

- Joint series fits and multi-start restarts are monodisperse.
- `onion` and `core_multi_shell` cannot currently be set up by IQuana
  ("Unused parameters in call") and are skipped; their single-shell
  counterparts are fitted instead.
- Guided sessions support atomic models with an optional structure factor;
  composite models still go through `fit-sans-model`.
- Language-model refinement is not available for series analyses yet; a model
  only writes the interpretation.
