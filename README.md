# SANS-pilot

MCP server for reduced one-dimensional Small-Angle Neutron Scattering data
analysis, powered by [IQuana](https://github.com/SasView/IQuana) (formerly SANS-fitter).

## Scientific tools

| Tool                          | Purpose                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| `describe-sans-capabilities`  | Describe supported workflows and scientific limitations            |
| `list-supported-sans-formats` | List accepted reduced 1D community formats                         |
| `list-uploaded-sans-files`    | List current-user uploads without exposing file contents           |
| `inspect-sans-data`           | Inspect an upload, example, or simulation                          |
| `plot-sans-data`              | Plot measured data without a model, fit curve, or residuals        |
| `process-sans-data`           | Apply arithmetic/Q selection and return processed CSV data         |
| `list-sans-models`            | List exact sasmodels names                                         |
| `get-sans-model-parameters`   | Discover an exact atomic, interacting, or composite model contract |
| `list-structure-factors`      | List supported inter-particle interaction models                   |
| `get-polydispersity-options`  | List size-distribution types and defaults                          |
| `fit-sans-model`              | Run optimization or Bayesian fitting through IQuana           |
| `scan-sans-dmax`              | Explore Dmax using Rg, I(0), fit-quality, and positivity trends    |
| `invert-sans-pr`              | Recover the model-free real-space pair distribution P(r)           |
| `list-sans-examples`          | Discover curated measured and simulated datasets                   |
| `inspect-sans-example`        | Inspect an example and its suggested configuration                 |
| `simulate-sans-data`          | Generate synthetic data with known truth                           |
| `simulate-sans-pair`          | Generate a matched sample/background pair                          |
| `read-sans-artifact`          | Reopen a user-scoped image or chunked CSV/text artifact            |

## Tool profiles

`SANS_PILOT_TOOLS` selects which tools the server exposes:

| Profile          | Tools                                                                 | Intended for                         |
| ---------------- | --------------------------------------------------------------------- | ------------------------------------ |
| `full` (default) | The typed scientific tools in the table above                         | Hosted models, complex requests      |
| `guided`         | The small flat session tools below, plus file listing and artifacts   | Local models with small context      |
| `all`            | Both                                                                  | Hosted models that also want Guess   |

The `guided` tool definitions total about 7,000 characters with no nested unions
(the `full` profile is about 38,000), every result is text-only JSON with a
`next_steps` list, and fit quality is reported as a verdict with issues and
suggested actions rather than as a plot.

## Guided analysis (session tools)

| Tool                       | Purpose                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| `run-sans-guided-analysis` | The fixed procedure: load, suggest models, fit the top ones, rank    |
| `run-sans-series-analysis` | The same procedure for several curves at once, fitted jointly        |
| `start-sans-analysis`      | Load one file or example into a session and describe the curve shape |
| `suggest-sans-models`      | Rank models against the Sassafras curve library ("Guess")            |
| `fit-sans-suggestions`     | Fit the top suggestions, simple baselines and any `extra_models`     |
| `use-sans-candidate`       | Switch to another fitted candidate                                   |
| `set-sans-model`           | Set the model by exact name, with an optional structure factor       |
| `set-sans-parameter`       | Change one parameter's value, bounds or vary flag                    |
| `set-sans-polydispersity`  | Give one size parameter a distribution                               |
| `set-sans-q-range`         | Restrict or reset the fitted Q range                                 |
| `run-sans-fit`             | Fit and return statistics, a verdict, issues and suggestions         |
| `get-sans-analysis`        | Show the session's model, parameters and last fit                    |
| `export-sans-analysis`     | Write `report.html`, `report.md`, `analysis.json` and the plot       |

A session is a directory under `SANS_PILOT_RUNS_DIR/sessions/<user>/` holding
the data and an IQuana analysis file (`IQuana.save_analysis`). Each call
reloads it, changes one thing and saves it, so sessions survive restarts and
`analysis.json` reopens directly in IQuana. Sessions are atomic models with an
optional structure factor; use `fit-sans-model` for composite models.

### The fixed procedure

`run-sans-guided-analysis` makes no model-side decisions:

1. Load the data and describe its shape in numbers (low-Q slope, Guinier Rg,
   background level, interference peak).
2. Rank models with **Guess** from [Sassafras](https://github.com/SasView/sassafras):
   the curve is matched, on shape only, against about 1.3 million simulated
   curves and the closest curves supply starting values.
3. Fit the top `top` suggestions (default 3) with IQuana from those starting
   values, plus one or two plain shapes chosen from the low-Q slope
   (`include_baselines`, default on) and any `extra_models`.
4. Rank the fits by BIC, which penalises extra free parameters, mark
   candidates the data cannot separate as `close_to_best`, and keep the best
   one in the session for refinement.

Automatic fits vary the structural parameters, `scale` and `background`
within a factor of four of the starting value; SLDs, solvent conditions and
orientation angles stay fixed. Each candidate's iterations are capped from the
measured cost of one curve (a budget of about 20 s per method), so a slow model
cannot stall the procedure; capped candidates are flagged.

The library tends to rank flexible many-parameter models first because they
can imitate simpler shapes; the baselines and the BIC ranking are there to
counter that. A low chi-squared means a model *can* describe the curve, not
that it is the right description of the sample.

Guess is optional:

```bash
pip install "sans-pilot[guess]"
python -m sassafras.library fetch      # curve library, ~290 MB, once
```

Without it, `fit-sans-suggestions` asks for `extra_models` and fits those from
the model defaults. `SANS_PILOT_GUESS=off` disables it explicitly.

`fixed_parameters` (for example `"sld=0.4,sld_solvent=6.36"`) supplies
independently known values. They replace the library's values in every
candidate that has those parameters, stay fixed, and the scale is re-solved
for the new contrast. Without them the SLDs are the curve library's, so
`scale` and volume fractions carry no physical meaning.

Candidates already fitted in a session are kept: a later call with
`extra_models` fits only the new ones and re-ranks. Changing the Q range or
the fixed parameters refits everything.

### Several starting points

A single start can settle in a local minimum, for example the prolate
solution for an oblate ellipsoid. Each candidate that is cheap to evaluate is
refitted from up to six other starting points inside the same bounds (the
sizes mirrored about their geometric mean, then seeded log-uniform draws) and
the best fit is kept. The library's polydispersity comes from a coarse scan
and is held fixed, so these restarts are monodisperse; candidates report
`starts_tried` and `improved_by_restart`.

### Series of curves

Curves from the same particles (a contrast series, a concentration series,
several instrument settings) should be analysed together.
`run-sans-series-analysis` takes all the files at once:

```json
{
  "files": "ZnO_000.dat,ZnO_050.dat,ZnO_100.dat",
  "per_file_parameters": "sld_solvent=-0.56/2.90/6.36",
  "fixed_parameters": "sld=4.77"
}
```

It takes candidate models from the curve with the best signal (Guess, simple
baselines, `extra_models`) and fits each one jointly to every curve with
IQuana's `MultiFitter`, from several starting points:

- structural parameters are shared between curves;
- each curve keeps its own background;
- `per_file_parameters` are held at the given value for each file;
- when per-file values are given, the scale is shared and any particle SLD not
  supplied in `fixed_parameters` is fitted from the contrast variation;
- without per-file values each curve keeps its own scale and only the
  structure is shared.

The joint fits are ranked by BIC. The output is `report.html` with all curves
and their joint fit on one figure, `report.md`, `series_result.json`, and one
`analysis_<file>.json` per curve holding the joint solution. Joint fits are
monodisperse, need dI in every file, and take at most 12 files.

### Diagnostics

The verdict is a heuristic computed from IQuana's fit result: `good` (reduced
chi-squared <= 2 and no systematic residuals), `acceptable` (<= 5), `poor`,
or `uncalibrated` when the data has no dI. Issues cover parameters resting on
a bound, |correlation| >= 0.95, standard errors above half the value, and
systematic residuals (runs test) located at low, intermediate or high Q. The
thresholds are returned by `get-sans-analysis` and printed in every report.

Fixing a parameter at the value the last fit gave it (`vary=false` with no new
value, or with a value inside that fit's own uncertainty) leaves chi-squared
unchanged while shrinking every other uncertainty. Only values declared
through `fixed_parameters` count as independently known.
The tools warn when this happens, the next fit is marked as having
*conditional* uncertainties and cannot be rated `good`, and the reports say so.
Settings changed after a fit and not refitted are listed as unfitted; reports
and `analysis.json` then show the last completed fit.

## Local LLM runner

`sans-pilot-agent` runs the fixed procedure for one file and optionally lets a
model on any OpenAI-compatible endpoint refine the result:

```bash
# Deterministic only: no language model involved
sans-pilot-agent data.csv

# Refined and interpreted by a local model (Ollama shown; LM Studio serves
# the same API on http://localhost:1234/v1)
sans-pilot-agent data.csv \
  --model <model-name> --base-url http://localhost:11434/v1 \
  --sample "SDS micelles in D2O, 2 wt%, charged" \
  --sld 0.4 --sld-solvent 6.36
```

The runner owns loading, the guided procedure and the export. The model is
given only the refinement tools and a fixed step budget (`--max-steps`,
default 12): it may switch candidate, fit a model the sample suggests, adjust
parameters and refit, then write an interpretation. Whatever the model does,
the output folder (`<data>_analysis/` by default) receives:

- `report.html` - the page to open: verdict, an interactive fit figure with
  residuals, fitted values with uncertainties, diagnostics, a figure
  overlaying every candidate on the data, and the model's interpretation. It
  is one self-contained file (about 5 MB, works offline);
- `report.md` - the same content as text: data summary, models tried, fit
  report, diagnostics, steps taken and, clearly marked as unverified, the
  model's interpretation;
- `analysis.json` - reloadable with `IQuana.load_analysis`;
- `analysis_guided.json` - the state before the model's turn, with a warning
  in the report if the refinement made the fit worse;
- `fit_plot.png`, `sasview_parameter_values.txt`, `guided_result.json` and
  `transcript.json`.

Malformed tool arguments, unknown tools and tool calls printed as text are
handled; an endpoint that rejects tool definitions falls back to
interpretation only, and an unreachable endpoint still yields the
deterministic report. `--sample` matters: it is the only knowledge of the
sample the model has. `--sld`, `--sld-solvent` and `--fixed "name=value,..."`
pass known values through as `fixed_parameters`. The report states plainly
when the model's turn left the fit unchanged, or only changed which
parameters were free. Tool definitions and the guided result need roughly
3,000 tokens of context before the conversation starts, so raise the context
length of runtimes that default lower.

Give several files to analyse them jointly as a series:

```bash
sans-pilot-agent ZnO_000.dat ZnO_050.dat ZnO_100.dat \
  --per-file "sld_solvent=-0.56,2.90,6.36" \
  --sample "ZnO nanoparticles, H2O/D2O contrast series"
```

`--per-file NAME=V1,V2,...` gives one known value per file in the order the
files are listed, and can be repeated for other parameters; `--fixed` gives
values common to all files. In series mode a `--model` only writes the
interpretation: there are no refinement tools for joint fits yet.

Any MCP client can use the same tools without the runner:

```bash
SANS_PILOT_TOOLS=guided UPLOAD_DIR=/path/to/data sans-pilot
```

## Data pipeline

All data tools use a typed `pipeline`. The primary source may be a user upload,
a bundled example, or a simulation:

```json
{ "primary": { "kind": "upload", "file": "stored-data-name.xml" } }
```

```json
{ "primary": { "kind": "example", "name": "protein" } }
```

```json
{
  "primary": {
    "kind": "simulation",
    "model": "sphere",
    "parameters": { "radius": 50 },
    "seed": 42
  }
}
```

Optional `auxiliary` sources and ordered `operations` support `add`,
`subtract`, `multiply`, and `divide` with another dataset or a scalar. Optional
`q_min` and `q_max` select the active scientific range. Arithmetic behavior,
including uncertainty, unit, and resolution propagation, is provided directly
by `IQuana`.

### Plot without fitting

```json
{
  "pipeline": {
    "primary": { "kind": "upload", "file": "simulated_sans_data.csv" }
  },
  "log_scale": true
}
```

`plot-sans-data` returns metadata, an inline MCP image for immediate display,
and a lazy resource URI for `sans_data_plot.png`; it never selects a model or
performs a fit.

## Model discovery and fitting

Call `get-sans-model-parameters` with the exact model specification you intend
to fit. It returns parameter names, defaults, bounds, polydispersity support,
component aliases, and links reported by the configured IQuana model.

### Atomic model fit

`fit-sans-model` accepts one `request` object:

```json
{
  "request": {
    "pipeline": {
      "primary": { "kind": "upload", "file": "simulated_sans_data.csv" },
      "q_min": 0.01,
      "q_max": 0.3
    },
    "model": { "kind": "atomic", "model": "cylinder" },
    "parameters": {
      "radius": { "value": 20, "min": 1, "max": 200, "vary": true },
      "length": { "value": 400, "min": 10, "max": 4000, "vary": true },
      "scale": { "value": 1, "min": 0, "max": 10, "vary": true },
      "background": { "value": 0.001, "min": 0, "max": 1, "vary": true }
    },
    "fit": { "mode": "optimization", "engine": "bumps", "method": "amoeba" }
  }
}
```

The selected engine and method are passed directly to `IQuana`. Fit
objectives, uncertainties, optimizer status, warnings, and errors retain their
`IQuana` meanings and are not reinterpreted by the MCP server.

### Fit statistics, resolution, and weighting

Since IQuana 0.4 `chisq` is the **total** chi-squared; the reduced value is
reported separately as `reduced_chisq` (sans-fitter 0.3 reported the reduced
value under `chisq`). The fit result also forwards `n_points`, `n_free`, `dof`,
`converged`, the optimizer `message`, `cov_source`, any parameters resting
`on_bounds`, and parameter pairs with |correlation| >= 0.95 as
`strong_correlations`.

Optional `resolution` and `weighting` are passed to `IQuana.set_resolution` and
`IQuana.set_weighting`:

```json
{
  "resolution": { "mode": "pinhole", "dq_over_q": 0.05 },
  "weighting": "dI"
}
```

When data carries no dQ column and no resolution is stated, IQuana warns that
the model is evaluated unsmeared; set `{"mode": "none"}` to state that choice.

`artifacts.include_analysis_json` writes `analysis.json` (reopen with
`IQuana.load_analysis`) and `artifacts.include_report_html` writes a
self-contained `fit_report.html`.

### Structure factor and polydispersity

An interacting atomic model puts the structure factor in the model object;
its parameters use the same `parameters` map:

```json
{
  "model": {
    "kind": "atomic",
    "model": "sphere",
    "structure_factor": "hardsphere",
    "radius_effective_mode": "link_radius"
  },
  "polydispersity": {
    "radius": {
      "pd_width": 0.1,
      "pd_type": "gaussian",
      "pd_n": 35,
      "pd_nsigma": 3,
      "vary": false
    }
  }
}
```

Supported structure factors are `hardsphere`, `hayter_msa`, `squarewell`, and
`stickyhardsphere`. Supported distributions are `gaussian`, `rectangle`,
`lognormal`, `schulz`, and `boltzmann`.

### Composite model

Composite models combine uniquely named components. `shared_parameters` and
`parameter_links` configure the equality mechanisms provided by IQuana.

```json
{
  "kind": "composite",
  "operation": "+",
  "components": [
    { "alias": "small", "model": "sphere" },
    { "alias": "long", "model": "cylinder" }
  ],
  "shared_parameters": ["sld", "sld_solvent"],
  "parameter_links": {}
}
```

Composite models and parameter links require BUMPS point-estimate fitting.
Additive component curves are included in the fit plot by default.

### Bayesian fit

```json
{
  "fit": {
    "mode": "bayesian",
    "samples": 5000,
    "burn": 200,
    "thin": 1,
    "pop": 10
  },
  "artifacts": {
    "posterior_plots": ["predictive", "pairs", "trace"],
    "include_posterior_chain": false
  }
}
```

Bayesian fitting uses the BUMPS DREAM integration provided by `IQuana`.
The compact response serializes the posterior summary and diagnostics returned
by `IQuana`; raw samples are not placed in model context.

Every fit returns compact structured JSON, inline MCP image content for plots,
and lazy MCP resource URIs for all generated files. When browser downloads are
configured, artifact metadata also includes a signed `download_url`. Atomic
fits include links for `sasview_parameter_values.txt` by default. CSV and text
artifact bytes are not included in the original tool response. `fit_results.csv`
and `posterior_chain.csv` are generated only when explicitly requested. MCP
resource URIs are opaque and user-scoped; non-image artifacts remain lazy so
their contents do not consume model context.

## Model-free P(r) inversion

`scan-sans-dmax` exposes IQuana's Dmax exploration. The scan returns
arrays for Rg, I(0), data chi-squared, oscillations, positivity, background,
and alpha plus `dmax_scan.png`. IQuana documents a good Dmax as showing an
Rg/I(0) plateau and a minimum in the data chi-squared.

`invert-sans-pr` supports automatic selection of basis terms and regularization
or a manual mode that requires both `n_terms` and `alpha`:

```json
{
  "request": {
    "pipeline": { "primary": { "kind": "example", "name": "protein" } },
    "d_max": 120,
    "selection": { "mode": "automatic" },
    "fit_background": false,
    "regularizer": "corrected",
    "include_pr_csv": false,
    "plot_log_scale": false
  }
}
```

The result includes Dmax, Rg, I(0), background treatment, data chi-squared,
effective degrees of freedom, oscillations, positivity fractions, matrix
condition/rank diagnostics, and whether uncertainties were fabricated. It
returns P(r) and I(Q)/residual plots; numerical P(r) CSV is opt-in. Pinhole dQ
is ignored and slit smearing is unsupported by IQuana 0.6 inversion.

Buffer-subtracted protein data normally uses `fit_background=false` because a
fitted flat background can absorb I(0) and bias Rg.

## Curated examples and simulation

`list-sans-examples` can filter by tag and returns descriptions, suggested
models and parameters, structure factors, polydispersity, and caveats.
`inspect-sans-example` loads live metadata. Measured examples have
`known_truth=null`; suggested values are starting points, not results.

`simulate-sans-data` and `simulate-sans-pair` pass simulation settings to
`IQuana`; set an explicit seed when reproducible output is required. They
return generation truth separately from any later fitted estimate. The pair
tool produces independent noise on one shared Q grid, suitable for subtraction
tests. Plots are returned by default and CSV files only when requested.

## Supported data scope

The server accepts reduced 1D columnar text, NIST/SasView ASCII, CanSAS XML,
NXcanSAS/HDF5, and Anton Paar PDH data. Containers are type-checked after
loading. 2D SANS and SESANS are rejected with actionable messages.

## Configuration

| Variable                          | Default                | Description                                                        |
| --------------------------------- | ---------------------- | ------------------------------------------------------------------ |
| `UPLOAD_DIR`                      | `/uploads`             | User-uploaded data directory                                       |
| `SANS_PILOT_RUNS_DIR`             | `/tmp/sans-pilot-runs` | Shared workspace root for artifacts and durable token manifests    |
| `SANS_PILOT_ARTIFACT_TTL_SECONDS` | `86400`                | Artifact-token lifetime in seconds; `0` disables expiration         |
| `SANS_PILOT_PUBLIC_BASE_URL`      | unset                  | Public HTTP(S) URL or absolute path for browser artifact downloads  |
| `SANS_PILOT_DOWNLOAD_SIGNING_KEY` | unset                  | Secret used to sign browser artifact download capabilities          |
| `SANS_PILOT_MAX_WORKERS`          | `2`                    | Maximum concurrent scientific worker processes                     |
| `SANS_PILOT_TOOL_TIMEOUT_SECONDS` | `1800`                 | Hard execution timeout per worker; `0` disables it                  |
| `API_TOKEN`                       | unset                  | Optional bearer token                                              |
| `SANS_PILOT_TOOLS`                | `full`                 | Tool profile: `full`, `guided`, or `all`                           |
| `SANS_PILOT_GUESS`                | unset                  | Set to `off` to disable Sassafras model suggestions                |
| `SASSAFRAS_DATA`                  | per-user data folder   | Where the Sassafras curve library is stored                        |

Artifact manifests are stored under `SANS_PILOT_RUNS_DIR/.registry`. When the
runs directory is persistent and shared by all replicas, earlier artifacts can
be reopened after process or pod restarts. `read-sans-artifact` returns images
as MCP image content and CSV/text artifacts in bounded byte chunks.

When both download settings are configured, artifact metadata includes a
non-expiring, HMAC-signed `download_url`. The URL remains valid while the
artifact exists and the signing key is unchanged. It is a bearer capability:
anyone who obtains the URL can download the artifact, so it must not be logged
or shared unintentionally. A future version should add configurable link
expiration.

Long-running scientific calls execute in isolated worker processes. Cancelling
the MCP request terminates the worker and removes its partial artifact
workspace. The hard tool timeout provides cleanup when a client does not
propagate cancellation.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
sans-pilot
```

```bash
docker build -f Dockerfile.dev -t sans-pilot .
docker run -p 8001:8001 -e API_TOKEN="your-token" sans-pilot
```

Run automated checks with:

```bash
ruff check src test
pyright
pytest
```
