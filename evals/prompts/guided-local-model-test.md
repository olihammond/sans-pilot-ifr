# Guided profile test for local models

Run against a server started with `SANS_PILOT_TOOLS=guided`, or through
`sans-pilot-agent`. The goal is to check that a small model can follow the
fixed procedure without inventing values.

## Prompt

Analyse the uploaded file `simulated_sans_data.csv`. Nothing is known about
the sample. Follow the server's procedure and report the result.

## Pass criteria

1. The first scientific call is `run-sans-guided-analysis` (after at most one
   `list-uploaded-sans-files`).
2. Every later call passes the `session_id` that call returned.
3. No more than three `run-sans-fit` refinements are made.
4. The final answer names the selected model and quotes reduced chi-squared,
   the verdict and each fitted value with its uncertainty exactly as returned.
5. The answer states that a good fit does not prove the model is correct, and
   mentions any candidate marked `close_to_best`.
6. No parameter name, value or statistic appears that no tool returned.

## Failure handling

Send one deliberately wrong call (`set-sans-parameter` with a parameter that
does not exist) and confirm the model recovers using the names in the error.
