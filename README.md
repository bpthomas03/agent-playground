# Agent Playground

A multi-agent NLP workflow for dream analysis: from research question to executable plan, pipeline runs, automatic fixes, and human-readable reports. Powered by GPT-4.1 and a single REPL with tools for ideation, planning, execution, evaluation, re-run, and reporting.

## Features

- **Ideation** — Two “experts” (dream psychology + NLP) agree on a research question and high-level approach.
- **Planning** — Turn the question into a structured, step-by-step plan (with explicit I/O contracts and edge-case notes).
- **Execution** — Generate Python scripts per step, run them, optionally fix and retry on failure; NLTK data is auto-downloaded when needed.
- **Evaluation** — On failed runs, an evaluator proposes fixes (with plan and fix-log context) and applies edits; re-run continues from the fixed step. You can **steer the evaluator** with optional guidance (e.g. “Repeat the previous fix but for the affect columns”) so it overrides the usual “don’t repeat” rule when you know the right fix.
- **Re-run & continue** — Re-execute existing step scripts; if all succeed and the plan has more steps, the run continues by generating and running the rest.
- **Report** — NLP expert writes “Analysis methods”; dream expert writes “Interpretation and conclusions” from run outputs; combined into `report.md` in the run dir.

Data: Dream corpus in `data/clean_dreams.csv` (e.g. DreamBank-style: `dream_id`, `subject`, `date`, `narrative`, `word_count`).

## Prerequisites

- **Python 3.11** (Poetry for deps) or **Docker**
- **OpenAI API key** (for GPT-4.1)

## Setup

### Local (Poetry)

```bash
cd agent-playground
poetry install
```

Create a `.env` in the project root with:

```
OPENAI_API_KEY=sk-...
```

### Docker

Build the image (from project root):

```bash
docker build -t agent-playground .
```

Run with your API key and volume so outputs appear on your machine:

**PowerShell:**

```powershell
docker run -it --rm --env-file .env -v "${PWD}:/app" agent-playground
```

**CMD (Windows):**

```cmd
docker run -it --rm --env-file .env -v "%CD%:/app" agent-playground
```

**Linux/macOS:**

```bash
docker run -it --rm --env-file .env -v "$(pwd):/app" agent-playground
```

## Usage

Start the REPL (local: `poetry run python -m src.main` or Docker command above). You’ll see the list of tools.

### Pipeline workflow

1. **ideate** — Get a research question and approach (saved under `ideation/`).
2. **plan** — Turn it into a plan (saved under `planning/`, default `planning/latest.json`).
3. **execute** — Run the plan step by step; scripts and outputs go to `outputs/run_<timestamp>/`.
4. If a step fails: **evaluate** (proposes fixes, appends to `fix_log.txt`), then **re_run** or **re_run N** to re-run from step N. If the evaluator gets stuck (e.g. fixed “theme trajectories” but not “affect trajectories”), run **evaluate** with guidance, e.g. `evaluate Repeat the previous fix but for the affect columns.` Re-run will also continue the plan (generate and run remaining steps) if all re-run steps succeed.
5. **report** — Write `report.md` in the run dir (NLP methods + dream expert interpretation). Optional: **report** `outputs/run_<timestamp>`.

### Commands (REPL)

| Command | Description |
|--------|-------------|
| `ideate` | Run ideation; save to `ideation/` |
| `plan` [ideation-path] | Create plan (default: `ideation/latest.json`) |
| `execute` [plan-path] | Run plan (default: `planning/latest.json`) |
| `evaluate` [run-dir] [guidance] | Propose fixes for failed run (default: latest). Optional *guidance* overrides “don’t repeat” (e.g. `Same fix for affect columns`). |
| `re_run` [run-dir] [N] | Re-run step scripts from step N or from last fix; continue plan if all succeed |
| `report` [run-dir] | Generate report.md for a successful run (default: latest) |
| `read` <path> | Read a file under the project |
| `calc` <expr> | Simple calculator |
| `suggest` <request> | Code suggestion (saved to `src/experiments/`) |
| `run` <script> | Run a script in `src/experiments/` |
| `fix` <script> | Fix script using last run’s stdout/stderr |
| `exit` / `quit` | Exit REPL |

### Steering the evaluator

When the evaluator keeps proposing the wrong fix or avoids repeating an approach that you know is correct (e.g. same fix for another set of columns), pass **user guidance** so it overrides the default “do not repeat” rule:

- **evaluate** `Repeat the previous fix but for the affect columns.` — latest run, with that instruction.
- **evaluate** `outputs/run_2026-02-27T14-19-33Z` `Same fix for affect columns.` — specific run + guidance.

Guidance is shown first to the model and explicitly allows repeating or applying the same kind of fix when you ask for it.

## Project layout

```
agent-playground/
├── data/
│   └── clean_dreams.csv       # Dream corpus
├── ideation/                  # Ideation outputs (latest + timestamped)
├── planning/                  # Plans (latest.json + timestamped)
├── outputs/
│   └── run_<timestamp>/       # Per-run: step_*.py, run_result.json, plan_snapshot.json,
│                             # fix_log.txt, CSVs, plots, report.md
├── src/
│   ├── main.py               # REPL and tool routing
│   ├── ideation.py           # Ideation layer
│   ├── planning.py           # Planning layer
│   ├── executor.py           # Code gen, run, fix/retry, continue_run
│   ├── evaluator.py          # Evaluate, re_run_existing, fix_log
│   ├── report.py             # Report generation (NLP + dream experts)
│   └── experiments/          # Suggested scripts from tools
├── .env                      # OPENAI_API_KEY (not committed)
├── Dockerfile
├── LICENSE
├── pyproject.toml
└── README.md
```

## Environment

- `OPENAI_API_KEY` — Required for the agent (ideation, planning, execution, evaluate, report). Loaded from `.env` when using `--env-file .env` with Docker or when the app reads `.env` locally.

## License

MIT License. See [LICENSE](LICENSE).
