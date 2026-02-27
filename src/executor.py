"""
Executor layer: run the plan step by step. For each step, generate code,
run it, capture results, and optionally fix and retry on failure.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime

_DEFAULT_PLAN_PATH = "planning/latest.json"
_RUNS_DIR = "outputs"
_SCRIPT_NAME = "step_{id}.py"
_MAX_FIX_RETRIES = 1
_RUN_TIMEOUT = 600  # 10 min; LDA on large DTM can be slow


def _project_root() -> str:
    return os.path.abspath(os.environ.get("AGENT_PROJECT_ROOT") or os.getcwd())


def load_plan(path: str | None = None) -> dict:
    """Load plan JSON from path (default: planning/latest.json)."""
    root = _project_root()
    resolved = os.path.normpath(os.path.join(root, path or _DEFAULT_PLAN_PATH))
    with open(resolved, encoding="utf-8") as f:
        return json.load(f)


def _is_file_path(s: str) -> bool:
    """True if this looks like a file path (for resolving inputs)."""
    s = s.strip()
    if not s:
        return False
    return "/" in s or "\\" in s or s.endswith((".csv", ".json", ".pkl", ".png", ".txt"))


def _output_path(step_id: str, output_name: str, run_dir: str) -> str:
    """Decide path for one output of a step. Use run_dir/step_N_name.ext or run_dir/name if name looks like a filename."""
    safe_name = output_name.strip().replace(" ", "_")
    if "." in safe_name and any(safe_name.endswith(ext) for ext in (".csv", ".json", ".pkl", ".png", ".txt")):
        return os.path.join(run_dir, os.path.basename(safe_name))
    return os.path.join(run_dir, f"step_{step_id}_{safe_name}.pkl")


def resolve_step_io(plan: dict, run_dir: str, output_name_to_path: dict[str, str]) -> list[tuple[list[str], list[str]]]:
    """
    For each step in the plan, resolve input paths and output paths.
    Returns list of (input_paths, output_paths) per step in order.
    output_name_to_path is updated in place as we assign outputs.
    """
    root = _project_root()
    steps = plan.get("steps") or []
    result: list[tuple[list[str], list[str]]] = []

    for step in steps:
        step_id = str(step.get("id", ""))
        inputs = list(step.get("inputs") or [])
        outputs = list(step.get("outputs") or [])

        input_paths: list[str] = []
        for inp in inputs:
            if inp in output_name_to_path:
                input_paths.append(output_name_to_path[inp])
            elif _is_file_path(inp):
                p = os.path.normpath(os.path.join(root, inp))
                input_paths.append(p)
            else:
                input_paths.append(os.path.join(run_dir, f"unknown_{inp}.pkl"))

        output_paths: list[str] = []
        for out in outputs:
            p = _output_path(step_id, out, run_dir)
            output_paths.append(p)
            output_name_to_path[out] = p

        result.append((input_paths, output_paths))

    return result


def _build_contract_context(
    step: dict,
    next_step: dict | None,
    agreed_question: str,
    approach: str,
) -> str:
    """Build a context string for code gen: research goal, this step's output_spec/notes, and next step's expectations."""
    parts = [
        f"Research question: {agreed_question[:300]}",
        f"Approach: {approach[:400]}",
        "",
        "THIS STEP'S CONTRACT (you must satisfy this):",
        (step.get("output_spec") or "(none specified)").strip() or "(none specified)",
        "",
    ]
    if step.get("notes"):
        parts.append("Edge cases / notes: " + step.get("notes", "").strip())
        parts.append("")
    if next_step:
        parts.append(
            "NEXT STEP will consume your outputs. It expects:\n"
            f"  description: {next_step.get('description', '')}\n"
            f"  inputs (names): {next_step.get('inputs', [])}\n"
            "Your outputs must be written so the next step can read them (correct columns, pickle keys, file paths)."
        )
    return "\n".join(parts)


EXECUTOR_STEP_SYSTEM = """You are a Python developer implementing one step of a data-analysis pipeline. You will be given:
- Step id, action name, and description.
- Exact absolute paths for INPUTS (read from these files).
- Exact absolute paths for OUTPUTS (you must write to these paths).
- A CONTRACT section: output_spec (exact format the next step expects), notes (edge cases), and what the next step will do.

Rules:
- Write a single, self-contained Python script that reads only from the input paths and writes only to the output paths.
- You MUST satisfy the CONTRACT: e.g. if output_spec says "DataFrame with column processed_narrative", include that column; if it says "pickle dict with keys lda_model, vectorizer", save exactly that so the next step can load it.
- Honor edge-case notes (e.g. "never yield empty vocabulary"—ensure preprocessing does not remove all text from every document, or use a fallback).
- Use standard libraries and common data-science packages: pandas, numpy, sklearn, pickle, json. If you use sklearn or nltk, add a comment at the top: # requires: scikit-learn (or nltk).
- LDA/topic modeling: use a fixed number of topics (e.g. n_topics=10 or from a single config value). Do NOT loop over k=2,3,...,K and fit many LDA models to pick the best—that causes timeouts on large corpora (e.g. 16k docs). One fit only.
- NLTK: If the step uses NLTK (word_tokenize, WordNetLemmatizer, pos_tag, stopwords), at script start ensure data is available: call nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('omw-1.4', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True). Use try/except around word_tokenize so that if tokenizer data is missing you fall back to a simple split (e.g. text.split()) rather than exiting.
- For DataFrames or objects passed to later steps, use pickle. For topic models, save both the model and the vectorizer (e.g. in one dict or separate vectorizer.pkl) so the next step can get feature names.
- If the step has multiple OUTPUT paths (e.g. .png and .csv), write to every path. CSV outputs are required for reporting: do not skip any.
- Final comparison/visualization step: You must produce (1) theme_trajectories.csv with columns such as life_stage, topic_id, topic_label (or top_terms—interpretable theme name from the topic model), value, group (diarist/others). (2) valence_trajectories.csv or affect_trajectories.csv with life_stage, metric (e.g. valence, arousal), value, group. (3) statistical_comparison_results.csv. (4) Plot(s). Load the topic labels (from the step that produces them, e.g. topic_labels_df with topic_id and top_terms) and use those interpretable labels on the plot axes and in the theme_trajectories CSV—never use only 'theme1', 'theme2' in the CSV or on the plot; include the actual top terms or a short label so readers know what each theme means.
- On errors: print(msg, file=sys.stderr) and sys.exit(1). On success: print a short message to stdout.
- Respond with ONLY a single markdown code block containing the full Python script, no other text."""


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else None


def generate_step_code(
    llm: "object",
    step: dict,
    input_paths: list[str],
    output_paths: list[str],
    contract_context: str,
) -> str | None:
    """Ask the LLM to generate Python code for this step. Returns code string or None."""
    step_id = step.get("id", "?")
    action = step.get("action", "?")
    description = step.get("description", "")

    user = (
        f"Step {step_id} | action: {action}\n"
        f"Description: {description}\n\n"
        f"INPUT paths (read from these):\n" + "\n".join(f"  - {p}" for p in input_paths) + "\n\n"
        f"OUTPUT paths (write to these, in order):\n" + "\n".join(f"  - {p}" for p in output_paths) + "\n\n"
        f"CONTRACT (satisfy this so the next step works):\n{contract_context}\n\n"
        "Generate the Python script."
    )
    response = llm.generate(system_prompt=EXECUTOR_STEP_SYSTEM, user_message=user)
    return _extract_code_block(response)


def run_script(script_path: str, cwd: str, timeout: int = _RUN_TIMEOUT) -> tuple[int, str, str]:
    """Run a Python script; return (returncode, stdout, stderr)."""
    out = subprocess.run(
        [sys.executable, script_path],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )
    return out.returncode, out.stdout or "", out.stderr or ""


def fix_script(llm: "object", script_path: str, stdout: str, stderr: str) -> str | None:
    """Ask the LLM to fix the script given last run output. Returns new code or None."""
    with open(script_path, encoding="utf-8") as f:
        code = f.read()
    nltk_hint = ""
    if "nltk" in (stderr or "").lower() or "LookupError" in (stderr or "") or "punkt" in (stderr or "").lower():
        nltk_hint = " If the error is NLTK 'Resource not found' or missing punkt/punkt_tab/wordnet: ensure at script start you call nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('omw-1.4', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True). Add a try/except around word_tokenize with fallback to text.split() if tokenizer fails."
    system = (
        "You are a helpful programmer. The user will give you a Python script and its run output (stdout and stderr). "
        "The script may have printed errors to stdout instead of stderr—check both. "
        "Produce a fixed version that addresses the error. Respond with a single markdown code block containing the full script. "
        "Preserve the script's purpose; only fix bugs or missing logic. "
        "Use print(msg, file=sys.stderr) and sys.exit(1) for errors so the next run surfaces them correctly."
        + nltk_hint
    )
    user = f"Script:\n\n```python\n{code}\n```\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
    response = llm.generate(system_prompt=system, user_message=user)
    return _extract_code_block(response)


# Pip package map for auto-install on ModuleNotFoundError (match main.py pattern).
_PIP_PACKAGE_MAP = {"bs4": "beautifulsoup4", "sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python", "nltk": "nltk", "spacy": "spacy"}

def _ensure_nltk_data(cwd: str) -> None:
    """Download common NLTK data so step scripts that use NLTK do not fail with LookupError."""
    code = """
import sys
try:
    import nltk
    for name in ["punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4", "averaged_perceptron_tagger"]:
        try:
            nltk.download(name, quiet=True)
        except Exception:
            pass
except ImportError:
    pass
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        timeout=120,
        env={**os.environ},
    )


def _is_nltk_resource_error(stderr: str) -> bool:
    """True if stderr indicates NLTK data/resource missing (LookupError, Resource not found, Downloading package)."""
    s = (stderr or "").lower()
    if "nltk" not in s:
        return False
    return (
        "lookuperror" in s
        or "resource " in s and "not found" in s
        or "downloading package" in s
        or "punkt_tab" in s
        or "punkt" in s and "token" in s
    )


def run_script_with_auto_install(script_path: str, cwd: str) -> tuple[int, str, str]:
    """Run script; on ModuleNotFoundError try pip install and re-run; on NLTK resource error run nltk.download and re-run."""
    returncode, stdout, stderr = run_script(script_path, cwd)
    if returncode != 0:
        if "ModuleNotFoundError" in stderr:
            match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", stderr)
            if match:
                mod = match.group(1).split(".")[0]
                pkg = _PIP_PACKAGE_MAP.get(mod, mod)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", pkg],
                    cwd=cwd,
                    capture_output=True,
                    timeout=60,
                )
                returncode, stdout, stderr = run_script(script_path, cwd)
        if returncode != 0 and _is_nltk_resource_error(stderr):
            _ensure_nltk_data(cwd)
            returncode, stdout, stderr = run_script(script_path, cwd)
    return returncode, stdout, stderr


@dataclass
class StepResult:
    step_id: str
    action: str
    returncode: int
    stdout: str
    stderr: str
    script_path: str
    fixed_and_retried: bool = False
    # When fixed_and_retried is True, these hold the first run so you can verify fix actually ran
    first_run_returncode: int | None = None
    first_run_stdout: str = ""
    first_run_stderr: str = ""


@dataclass
class ExecutorResult:
    run_dir: str
    plan_path: str
    agreed_question: str
    step_results: list[StepResult] = field(default_factory=list)
    success: bool = False
    timestamp: str = ""


def run_executor(
    llm: "object",
    plan_path: str | None = None,
    base_path: str | None = None,
) -> ExecutorResult:
    """
    Load plan, create run dir, for each step: generate code, save, run, optionally fix and retry.
    Returns ExecutorResult with step results and run_dir.
    """
    root = base_path or _project_root()
    plan_path = plan_path or _DEFAULT_PLAN_PATH
    resolved_plan_path = os.path.normpath(os.path.join(root, plan_path))

    plan = load_plan(plan_path)
    steps = plan.get("steps") or []
    agreed_question = plan.get("agreed_question", "")
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = os.path.join(root, _RUNS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    output_name_to_path: dict[str, str] = {}
    io_per_step = resolve_step_io(plan, run_dir, output_name_to_path)

    # Save plan snapshot so evaluator can load it
    plan_snapshot_path = os.path.join(run_dir, "plan_snapshot.json")
    with open(plan_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    step_results: list[StepResult] = []
    all_ok = True

    for idx, step in enumerate(steps):
        step_id = str(step.get("id", idx + 1))
        action = step.get("action", "?")
        input_paths, output_paths = io_per_step[idx]
        script_name = _SCRIPT_NAME.format(id=step_id)
        script_path = os.path.join(run_dir, script_name)

        # Pre-step check: all inputs must exist (fail fast if a previous step didn't produce them)
        missing = [p for p in input_paths if not os.path.isfile(p)]
        if missing:
            step_results.append(StepResult(
                step_id=step_id,
                action=action,
                returncode=1,
                stdout="",
                stderr=f"Executor: missing required inputs (previous step failed or did not write outputs):\n  " + "\n  ".join(missing[:5]) + ("\n  ..." if len(missing) > 5 else ""),
                script_path=script_path,
            ))
            all_ok = False
            break

        # Build contract context: this step's output_spec/notes + next step's expectations
        next_step = steps[idx + 1] if idx + 1 < len(steps) else None
        contract_context = _build_contract_context(step, next_step, agreed_question, plan.get("approach", ""))
        # Generate code
        code = generate_step_code(llm, step, input_paths, output_paths, contract_context)
        if not code:
            step_results.append(StepResult(
                step_id=step_id,
                action=action,
                returncode=-1,
                stdout="",
                stderr="Executor: LLM did not return valid code block.",
                script_path=script_path,
            ))
            all_ok = False
            break

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Run (and optionally fix once)
        returncode, stdout, stderr = run_script_with_auto_install(script_path, root)
        fixed = False
        first_run_returncode: int | None = None
        first_run_stdout = ""
        first_run_stderr = ""
        if returncode != 0 and _MAX_FIX_RETRIES > 0:
            first_run_returncode, first_run_stdout, first_run_stderr = returncode, stdout, stderr
            new_code = fix_script(llm, script_path, stdout, stderr)
            if new_code:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                returncode, stdout, stderr = run_script_with_auto_install(script_path, root)
                fixed = True

        step_results.append(StepResult(
            step_id=step_id,
            action=action,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            script_path=script_path,
            fixed_and_retried=fixed,
            first_run_returncode=first_run_returncode,
            first_run_stdout=first_run_stdout,
            first_run_stderr=first_run_stderr,
        ))
        if returncode != 0:
            all_ok = False
            break

    return ExecutorResult(
        run_dir=run_dir,
        plan_path=resolved_plan_path,
        agreed_question=agreed_question,
        step_results=step_results,
        success=all_ok,
        timestamp=timestamp,
    )


def continue_run(
    llm: "object",
    run_dir: str,
    base_path: str | None = None,
) -> tuple[bool, list[dict], str]:
    """
    After a re_run, if all existing scripts succeeded, generate and run the remaining plan steps.
    Loads plan from run_dir/plan_snapshot.json and run_result.json; appends new step results and writes run_result.json.
    Returns (success, new_step_dicts, message).
    """
    root = base_path or _project_root()
    run_dir = os.path.normpath(run_dir)
    if not os.path.isdir(run_dir):
        return False, [], f"Run directory not found: {run_dir}"

    plan_path = os.path.join(run_dir, "plan_snapshot.json")
    if not os.path.isfile(plan_path):
        return False, [], "Cannot continue: plan_snapshot.json not found in run dir (required to generate remaining steps)."

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    plan_steps = plan.get("steps") or []
    agreed_question = plan.get("agreed_question", "")
    approach = plan.get("approach", "")

    run_result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.isfile(run_result_path):
        return False, [], "run_result.json not found."
    with open(run_result_path, encoding="utf-8") as f:
        existing = json.load(f)
    steps_done = existing.get("steps") or []
    next_idx = len(steps_done)
    if next_idx >= len(plan_steps):
        return True, [], "Run already complete; all plan steps have been executed."

    output_name_to_path: dict[str, str] = {}
    io_per_step = resolve_step_io(plan, run_dir, output_name_to_path)

    new_step_dicts: list[dict] = []
    all_ok = True
    for idx in range(next_idx, len(plan_steps)):
        step = plan_steps[idx]
        step_id = str(step.get("id", idx + 1))
        action = step.get("action", "?")
        input_paths, output_paths = io_per_step[idx]
        script_path = os.path.join(run_dir, _SCRIPT_NAME.format(id=step_id))

        missing = [p for p in input_paths if not os.path.isfile(p)]
        if missing:
            new_step_dicts.append({
                "step_id": step_id,
                "action": action,
                "returncode": 1,
                "stdout": "",
                "stderr": f"Executor: missing required inputs:\n  " + "\n  ".join(missing[:5]) + ("\n  ..." if len(missing) > 5 else ""),
                "script_path": script_path,
                "fixed_and_retried": False,
            })
            all_ok = False
            break

        next_step = plan_steps[idx + 1] if idx + 1 < len(plan_steps) else None
        contract_context = _build_contract_context(step, next_step, agreed_question, approach)
        code = generate_step_code(llm, step, input_paths, output_paths, contract_context)
        if not code:
            new_step_dicts.append({
                "step_id": step_id,
                "action": action,
                "returncode": -1,
                "stdout": "",
                "stderr": "Executor: LLM did not return valid code block.",
                "script_path": script_path,
                "fixed_and_retried": False,
            })
            all_ok = False
            break

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        returncode, stdout, stderr = run_script_with_auto_install(script_path, root)
        fixed = False
        first_run_returncode: int | None = None
        first_run_stdout = ""
        first_run_stderr = ""
        if returncode != 0 and _MAX_FIX_RETRIES > 0:
            first_run_returncode, first_run_stdout, first_run_stderr = returncode, stdout, stderr
            new_code = fix_script(llm, script_path, stdout, stderr)
            if new_code:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                returncode, stdout, stderr = run_script_with_auto_install(script_path, root)
                fixed = True

        step_dict = {
            "step_id": step_id,
            "action": action,
            "returncode": returncode,
            "stdout": stdout[:2000] + ("..." if len(stdout) > 2000 else ""),
            "stderr": stderr[:2000] + ("..." if len(stderr) > 2000 else ""),
            "script_path": script_path,
            "fixed_and_retried": fixed,
        }
        if fixed and first_run_returncode is not None:
            step_dict["first_run"] = {
                "returncode": first_run_returncode,
                "stdout": (first_run_stdout or "")[:2000] + ("..." if len(first_run_stdout or "") > 2000 else ""),
                "stderr": (first_run_stderr or "")[:2000] + ("..." if len(first_run_stderr or "") > 2000 else ""),
            }
        new_step_dicts.append(step_dict)
        if returncode != 0:
            all_ok = False
            break

    all_steps = steps_done + new_step_dicts
    new_payload = {
        "run_dir": run_dir,
        "plan_path": existing.get("plan_path", ""),
        "agreed_question": existing.get("agreed_question", agreed_question),
        "success": all_ok and len(all_steps) == len(plan_steps),
        "timestamp": existing.get("timestamp", ""),
        "re_run": True,
        "continued": len(new_step_dicts) > 0,
        "steps": all_steps,
    }
    with open(run_result_path, "w", encoding="utf-8") as f:
        json.dump(new_payload, f, indent=2, ensure_ascii=False)

    msg_lines = [f"Continued run: generated and ran steps {next_idx + 1}–{next_idx + len(new_step_dicts)}.", f"Success: {all_ok}"]
    for d in new_step_dicts:
        status = "ok" if d.get("returncode") == 0 else "FAILED"
        msg_lines.append(f"  {d['step_id']}. [{d.get('action', '')}] {status}")
        if d.get("returncode", 0) != 0:
            err = (d.get("stderr") or d.get("stdout") or "").strip()[:300]
            msg_lines.append(f"      error: {err}")
    return all_ok, new_step_dicts, "\n".join(msg_lines)


def save_run_result(result: ExecutorResult) -> str:
    """Write run_result.json into the run_dir. Returns path to that file."""
    path = os.path.join(result.run_dir, "run_result.json")
    steps_payload = []
    for r in result.step_results:
        step_dict = {
            "step_id": r.step_id,
            "action": r.action,
            "returncode": r.returncode,
            "stdout": r.stdout[:2000] + ("..." if len(r.stdout) > 2000 else ""),
            "stderr": r.stderr[:2000] + ("..." if len(r.stderr) > 2000 else ""),
            "script_path": r.script_path,
            "fixed_and_retried": r.fixed_and_retried,
        }
        if r.fixed_and_retried and r.first_run_returncode is not None:
            step_dict["first_run"] = {
                "returncode": r.first_run_returncode,
                "stdout": (r.first_run_stdout or "")[:2000] + ("..." if len(r.first_run_stdout or "") > 2000 else ""),
                "stderr": (r.first_run_stderr or "")[:2000] + ("..." if len(r.first_run_stderr or "") > 2000 else ""),
            }
        steps_payload.append(step_dict)
    payload = {
        "run_dir": result.run_dir,
        "plan_path": result.plan_path,
        "agreed_question": result.agreed_question,
        "success": result.success,
        "timestamp": result.timestamp,
        "steps": steps_payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def format_result(result: ExecutorResult) -> str:
    """Human-readable summary of the executor run. On failure, show both stdout and stderr (scripts often print errors to stdout)."""
    lines = [
        "--- Executor run ---",
        "",
        f"Run dir: {result.run_dir}",
        f"Plan: {result.plan_path}",
        f"Success: {result.success}",
        "",
        "Steps:",
    ]
    for r in result.step_results:
        status = "ok" if r.returncode == 0 else "FAILED"
        lines.append(f"  {r.step_id}. [{r.action}] {status}")
        if r.returncode != 0:
            err_msg = (r.stderr or r.stdout or "(no output)").strip()
            if len(err_msg) > 400:
                err_msg = err_msg[:400] + "..."
            lines.append(f"      error: {err_msg}")
            if r.fixed_and_retried and (r.first_run_stdout or r.first_run_stderr):
                first_err = (r.first_run_stderr or r.first_run_stdout or "").strip()
                if len(first_err) > 200:
                    first_err = first_err[:200] + "..."
                lines.append(f"      (first run before fix: {first_err})")
                lines.append(f"      (above error is after fix and retry)")
    return "\n".join(lines)


if __name__ == "__main__":
    from src.main import GPTAgent

    llm = GPTAgent()
    result = run_executor(llm, plan_path=None)
    save_run_result(result)
    print(format_result(result))
    print(f"\nRun dir: {result.run_dir}")
    print("Scripts and outputs are in the run dir; run_result.json has full step log.")
