import os
import re
import subprocess
import sys
from datetime import datetime

from openai import OpenAI


class GPTAgent:
    """Thin wrapper around a real GPT‑4.1 model."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Did you put it in .env and pass --env-file .env?")

        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""


def calculator_tool(expression: str) -> str:
    """Very small, unsafe calculator for simple expressions.

    Supports things like: 1+2, 3 * (4 + 5) / 2
    """

    try:
        # eval is fine here for learning in a controlled environment.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# Max chars to send to the model for HTML/code to avoid token limits
_MAX_INPUT_LENGTH = 12_000


def html_summary_tool(html: str, llm: "GPTAgent") -> str:
    """Ask the LLM to summarize and label sections of HTML (no code execution)."""

    if not html.strip():
        return "No HTML provided."

    truncated = html.strip()[: _MAX_INPUT_LENGTH]
    if len(html.strip()) > _MAX_INPUT_LENGTH:
        truncated += "\n... [truncated]"

    system = (
        "You are an assistant that summarizes and labels HTML. "
        "Given raw HTML, return a concise, structured summary: main sections, headings, "
        "links if relevant, and visible text. Describe structure and content only; do not execute or assume."
    )
    return llm.generate(system_prompt=system, user_message=truncated)


def code_suggestion_tool(prompt: str, llm: "GPTAgent") -> str:
    """Return proposed code (or edits) as text; also save to src/experiments/ if a code block is found."""

    if not prompt.strip():
        return "No request provided. Describe what code you want (e.g. 'parse HTML with BeautifulSoup')."

    system = (
        "You are a helpful programmer. Respond with working code that matches the request. "
        "Use a single markdown code block. Add a one-line comment at the top if the language is not obvious. "
        "Keep explanation outside the block minimal."
    )
    response = llm.generate(system_prompt=system, user_message=prompt.strip())
    return _append_saved_code(response)


# Base dir for safe file access (project root; in Docker this is /app)
_SAFE_BASE = os.path.abspath(os.environ.get("AGENT_PROJECT_ROOT", os.getcwd()))
_MAX_FILE_LENGTH = 50_000

# Only allow writing under this dir; use a volume mount so files appear on your machine
_EXPERIMENTS_DIR = "src/experiments"
_ALLOWED_WRITE_EXTENSIONS = (".py", ".json", ".txt")


def _extract_code_block(text: str) -> tuple[str | None, str]:
    """Extract first markdown code block; return (code, ext) e.g. (code, '.py') or (None, '')."""
    match = re.search(r"```(?:(\w+))?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        return None, ""
    lang = (match.group(1) or "").lower()
    code = match.group(2).strip()
    ext = ".py" if "python" in lang or lang == "py" else ".json" if "json" in lang else ".txt"
    return code, ext


def write_file_tool(path: str, content: str) -> str:
    """Write content to a file under _EXPERIMENTS_DIR. Only .py, .json, .txt allowed."""

    path = path.strip().replace("\\", "/")
    if not path or ".." in path:
        return "Path not allowed (no '..')."
    if not any(path.endswith(e) for e in _ALLOWED_WRITE_EXTENSIONS):
        return f"Only these extensions allowed: {_ALLOWED_WRITE_EXTENSIONS}"

    resolved = os.path.abspath(os.path.join(_SAFE_BASE, _EXPERIMENTS_DIR, path))
    allowed_base = os.path.abspath(os.path.join(_SAFE_BASE, _EXPERIMENTS_DIR))
    if not resolved.startswith(allowed_base):
        return f"Path must be under {_EXPERIMENTS_DIR}."

    try:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        rel = os.path.join(_EXPERIMENTS_DIR, path)
        return f"Wrote {rel}"
    except OSError as e:
        return f"Write error: {e}"


def _append_saved_code(response: str) -> str:
    """If response contains a code block, save it to src/experiments/ and append the path."""
    code, ext = _extract_code_block(response)
    if not code:
        return response
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"suggested_{stamp}{ext}"
    result = write_file_tool(filename, code)
    if result.startswith("Wrote"):
        return response + f"\n\n[{result}]"
    return response


def read_file_tool(path: str) -> str:
    """Read a file under the project. Only paths under project root, no '..'."""

    path = path.strip()
    if not path:
        return "No path provided. Example: read data/search_dreams.html"

    resolved = os.path.abspath(os.path.join(_SAFE_BASE, path))
    if not resolved.startswith(_SAFE_BASE) or ".." in path:
        return f"Path not allowed: {path} (must be under project root, no '..')"

    try:
        with open(resolved, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except OSError as e:
        return f"Read error: {e}"

    if len(content) > _MAX_FILE_LENGTH:
        content = content[:_MAX_FILE_LENGTH] + "\n... [truncated]"
    return content


def suggest_file_tool(arg: str, llm: "GPTAgent") -> str:
    """Read a file and ask the LLM for code to process it. Arg: 'path' then your request."""

    arg = arg.strip()
    if not arg:
        return (
            "Usage: suggest_file <path-to-existing-file> <request>. "
            "Example: suggest_file data/search_dreams.html convert to JSON (code is auto-saved to src/experiments/)."
        )

    # First token that looks like a path (has / or ends with .html/.json/.txt etc), rest is request
    parts = arg.split(maxsplit=1)
    if not parts:
        return "Usage: suggest_file <path> <request>"
    path = parts[0]
    request = parts[1].strip() if len(parts) > 1 else "process this file into a useful structured format"

    content = read_file_tool(path)
    if content.startswith("Path not allowed") or content.startswith("File not found") or content.startswith("Read error") or content.startswith("No path"):
        if content.startswith("File not found"):
            content += (
                " The first argument must be the path to the file to *read* (e.g. data/search_dreams.html). "
                "Suggested code is saved automatically to src/experiments/."
            )
        return content

    system = (
        "You are a helpful programmer. The user will give you file contents and a request. "
        "Respond with working code (e.g. Python) that processes the file as requested. "
        "Use a single markdown code block. Keep explanation outside the block brief."
    )
    user_message = f"File path: {path}\n\nRequest: {request}\n\nFile contents (first part):\n{content[: _MAX_INPUT_LENGTH]}"
    if len(content) > _MAX_INPUT_LENGTH:
        user_message += "\n... [truncated for context]"
    response = llm.generate(system_prompt=system, user_message=user_message)
    return _append_saved_code(response)


# Module name -> pip package name (for run_experiment auto-install)
_PIP_PACKAGE_MAP = {"bs4": "beautifulsoup4", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python"}

# Last run output, so the fix tool can use it (script name, stdout, stderr, returncode)
_last_run_result: dict | None = None


def run_experiment_tool(arg: str) -> str:
    """Run a Python script under src/experiments/. On ModuleNotFoundError, pip-install and retry once."""
    global _last_run_result

    arg = arg.strip()
    if not arg:
        return "Usage: run <script>. Example: run suggested_20260220_145947.py"

    # Allow "suggested_xxx.py" or "src/experiments/suggested_xxx.py"; resolve under experiments only
    name = os.path.basename(arg.replace("\\", "/"))
    if not name.endswith(".py"):
        return "Only .py scripts under src/experiments/ are allowed."
    if ".." in name or name.startswith("/"):
        return "Path not allowed."

    resolved = os.path.abspath(os.path.join(_SAFE_BASE, _EXPERIMENTS_DIR, name))
    allowed_base = os.path.abspath(os.path.join(_SAFE_BASE, _EXPERIMENTS_DIR))
    if not resolved.startswith(allowed_base) or not os.path.isfile(resolved):
        return f"Script not found: {name} (must be in {_EXPERIMENTS_DIR})"

    def run_script() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, resolved],
            cwd=_SAFE_BASE,
            capture_output=True,
            text=True,
            timeout=120,
        )

    out = run_script()
    stderr = out.stderr or ""
    stdout = out.stdout or ""

    if out.returncode != 0 and "ModuleNotFoundError" in stderr:
        match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", stderr)
        if match:
            mod = match.group(1).split(".")[0]
            pkg = _PIP_PACKAGE_MAP.get(mod, mod)
            install = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if install.returncode == 0:
                out = run_script()
                stdout = out.stdout or ""
                stderr = out.stderr or ""
                result = f"[Installed {pkg} and re-ran]\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}".strip()
            else:
                result = f"[Missing module: {mod}. Tried: pip install {pkg}]\n\nstderr:\n{install.stderr or install.stdout or stderr}".strip()
            _last_run_result = {"script": name, "stdout": stdout, "stderr": stderr, "returncode": out.returncode}
            return result

    _last_run_result = {"script": name, "stdout": stdout, "stderr": stderr, "returncode": out.returncode}
    return f"stdout:\n{stdout}\n\nstderr:\n{stderr}".strip() if stdout or stderr else "(no output)"


def fix_experiment_tool(arg: str, llm: "GPTAgent") -> str:
    """Let the LLM see the script and last run's stdout/stderr, then overwrite the script with a fixed version."""

    name = arg.strip() and os.path.basename(arg.replace("\\", "/")) or None
    if not name and _last_run_result:
        name = _last_run_result["script"]
    if not name or not name.endswith(".py"):
        return (
            "Usage: fix <script> (e.g. fix suggested_20260220_151728.py). "
            "Run the script first so the agent can see stdout/stderr, then fix."
        )

    resolved = os.path.abspath(os.path.join(_SAFE_BASE, _EXPERIMENTS_DIR, name))
    allowed_base = os.path.abspath(os.path.join(_SAFE_BASE, _EXPERIMENTS_DIR))
    if not resolved.startswith(allowed_base) or not os.path.isfile(resolved):
        return f"Script not found: {name}"

    with open(resolved, encoding="utf-8") as f:
        current_code = f.read()

    run_info = ""
    if _last_run_result and _last_run_result.get("script") == name:
        r = _last_run_result
        run_info = f"\n\nLast run (returncode={r['returncode']}):\n--- stdout ---\n{r['stdout']}\n--- stderr ---\n{r['stderr']}"
    else:
        run_info = "\n\n(No previous run output for this script. Run it first with: run " + name + ")"

    system = (
        "You are a helpful programmer. The user will give you a Python script and its last run output (stdout/stderr). "
        "Produce a fixed version of the script that addresses any errors or failures. "
        "Respond with a single markdown code block containing the full corrected script. "
        "Preserve the script's purpose and only fix bugs or missing logic. Do not add extra explanation outside the block."
    )
    user_message = f"Script ({name}):\n\n```python\n{current_code}\n```{run_info}"

    response = llm.generate(system_prompt=system, user_message=user_message)
    code, _ = _extract_code_block(response)
    if not code:
        return f"No code block in the response. Agent said:\n{response[:500]}..."

    write_result = write_file_tool(name, code)
    if not write_result.startswith("Wrote"):
        return write_result
    return f"{write_result}\n\nYou can run it again with: run {name}"


class Agent:
    """Single agent that can call tools or the LLM."""

    def __init__(self, llm: GPTAgent, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        # (prefix, tool_fn, use_llm): use_llm=True means call tool(arg, self.llm)
        self._tools: list[tuple[str, object, bool]] = [
            ("calc", calculator_tool, False),
            ("html", html_summary_tool, True),
            ("suggest", code_suggestion_tool, True),
            ("read", read_file_tool, False),
            ("suggest_file", suggest_file_tool, True),
            ("run", run_experiment_tool, False),
            ("fix", fix_experiment_tool, True),
        ]

    def handle(self, user_input: str) -> str:
        """Route the user input either to a tool or the LLM."""

        stripped = user_input.strip()
        lower = stripped.lower()

        for prefix, tool, use_llm in self._tools:
            if lower.startswith(prefix + " "):
                arg = stripped[len(prefix) :].strip()
                if use_llm:
                    result = tool(arg, self.llm)
                else:
                    result = tool(arg)
                return f"(tool={prefix}) {result}"

        # Otherwise, ask the LLM
        return self.llm.generate(
            system_prompt=self.system_prompt,
            user_message=stripped,
        )


def main() -> None:
    system_prompt = (
        "You are a thoughtful assistant that reasons step by step. "
        "When the user asks general questions, answer clearly. "
        "When they share results from tools (calculator, HTML summary, code suggestion), incorporate them into your reasoning."
    )
    llm = GPTAgent()
    agent = Agent(llm=llm, system_prompt=system_prompt)

    print("GPT‑4.1-powered agent with tools: calc, html, suggest, read, suggest_file.")
    print("  calc <expr>              – calculator")
    print("  html <html>              – summarize/label HTML (no execution)")
    print("  suggest <req>            – get code suggestions (you apply manually)")
    print("  read <path>              – read file under project (e.g. data/search_dreams.html)")
    print("  suggest_file <input-file> <req> – e.g. suggest_file data/search_dreams.html parse to JSON (saved to src/experiments/)")
    print("  run <script>              – run a script in src/experiments/ (auto-installs missing deps once)")
    print("  fix <script>             – use last run's stdout/stderr to fix the script (then run again)")
    print("  exit / quit              – quit")
    print("\n  Tip: to see saved files in your repo, run Docker with the volume BEFORE the image:")
    print('       docker run -it --rm --env-file .env -v "${PWD}:/app" agent-playground')
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        reply = agent.handle(user_input)
        print("\nAgent:")
        print(reply)
        print()


if __name__ == "__main__":
    main()

