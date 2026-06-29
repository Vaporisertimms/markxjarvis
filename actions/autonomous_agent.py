"""
autonomous_agent.py  —  MARK XLVII  |  Autonomous Work Agent
=============================================================
Provides Jarvis with autonomous, multi-step work capabilities including:

  • Skill-based task routing   — maps intents to specialist sub-agents
    • Coding Agent              — write, refactor, debug, explain code
      • Research Agent            — deep web research with structured reports
        • File Agent                — create / read / edit / organise files
          • Shell Agent               — run terminal commands safely with approval
            • Plan-Execute loop         — Gemini plans steps, each step dispatched
                                            to the right sub-agent and results fed back
                                              • Work Memory               — persists task context across turns

                                              All agents share the project's existing config / API-key infrastructure.
                                              """

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths & config helpers  (mirrors existing project pattern)
# ---------------------------------------------------------------------------

def _get_base_dir() -> Path:
      if getattr(sys, "frozen", False):
                return Path(sys.executable).parent
            return Path(__file__).resolve().parent.parent

BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
WORK_DIR        = Path.home() / "Desktop" / "JarvisWork"
TASK_LOG_PATH   = BASE_DIR / "memory" / "agent_tasks.json"

PLANNER_MODEL = "gemini-2.5-flash"
WORKER_MODEL  = "gemini-2.5-flash"
MAX_STEPS     = 10          # safety ceiling for plan-execute loops
SHELL_TIMEOUT = 30          # seconds


def _get_api_key() -> str:
      with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)["gemini_api_key"]


def _gemini(model: str = WORKER_MODEL):
      """Return a thin Gemini wrapper consistent with the rest of the project."""
    from google import genai
    client = genai.Client(api_key=_get_api_key())

    class _Wrapper:
              def generate(self, prompt: str) -> str:
                            resp = client.models.generate_content(model=model, contents=prompt)
                            return resp.text.strip()

          return _Wrapper()


def _strip_fences(text: str) -> str:
      text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\r?\n?", "", text)
    text = re.sub(r"\r?\n?```\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Task / work memory helpers
# ---------------------------------------------------------------------------

def _load_task_log() -> list:
      try:
                return json.loads(TASK_LOG_PATH.read_text(encoding="utf-8"))
except Exception:
        return []


def _save_task_log(log: list) -> None:
      TASK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASK_LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_task(goal: str, steps: list, outcome: str) -> None:
      log = _load_task_log()
    log.append({
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "goal": goal,
              "steps": steps,
              "outcome": outcome,
    })
    _save_task_log(log[-50:])   # keep last 50 tasks


# ---------------------------------------------------------------------------
# Sub-Agent: Coding Agent
# ---------------------------------------------------------------------------

def coding_agent(task: str, language: str = "python", save_path: str = "") -> dict:
      """
          Write, refactor, debug, or explain code.

              Returns:
                      {"success": bool, "code": str, "explanation": str, "file": str or None}
                          """
    gem = _gemini()

    # Classify intent
    intent_map = {
              "write":    ["write", "create", "build", "make", "generate", "implement"],
              "refactor": ["refactor", "improve", "optimise", "optimize", "clean", "rewrite"],
              "debug":    ["debug", "fix", "error", "bug", "broken", "failing"],
              "explain":  ["explain", "what does", "how does", "understand", "describe"],
    }
    task_lower = task.lower()
    intent = "write"
    for key, keywords in intent_map.items():
              if any(kw in task_lower for kw in keywords):
                            intent = key
                            break

          if intent == "explain":
                    prompt = (
                                  f"You are an expert {language} developer. Explain the following clearly "
                                  f"and concisely for a professional developer:\n\n{task}"
                    )
                    explanation = gem.generate(prompt)
                    return {"success": True, "code": "", "explanation": explanation, "file": None}

    # Write / refactor / debug
    prompt = (
              f"You are an expert {language} developer. {intent.capitalize()} the following:\n\n"
              f"{task}\n\n"
              f"Rules:\n"
              f"- Output ONLY the final {language} code, inside a single code block.\n"
              f"- After the code block, write a SHORT explanation (3-5 sentences) of what you did.\n"
              f"- The code must be complete, runnable, and well-commented.\n"
              f"- Use best practices for {language}."
    )

    raw = gem.generate(prompt)

    # Split code from explanation
    code_match = re.search(r"```[a-zA-Z]*\n?(.*?)```", raw, re.DOTALL)
    code        = _strip_fences(code_match.group(0)) if code_match else _strip_fences(raw)
    explanation = raw[code_match.end():].strip() if code_match else ""

    # Save to file if path provided or default to Desktop
    ext_map = {
              "python": ".py", "javascript": ".js", "typescript": ".ts",
              "html": ".html", "css": ".css", "java": ".java", "cpp": ".cpp",
              "bash": ".sh", "shell": ".sh", "rust": ".rs", "go": ".go",
              "sql": ".sql", "json": ".json",
    }
    ext  = ext_map.get(language.lower(), ".py")
    dest = Path(save_path) if save_path else (WORK_DIR / f"agent_{intent}{ext}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(code, encoding="utf-8")

    return {"success": True, "code": code, "explanation": explanation, "file": str(dest)}


# ---------------------------------------------------------------------------
# Sub-Agent: Research Agent
# ---------------------------------------------------------------------------

def research_agent(topic: str, depth: str = "detailed") -> dict:
      """
          Perform deep research on a topic using Gemini's knowledge + grounding.

              depth: "quick" | "detailed" | "expert"

    Returns:
            {"success": bool, "report": str, "key_points": list, "file": str}
                """
    gem = _gemini(PLANNER_MODEL)   # use the most capable model for research

    depth_instruction = {
              "quick":    "Provide a concise 3-5 paragraph summary.",
              "detailed": "Provide a detailed report with sections: Overview, Key Findings, Implications, and Further Reading.",
              "expert":   "Provide an expert-level deep-dive with technical details, data points, conflicting viewpoints, and a structured conclusion.",
    }.get(depth, "Provide a detailed report.")

    prompt = (
              f"You are a world-class research analyst. Research the following topic thoroughly:\n\n"
              f"TOPIC: {topic}\n\n"
              f"{depth_instruction}\n\n"
              f"At the end, provide a JSON block with a 'key_points' list (5-10 bullet strings).\n"
              f"Format: ```json\n{{\"key_points\": [\"...\", ...]}}\n```"
    )

    raw     = gem.generate(prompt)
    json_m  = re.search(r"```json\s*(.*?)```", raw, re.DOTALL)
    key_pts = []
    report  = raw

    if json_m:
              try:
                            key_pts = json.loads(json_m.group(1)).get("key_points", [])
                            report  = raw[:json_m.start()].strip()
except Exception:
            pass

    # Save report
    safe_name = re.sub(r"[^\w\s-]", "", topic)[:40].strip().replace(" ", "_")
    dest = WORK_DIR / "research" / f"{safe_name}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"# Research: {topic}\n\n{report}", encoding="utf-8")

    return {"success": True, "report": report, "key_points": key_pts, "file": str(dest)}


# ---------------------------------------------------------------------------
# Sub-Agent: File Agent
# ---------------------------------------------------------------------------

def file_agent(action: str, path: str, content: str = "") -> dict:
      """
          Safe file operations within the JarvisWork directory.

              action: "read" | "write" | "append" | "list" | "delete" | "summarise"

    Returns:
            {"success": bool, "result": str}
                """
    # Resolve and sandbox path
    target = Path(path) if Path(path).is_absolute() else WORK_DIR / path
    target = target.resolve()

    # Security: keep within WORK_DIR or Desktop
    allowed_roots = [WORK_DIR.resolve(), Path.home().resolve() / "Desktop"]
    if not any(str(target).startswith(str(r)) for r in allowed_roots):
              return {"success": False, "result": f"Access denied: {target} is outside allowed directories."}

    if action == "read":
              if not target.exists():
                            return {"success": False, "result": f"File not found: {target}"}
                        return {"success": True, "result": target.read_text(encoding="utf-8")}

elif action == "write":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"success": True, "result": f"Written {len(content)} chars to {target}"}

elif action == "append":
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
                      f.write(content)
                  return {"success": True, "result": f"Appended to {target}"}

elif action == "list":
        scan = target if target.is_dir() else target.parent
        if not scan.exists():
                      return {"success": False, "result": f"Directory not found: {scan}"}
                  items = [str(p.relative_to(scan)) for p in scan.iterdir()]
        return {"success": True, "result": "\n".join(items)}

elif action == "delete":
        if not target.exists():
                      return {"success": False, "result": f"Not found: {target}"}
                  target.unlink()
        return {"success": True, "result": f"Deleted {target}"}

elif action == "summarise":
        if not target.exists():
                      return {"success": False, "result": f"File not found: {target}"}
                  text = target.read_text(encoding="utf-8")[:8000]   # limit tokens
        gem  = _gemini()
        summary = gem.generate(
                      f"Summarise this file concisely (3-5 sentences):\n\n{text}"
        )
        return {"success": True, "result": summary}

    return {"success": False, "result": f"Unknown action: {action}"}


# ---------------------------------------------------------------------------
# Sub-Agent: Shell Agent
# ---------------------------------------------------------------------------

def shell_agent(command: str, cwd: str = "", require_approval: bool = True) -> dict:
      """
          Execute a terminal command safely.

              require_approval: if True, prints the command and asks for confirmation
                                    before running (recommended for destructive commands).

                                        Returns:
                                                {"success": bool, "stdout": str, "stderr": str, "returncode": int}
                                                    """
    DANGEROUS = ["rm -rf", "rmdir /s", "format", "del /f", "shutdown", "reboot",
                                  "mkfs", "dd if=", ":(){", "fork bomb", "curl | bash", "wget | sh"]

    for bad in DANGEROUS:
              if bad.lower() in command.lower():
                            return {
                                              "success": False,
                                              "stdout": "",
                                              "stderr": f"Blocked: dangerous pattern '{bad}' detected.",
                                              "returncode": -1,
                            }

    if require_approval:
              print(f"\n[ShellAgent] About to run: {command}")
        ans = input("  Approve? (yes/no): ").strip().lower()
        if ans not in ("yes", "y"):
                      return {"success": False, "stdout": "", "stderr": "Cancelled by user.", "returncode": -1}

    run_dir = Path(cwd) if cwd else WORK_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
              proc = subprocess.run(
                  command,
                  shell=True,
                  capture_output=True,
                  text=True,
                  cwd=str(run_dir),
                  timeout=SHELL_TIMEOUT,
    )
        return {
                      "success": proc.returncode == 0,
                      "stdout": proc.stdout.strip(),
                      "stderr": proc.stderr.strip(),
                      "returncode": proc.returncode,
        }
except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timed out.", "returncode": -1}
except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


# ---------------------------------------------------------------------------
# Planner  —  decomposes a high-level goal into ordered steps
# ---------------------------------------------------------------------------

_AGENT_REGISTRY = {
      "coding":   "Use coding_agent(task, language, save_path) — write/refactor/debug/explain code",
      "research": "Use research_agent(topic, depth) — research a topic and produce a report",
      "file":     "Use file_agent(action, path, content) — read/write/list/delete/summarise files",
      "shell":    "Use shell_agent(command, cwd) — run terminal commands (requires approval)",
}


def _plan(goal: str, context: str = "") -> list[dict]:
      """
          Ask Gemini to decompose the goal into an ordered list of agent steps.

              Returns a list of dicts:
                      [{"step": int, "agent": str, "instruction": str, "params": dict}, ...]
                          """
    registry_text = "\n".join(f"  - {k}: {v}" for k, v in _AGENT_REGISTRY.items())

    prompt = (
              f"You are a task-planning AI for an autonomous assistant.\n"
              f"Available agents:\n{registry_text}\n\n"
              f"Goal: {goal}\n"
              f"{'Prior context: ' + context if context else ''}\n\n"
              f"Break the goal into {MAX_STEPS} or fewer concrete steps.\n"
              f"Return ONLY a JSON array, no explanation:\n"
              f"[\n"
              f"  {{\"step\": 1, \"agent\": \"<agent_name>\", \"instruction\": \"<what to do>\","
              f" \"params\": {{<key value pairs for the agent call>}}}},\n"
              f"  ...\n"
              f"]\n"
              f"Use only agents from the list above. Params must match each agent's signature."
    )

    gem = _gemini(PLANNER_MODEL)
    raw = gem.generate(prompt)
    raw = _strip_fences(raw)
    # extract JSON array
    arr_m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not arr_m:
              raise ValueError(f"Planner returned non-JSON: {raw[:300]}")
    return json.loads(arr_m.group(0))


# ---------------------------------------------------------------------------
# Dispatcher  —  routes a planned step to the correct sub-agent
# ---------------------------------------------------------------------------

def _dispatch(step: dict) -> Any:
      agent = step.get("agent", "").lower()
    p     = step.get("params", {})

    if agent == "coding":
              return coding_agent(
                  task      = p.get("task", step.get("instruction", "")),
                  language  = p.get("language", "python"),
                  save_path = p.get("save_path", ""),
    )
elif agent == "research":
        return research_agent(
                      topic = p.get("topic", step.get("instruction", "")),
                      depth = p.get("depth", "detailed"),
        )
elif agent == "file":
        return file_agent(
                      action  = p.get("action", "read"),
                      path    = p.get("path", ""),
                      content = p.get("content", ""),
        )
elif agent == "shell":
        return shell_agent(
                      command          = p.get("command", ""),
                      cwd              = p.get("cwd", ""),
                      require_approval = p.get("require_approval", True),
        )
else:
        return {"success": False, "result": f"Unknown agent: {agent}"}


# ---------------------------------------------------------------------------
# Main entry point  —  run_autonomous_agent()
# ---------------------------------------------------------------------------

def run_autonomous_agent(goal: str, mode: str = "auto") -> str:
      """
          High-level entry point for Jarvis to invoke autonomous work.

              mode:
                      "auto"     — Planner decomposes + dispatches automatically
                              "coding"   — Route directly to coding agent
                                      "research" — Route directly to research agent
                                              "file"     — Route directly to file agent
                                                      "shell"    — Route directly to shell agent

                                                          Returns a natural-language summary string for Jarvis to speak aloud.
                                                              """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []
    steps_log:     list[dict] = []

    try:
              # --- Direct mode shortcuts ---
              if mode == "coding":
                            result = coding_agent(task=goal)
                            if result["success"]:
                                              return (
                                                                    f"I've written the code, sir. It's saved to {result['file']}. "
                                                                    f"{result['explanation']}"
                                              )
                                          return f"The coding agent encountered an issue, sir: {result.get('explanation', 'Unknown error.')}"

        if mode == "research":
                      result = research_agent(topic=goal)
            if result["success"]:
                              kp = "; ".join(result["key_points"][:3])
                              return (
                                  f"Research complete, sir. Report saved to {result['file']}. "
                                  f"Key points: {kp}."
                              )
                          return "Research agent encountered an issue, sir."

        if mode == "shell":
                      result = shell_agent(command=goal)
            if result["success"]:
                              out = result["stdout"][:300] or "No output."
                              return f"Command executed, sir. Output: {out}"
                          return f"Shell command failed, sir: {result['stderr'][:200]}"

        # --- Auto planning mode ---
        print(f"[AutonomousAgent] Planning goal: {goal}")
        plan = _plan(goal)
        print(f"[AutonomousAgent] Plan has {len(plan)} steps.")

        context_so_far = ""
        for step in plan:
                      step_num   = step.get("step", "?")
            instruction = step.get("instruction", "")
            agent_name  = step.get("agent", "unknown")

            print(f"[AutonomousAgent] Step {step_num} ({agent_name}): {instruction}")

            result     = _dispatch(step)
            success    = result.get("success", False)
            step_entry = {
                              "step":        step_num,
                              "agent":       agent_name,
                              "instruction": instruction,
                              "success":     success,
                              "result_key":  list(result.keys()),
            }
            steps_log.append(step_entry)

            # Build running context for planner feedback
            if agent_name == "coding" and success:
                              snippet = (result.get("explanation") or "")[:200]
                              context_so_far += f"\nStep {step_num} (coding): {snippet}"
                              summary_lines.append(f"Step {step_num}: Code written → {result.get('file', '')}")
elif agent_name == "research" and success:
                pts = "; ".join((result.get("key_points") or [])[:2])
                context_so_far += f"\nStep {step_num} (research): {pts}"
                summary_lines.append(f"Step {step_num}: Research done → {result.get('file', '')}")
elif agent_name == "file" and success:
                res_text = (result.get("result") or "")[:150]
                context_so_far += f"\nStep {step_num} (file): {res_text}"
                summary_lines.append(f"Step {step_num}: File op OK → {res_text[:80]}")
elif agent_name == "shell" and success:
                out = (result.get("stdout") or "")[:150]
                context_so_far += f"\nStep {step_num} (shell): {out}"
                summary_lines.append(f"Step {step_num}: Shell OK → {out[:80]}")
else:
                err = result.get("stderr") or result.get("result") or "Unknown error"
                summary_lines.append(f"Step {step_num}: FAILED — {err[:100]}")
                context_so_far += f"\nStep {step_num} FAILED: {err[:100]}"

        _log_task(goal=goal, steps=steps_log, outcome="\n".join(summary_lines))

        if not summary_lines:
                      return "The autonomous agent completed the task, sir, but produced no output."

        bullet_summary = "\n".join(f"  • {l}" for l in summary_lines)
        return (
                      f"Autonomous task complete, sir. Here's what I did:\n"
                      f"{bullet_summary}\n"
                      f"All work saved to {WORK_DIR}."
        )

except Exception as exc:
        _log_task(goal=goal, steps=steps_log, outcome=f"ERROR: {exc}")
        return f"The autonomous agent encountered an error, sir: {exc}"


# ---------------------------------------------------------------------------
# Skill registry  —  exposes this module to Jarvis's tool router
# ---------------------------------------------------------------------------

SKILL_DESCRIPTION = (
      "Run a complex autonomous work task. Use for: writing / debugging / refactoring code in any language, "
      "deep research and reports, file creation and management, running terminal commands, or any "
      "multi-step goal that requires planning and sequential execution."
)

SKILL_PARAMS = {
      "goal": "The high-level work goal or instruction, in plain English.",
      "mode": (
                "Optional routing hint: 'auto' (default, Gemini plans steps), "
                "'coding' (go straight to code generation), "
                "'research' (go straight to research), "
                "'shell' (run a terminal command)."
      ),
}


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
      import argparse
    ap = argparse.ArgumentParser(description="Autonomous Agent CLI")
    ap.add_argument("goal", help="The work goal")
    ap.add_argument("--mode", default="auto", choices=["auto", "coding", "research", "shell", "file"])
    args = ap.parse_args()
    print("\n" + run_autonomous_agent(args.goal, mode=args.mode))
