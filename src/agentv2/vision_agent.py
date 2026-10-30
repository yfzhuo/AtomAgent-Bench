import os
import json
import re
import base64

from agentv2.structure_env import StructureEnvironment
from agentv2.tools_schema import TOOLS

SYSTEM_PROMPT = """You are a structure modeling agent.
You have exactly two capabilities:
  1) `view` — render the current or input structure to an image to inspect it
     (you may add atom-index labels or highlight atoms, and pick the viewpoint).
  2) `run_python` — write arbitrary Python to transform the structure. A pymatgen
     `structure` is preloaded; whatever `structure` holds after your code runs
     becomes the new state. `input_cif` (str), the original structure, is also
     available (read-only). To start over, reload it with
     `structure = Structure.from_str(input_cif, fmt="cif")`.
In run_python you may freely import and use scientific libraries such as
pymatgen, ase, numpy, etc.
You can also use run_python with print() to inspect sites, distances, lattice, etc.
There is NO pre-made editing tool: you must write the code yourself for whatever
the task requires. When the structure is correct, call `submit`.
Always respond with a tool call, not plain prose."""

SYSTEM_PROMPT_NOVIEW = """You are a structure modeling agent.
You have exactly ONE capability to edit the structure:
  - `run_python` — write arbitrary Python to transform the structure. A pymatgen
    `structure` is preloaded; whatever `structure` holds after your code runs
    becomes the new state. `input_cif` (str), the original structure, is also
    available (read-only). To start over, reload it with
    `structure = Structure.from_str(input_cif, fmt="cif")`.
In run_python you may freely import and use scientific libraries such as
pymatgen, ase, numpy, etc.
You CANNOT view or render the structure — work from the CIF / coordinates.
You can use run_python with print() to inspect sites, distances, lattice, etc.
When the structure is correct, call `submit`."""

def _image_msg(text, img_b64):
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}


def _save_b64_png(b64, path):
    try:
        with open(path, "wb")as f:
            f.write(base64.b64decode(b64))
        return path
    except Exception:
        return None


class VisionModelingAgent:
    def __init__(self, client, model, max_steps=15, verbose=True,
                 allow_imaging=True, use_view=True):
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.use_view = use_view
        self.allow_imaging = allow_imaging and use_view
        self.tools = TOOLS if use_view else [
            t for t in TOOLS if t["function"]["name"] != "view"]
        self.trace = []
        self.usage = {}
        self.repair_attempts = 0
        self.repair_history = []

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _create(self, messages):
        try:
            return self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self.tools, tool_choice="required",
                extra_body={"enable_thinking": False})
        except Exception as e:
            self._log(f"[warn] tool_choice='required' failed: "
                      f"{type(e).__name__}: {e}; fallback 'auto'")
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    tools=self.tools, tool_choice="auto",
                    extra_body={"enable_thinking": False})
            except Exception as e2:
                self._log(f"[FATAL] _create totally failed: "
                          f"{type(e2).__name__}: {e2}")
                raise

    @staticmethod
    def _parse_text_toolcall(content):
        if not content:
            return None
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
        raw = m.group(1) if m else None
        if raw is None:
            m = re.search(r"(\{.*\})", content, re.S)
            raw = m.group(1) if m else None
        if raw is None:
            return None
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        name = obj.get("tool") or obj.get("name")
        if not name:
            return None
        args = obj.get("args", obj)
        if name == "submit":
            return name, {}
        if name == "view":
            keep = {k: args[k] for k in
                    ("target", "axis", "azim", "elev", "show_indices", "highlight")
                    if k in args}
            return name, keep
        if name == "run_python":
            code = args.get("code") or args.get("source") or args.get("python")
            if not code:
                return None
            return name, {"code": str(code)}
        return None

    def _dispatch(self, env, name: str, kwargs: dict):
        if name == "submit":
            return None, True

        base = {"image_base64": None, "stdout": "",
                "n_atoms": len(env.structure),
                "counts": {"view": env.view_calls, "code": env.code_calls}}

        if name == "view" and not self.use_view:
            return {"ok": False,
                    "message": "view tool is disabled in this run", **base}, False

        fn = getattr(env, name, None)
        if fn is None:
            return {"ok": False, "message": f"UnknownTool:{name}", **base}, False

        try:
            return fn(**kwargs), False

        except TypeError as e:
            msg = str(e)
            if ("unexpected keyword" in msg
                    or "missing" in msg
                    or "positional argument" in msg):
                return {"ok": False,
                        "message": f"BadArgs for {name}: {msg}", **base}, False
            if "not iterable" in msg or "not subscriptable" in msg:
                import traceback
                tb = traceback.format_exc()
                try:
                    with open("bool_bug_trace.txt", "a") as f:
                        f.write(f"\n=== tool={name} kwargs={kwargs} ===\n{tb}\n")
                except Exception:
                    pass
                return {"ok": False,
                        "message": f"EnvTypeError:{msg}", **base}, False
            return {"ok": False,
                    "message": f"TypeError:{msg}", **base}, False

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                with open("bool_bug_trace.txt", "a") as f:
                    f.write(f"\n=== tool={name} kwargs={kwargs} "
                            f"EXC={type(e).__name__} ===\n{tb}\n")
            except Exception:
                pass
            return {"ok": False,
                    "message": f"EnvError:{type(e).__name__}:{e}", **base}, False
    def _finish(self, env):
        total_steps = getattr(self, "_total_steps", None)
        if total_steps is None:
            total_steps = len([t for t in self.trace if t.get("step", 0) > 0])
        self.usage = {
            "steps": total_steps,
            "per_attempt_step_budget": self.max_steps,
            "attempts": getattr(self, "repair_attempts", 0) + 1,
            "view_calls": getattr(env, "view_calls", None),
            "code_calls": getattr(env, "code_calls", None),
            "n_atoms": len(env.structure),
            "repair_attempts": getattr(self, "repair_attempts", 0),
            "repair_history": getattr(self, "repair_history", []),
        }
        self._log(f"[Finish] {self.usage}")
        return env.structure
    def run(self, init_cif, task_prompt, log_dir=None,
            score_fn=None, max_repairs=0):
        env = StructureEnvironment(init_cif, allow_imaging=self.allow_imaging)
        self._log(f"[Task] {task_prompt}")
        self._log(f"[Init] {len(env.structure)} atoms")

        self.trace = []
        self.repair_attempts = 0
        self.repair_history = []
        img_idx = [0]
        total_step = 0

        def _save_img(b64, tag):
            if not (log_dir and b64):
                return None
            img_idx[0] += 1
            fname = f"img_{img_idx[0]:02d}_{tag}.png"
            _save_b64_png(b64, os.path.join(log_dir, fname))
            return fname

        init_img_b64 = None
        if self.allow_imaging:
            try:
                init_img_b64 = env._render(env.structure)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"[warn] init render failed: {type(e).__name__}: {e}")
                try:
                    with open("bool_bug_trace.txt", "a") as f:
                        f.write(f"\n=== INIT RENDER FAILED ===\n{tb}\n")
                except Exception:
                    pass
                init_img_b64 = None
        init_img_file = _save_img(init_img_b64, "init")
        self.trace.append({
            "step": 0, "type": "init",
            "task_prompt": task_prompt,
            "init_n_atoms": len(env.structure),
            "image_file": init_img_file,
        })

        sys_prompt = SYSTEM_PROMPT if self.use_view else SYSTEM_PROMPT_NOVIEW
        intro = (f"Task: {task_prompt}\n\nThe current structure has "
                 f"{len(env.structure)} atoms.")
        if init_img_b64:
            messages = [{"role": "system", "content": sys_prompt},
                        _image_msg(intro + " It is shown below.", init_img_b64)]
        else:
            messages = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": intro}]

        no_call_streak = 0

        def _record_and_feedback(rec, channel, name, kwargs, obs, step):
            img_file = _save_img(obs.get("image_base64"), f"s{step}_{name}")
            rec["actions"].append({
                "channel": channel, "name": name, "args": kwargs,
                "result": obs.get("message"),
                "stdout": obs.get("stdout"),
                "n_atoms": obs.get("n_atoms"),
                "counts": obs.get("counts"),
                "image_file": img_file})
            txt = f"[{name}] {obs.get('message')}. n_atoms={obs.get('n_atoms')}."
            if obs.get("stdout"):
                txt += f"\nstdout:\n{obs['stdout']}"
            if obs.get("image_base64"):
                return _image_msg(txt, obs["image_base64"])
            return {"role": "user", "content": txt}

        def _handle_submit():
            try:
                cif_now = env.export_cif()
            except Exception as e:
                wrong_type = f"ExportCifError:{type(e).__name__}"
                err_detail = f"{type(e).__name__}: {e}"
                self.repair_history.append(wrong_type)
                self._log(f"[Submit] export_cif FAILED: {err_detail}")

                if self.repair_attempts >= max_repairs:
                    self._log(f"[Submit] repair budget ({max_repairs}) exhausted "
                              f"after ExportCifError -> final FAIL")
                    return True, None

                self.repair_attempts += 1
                self._log(f"[Submit] ExportCifError; repair "
                          f"{self.repair_attempts}/{max_repairs}, asking agent to fix")
                tools_hint = "view / run_python" if self.use_view else "run_python"
                fb = (f"Your submission was FAILED because export_cif crashed.\n"
                      f"error = {err_detail}\n"
                      f"This usually means the current structure is invalid "
                      f"(e.g. 0 atoms after deleting everything, or corrupted "
                      f"sites). A NEW attempt starts now with a fresh "
                      f"{self.max_steps}-step budget. Inspect the current "
                      f"structure ({tools_hint}), rebuild it from `input_cif` "
                      f"if needed, and only then submit.")
                return False, {"role": "user", "content": fb}
            if score_fn is None:
                return True, None
            try:
                passed, wrong_type, msg = score_fn(cif_now)
            except Exception as e:
                passed, wrong_type, msg = False, f"ScoreFnError:{e}", str(e)
            self.repair_history.append(None if passed else wrong_type)

            if passed:
                self._log(f"[Submit] PASS (after {self.repair_attempts} repairs)")
                return True, None

            if self.repair_attempts >= max_repairs:
                self._log(f"[Submit] FAIL ({wrong_type}); repair budget "
                          f"({max_repairs}) exhausted -> final FAIL")
                return True, None

            self.repair_attempts += 1
            self._log(f"[Submit] FAIL ({wrong_type}); repair "
                      f"{self.repair_attempts}/{max_repairs}, asking agent to fix")
            tools_hint = "view / run_python" if self.use_view else "run_python"
            fb = (f"Your submission was FAILED.\n"
                  f"error_type = {wrong_type}\n"
                  f"A NEW attempt starts now with a fresh {self.max_steps}-step "
                  f"budget. Do NOT submit again immediately. First inspect the "
                  f"current structure ({tools_hint}) to find what is wrong, fix "
                  f"it, and only then submit.")
            return False, {"role": "user", "content": fb}

        outer_stop = False
        while not outer_stop:
            restart_attempt = False
            finished = False
            attempt_id = self.repair_attempts

            for inner_step in range(1, self.max_steps + 1):
                total_step += 1
                step = total_step

                resp = self._create(messages)
                msg = resp.choices[0].message
                print(f">>> content={msg.content!r}")
                print(f">>> tool_calls={getattr(msg, 'tool_calls', 'NO_ATTR')!r}")
                messages.append(msg)

                rec = {"step": step,
                       "inner_step": inner_step,
                       "attempt": attempt_id,
                       "model_content": msg.content,
                       "model_tool_calls": [
                           {"name": tc.function.name,
                            "arguments": tc.function.arguments}
                           for tc in (msg.tool_calls or [])],
                       "actions": []}

                if not msg.tool_calls:
                    parsed = self._parse_text_toolcall(msg.content)
                    if parsed is not None:
                        name, kwargs = parsed
                        no_call_streak = 0
                        self._log(f"[Step {step}/attempt{attempt_id} "
                                  f"inner{inner_step}] (text-fallback) {name} {kwargs}")
                        obs, is_submit = self._dispatch(env, name, kwargs)
                        if is_submit:
                            rec["actions"].append({"channel": "text-fallback",
                                                   "name": "submit", "args": {}})
                            done, fb = _handle_submit()
                            rec["actions"][-1]["score"] = (
                                self.repair_history[-1] if self.repair_history else None)
                            self.trace.append(rec)
                            if done:
                                finished = True
                                break
                            messages.append(fb)
                            restart_attempt = True
                            break
                        self._log(f"  -> {obs['message']} (n={obs.get('n_atoms')})")
                        fb = _record_and_feedback(rec, "text-fallback", name,
                                                  kwargs, obs, step)
                        self.trace.append(rec)
                        messages.append(fb)
                        continue

                    no_call_streak += 1
                    self._log(f"[Step {step}] (no tool call x{no_call_streak}) "
                              f"{(msg.content or '')[:100]}")
                    rec["actions"].append({"channel": "none",
                                           "note": f"no tool call x{no_call_streak}"})
                    self.trace.append(rec)
                    if no_call_streak >= 3:
                        self._log("[Abort] model refuses to call tools, stop early.")
                        finished = True
                        break
                    allowed = ("run_python / submit" if not self.use_view
                               else "view / run_python / submit")
                    messages.append({"role": "user",
                                     "content": "Do NOT explain. Respond with ONE tool "
                                                f"call now: {allowed}."})
                    continue

                no_call_streak = 0
                submitted_done = False
                pending_feedback = None
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        kwargs = json.loads(tc.function.arguments or "{}")
                    except Exception as e:
                        self._log(f"[Step {step}] bad JSON args for {name}: "
                                  f"{e}; raw={tc.function.arguments!r}")
                        kwargs = {}

                    self._log(f"[Step {step}/attempt{attempt_id} "
                              f"inner{inner_step}] (native) {name} {kwargs}")
                    obs, is_submit = self._dispatch(env, name, kwargs)

                    if is_submit:
                        rec["actions"].append({"channel": "native",
                                               "name": "submit", "args": {}})
                        done, fb = _handle_submit()
                        rec["actions"][-1]["score"] = (
                            self.repair_history[-1] if self.repair_history else None)
                        if score_fn is None:
                            tool_reply = "submitted"
                        elif done and (not self.repair_history
                                       or self.repair_history[-1] is None):
                            tool_reply = "submitted; result=PASS"
                        else:
                            tool_reply = (f"submitted; result=FAIL "
                                          f"({self.repair_history[-1]})")
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": tool_reply})
                        if done:
                            submitted_done = True
                            continue
                        else:
                            pending_feedback = fb
                            continue

                    self._log(f"  -> {obs['message']} (n={obs.get('n_atoms')})")
                    img_file = _save_img(obs.get("image_base64"), f"s{step}_{name}")
                    rec["actions"].append({
                        "channel": "native", "name": name, "args": kwargs,
                        "result": obs.get("message"),
                        "stdout": obs.get("stdout"),
                        "n_atoms": obs.get("n_atoms"),
                        "counts": obs.get("counts"),
                        "image_file": img_file})

                    txt = (f"[{name}] {obs.get('message')}. "
                           f"n_atoms={obs.get('n_atoms')}.")
                    if obs.get("stdout"):
                        txt += f"\nstdout:\n{obs['stdout']}"
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": txt})
                    if obs.get("image_base64"):
                        messages.append(_image_msg(f"[{name}] rendered image above.",
                                                   obs["image_base64"]))

                self.trace.append(rec)
                if submitted_done:
                    finished = True
                    break
                if pending_feedback is not None:
                    messages.append(pending_feedback)
                    restart_attempt = True
                    break

            else:
                self._log(f"[Attempt {attempt_id}] step budget "
                          f"({self.max_steps}) exhausted without submit.")
                if self.repair_attempts < max_repairs:
                    self.repair_attempts += 1
                    self.repair_history.append("StepBudgetExhausted")
                    tools_hint = "view / run_python" if self.use_view else "run_python"
                    messages.append({"role": "user", "content": (
                        f"You did not submit within the {self.max_steps}-step "
                        f"budget for this attempt. A NEW attempt starts now with "
                        f"a fresh {self.max_steps}-step budget. Plan concisely "
                        f"and use {tools_hint}; submit as soon as the structure "
                        f"is correct.")})
                    continue
                else:
                    self.repair_history.append("StepBudgetExhausted")
                    self._log("[Stop] max_steps reached in final attempt.")
                    outer_stop = True
                    continue

            if finished:
                outer_stop = True
            elif restart_attempt:
                continue
            else:
                outer_stop = True

        self._total_steps = total_step
        return self._finish(env)
