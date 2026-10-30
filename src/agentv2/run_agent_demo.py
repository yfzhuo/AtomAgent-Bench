import os
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
          'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)

import sys
import argparse
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from openai import OpenAI
from agentv2.vision_agent import VisionModelingAgent
from evaluation.evaluator import AtomWorldEvaluator

def build_evaluator(action_name, data_dir=None, input_cifs_file="input_cifs.hdf5"):
    if data_dir is None:
        data_dir = os.path.join(SRC_DIR, "data")
    evaluator = AtomWorldEvaluator(
        model=None,
        data_folder=data_dir,
        action_name=action_name,
        input_cifs_filename=input_cifs_file,
    )
    return evaluator

def score_with_evaluator(evaluator, row, agent_cif):
    stats = evaluator._initialize_stats()
    wrapped = f"<cif>\n{agent_cif}\n</cif>"
    return evaluator._process_single_output(row, wrapped, stats)

def print_score(result):
    print("\n========== Score ==========")
    if result.get('is_error'):
        print(f"[FAIL] wrong_type = {result.get('wrong_type')}")
        print("  说明: 结构未通过校验（格式/解析/原子数/结构不匹配）。")
    else:
        rmsd = result['rmsd']
        max_diff = result['max_diff']
        print(f"[PASS] structures matched")
        print(f"  RMSD     = {rmsd}")
        print(f"  Max Diff = {max_diff}")
        if rmsd == 0 and max_diff == 0:
            print("  -> 完全精确匹配 (exact match)")

def write_run_txt(run_dir, args, sample_idx, rep, task_prompt, agent, result, cif_out):
    lines = []
    lines.append(f"model   : {args.model}")
    lines.append(f"action  : {args.action}   sample: {sample_idx}   rep: {rep}   "
                 f"group: {args.group}")
    lines.append(f"vision  : {'OFF (ablation)' if args.no_vision else 'ON'}")
    lines.append(f"view    : {'OFF (tool removed)' if args.no_view else 'ON'}")
    lines.append(f"task    : {task_prompt}")
    lines.append("=" * 60)
    for rec in agent.trace:
        if rec.get("type") == "init":
            lines.append(f"[init] n_atoms={rec['init_n_atoms']} "
                         f"image={rec.get('image_file')}")
            continue
        lines.append(f"\n[Step {rec['step']}]")
        if rec.get("model_content"):
            lines.append(f"  model said: {rec['model_content']}")
        if rec.get("model_tool_calls"):
            lines.append(f"  native tool_calls: {rec['model_tool_calls']}")
        for a in rec.get("actions", []):
            score_str = f" | score={a.get('score')}" if "score" in a else ""
            lines.append(f"  -> [{a.get('channel')}] {a.get('name')} "
                         f"{a.get('args', '')} | {a.get('result', a.get('note',''))} "
                         f"| n_atoms={a.get('n_atoms')} | img={a.get('image_file')}"
                         f"{score_str}")
    lines.append("\n" + "=" * 60)
    lines.append(f"usage  : {agent.usage}")
    lines.append(f"repairs: attempts={agent.usage.get('repair_attempts')} "
                 f"history={agent.usage.get('repair_history')}")
    if result.get("is_error"):
        lines.append(f"SCORE: FAIL ({result.get('wrong_type')})")
    else:
        lines.append(f"SCORE: PASS  RMSD={result['rmsd']}  "
                     f"MaxDiff={result['max_diff']}")
    lines.append("\n----- Exported CIF -----")
    lines.append(cif_out)

    lines = [str(x) if x is not None else "<<None>>" for x in lines]
    with open(os.path.join(run_dir, "run.txt"), "w") as f:
        f.write("\n".join(lines))

def run_one(client, args, evaluator, sample_idx, batch_dir, rep=0):
    import json

    row = evaluator.data.iloc[sample_idx]
    init_cif = row["input_cif"]
    task_prompt = row["action_prompt"]

    run_dir = os.path.join(batch_dir, f"s{sample_idx}_r{rep}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n############ sample {sample_idx} rep {rep} ############")
    print(f"[Log] run dir: {run_dir}")

    def score_fn(cif_str):
        res = score_with_evaluator(evaluator, row, cif_str)
        if res.get("is_error"):
            wt = res.get("wrong_type")
            return False, wt, f"structure rejected by evaluator ({wt})"
        return True, None, (f"matched (rmsd={res.get('rmsd')}, "
                            f"max_diff={res.get('max_diff')})")

    agent = VisionModelingAgent(client, args.model,
                                max_steps=args.max_steps, verbose=True,
                                allow_imaging=not args.no_vision,
                                use_view=not args.no_view)
    cif_out = agent.run(init_cif, task_prompt, log_dir=run_dir,
                        score_fn=score_fn, max_repairs=args.max_repairs)
    if cif_out is not None and not isinstance(cif_out, str):
        cif_out = cif_out.to(fmt="cif")

    print("\n========== Exported CIF ==========")
    print(repr(cif_out))
    if cif_out is None:
        print("[WARN] agent.run() 返回 None —— agent 没导出 CIF。")

    if cif_out is None:
        result = {"is_error": True, "wrong_type": "NoCIFExported"}
        print("\n========== Score ==========")
        print("[SKIP] cif_out 是 None,不送进 evaluator")
    else:
        result = score_with_evaluator(evaluator, row, cif_out)
        print_score(result)

    print(f"[Repair] attempts={agent.usage.get('repair_attempts')} "
          f"history={agent.usage.get('repair_history')}")

    archive = {
        "model": args.model, "action": args.action, "sample": sample_idx,
        "repeat": rep, "group": args.group,
        "allow_imaging": not args.no_vision,
        "use_view": not args.no_view,
        "task_prompt": task_prompt,
        "input_cif": init_cif,
        "target_cif": row.get("output_cif"),
        "exported_cif": cif_out,
        "score": {k: v for k, v in result.items()
                  if k in ("is_error", "wrong_type", "rmsd", "max_diff")},
        "repair_attempts": agent.usage.get("repair_attempts"),
        "repair_history": agent.usage.get("repair_history"),
        "usage": agent.usage,
        "trace": agent.trace,
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)

    write_run_txt(run_dir, args, sample_idx, rep, task_prompt, agent, result, cif_out)

    print(f"[Log] 单样本存档: {run_dir}/run.json 和 run.txt")

    passed = (not result.get("is_error"))
    return {
        "sample": sample_idx,
        "repeat": rep,
        "group": args.group,
        "passed": passed,
        "wrong_type": result.get("wrong_type"),
        "rmsd": result.get("rmsd"),
        "max_diff": result.get("max_diff"),
        "exact": bool(passed and result.get("rmsd") == 0
                      and result.get("max_diff") == 0),
        "repair_attempts": agent.usage.get("repair_attempts"),
        "repair_history": agent.usage.get("repair_history"),
        "usage": agent.usage,
        "run_dir": run_dir,
    }

def compute_stats(summaries, repeat, max_repairs):
    n = len(summaries)
    if n == 0:
        return None

    single = sum(1 for s in summaries
                 if s.get("passed") and (s.get("repair_attempts") or 0) == 0)
    multi = sum(1 for s in summaries
                if s.get("passed") and (s.get("repair_attempts") or 0) >= 1)
    fail = sum(1 for s in summaries if not s.get("passed"))
    exact = sum(1 for s in summaries if s.get("exact"))

    K = max_repairs
    cum_counts = [0] * (K + 1)
    for s in summaries:
        if not s.get("passed"):
            continue
        rp = min(s.get("repair_attempts") or 0, K)
        cum_counts[rp] += 1
    for i in range(1, K + 1):
        cum_counts[i] += cum_counts[i - 1]
    cumulative_pass_at_k = [c / n for c in cum_counts]

    per_sample = defaultdict(dict)
    for s in summaries:
        per_sample[s["sample"]][s.get("repeat", 0)] = bool(s.get("passed"))
    R = repeat if (repeat and repeat > 0) else \
        max((max(d.keys()) + 1 for d in per_sample.values()), default=1)
    n_samples = len(per_sample)
    pass_at_k_by_rep = []
    for k in range(1, R + 1):
        cnt = sum(1 for reps in per_sample.values()
                  if any(reps.get(r, False) for r in range(k)))
        pass_at_k_by_rep.append(cnt / n_samples if n_samples else 0.0)

    return {
        "n_runs": n,
        "n_samples": n_samples,
        "n_single": single,
        "n_multi": multi,
        "n_fail": fail,
        "n_exact": exact,
        "single_rate": single / n,
        "multi_rate": multi / n,
        "fail_rate": fail / n,
        "exact_rate": exact / n,
        "cumulative_k": list(range(0, K + 1)),
        "cumulative_pass_at_k": cumulative_pass_at_k,
        "repeat_k": list(range(1, R + 1)),
        "pass_at_k_by_rep": pass_at_k_by_rep,
    }
def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[WARN] 未能导入 matplotlib，跳过画图:{e}")
        print("       安装:pip install --user --break-system-packages matplotlib")
        return None

def make_stacked_plot(batch_dir, group_label, action_label, stats, max_repairs):
    plt = _get_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(4, 6))

    single = stats["single_rate"] * 100
    multi = stats["multi_rate"] * 100
    fail = stats["fail_rate"] * 100

    x = [action_label]
    ax.bar(x, [single], color="#54A24B", width=0.5,
           label=f"single-step (first try): {single:.1f}%")
    ax.bar(x, [multi], bottom=[single], color="#F58518", width=0.5,
           label=f"multi-step (with repair): {multi:.1f}%")
    ax.bar(x, [fail], bottom=[single + multi], color="#E45756", width=0.5,
           label=f"fail (out of budget): {fail:.1f}%")

    if single > 3:
        ax.text(0, single / 2, f"{single:.1f}%",
                ha="center", va="center", color="white", fontsize=12)
    if multi > 3:
        ax.text(0, single + multi / 2, f"{multi:.1f}%",
                ha="center", va="center", color="white", fontsize=12)
    if fail > 3:
        ax.text(0, single + multi + fail / 2, f"{fail:.1f}%",
                ha="center", va="center", color="white", fontsize=12)

    ax.set_ylim(0, 100)
    ax.set_ylabel("proportion (%)")
    title = f"Outcome breakdown — {action_label}\n" \
            f"(runs={stats['n_runs']}, max_repairs={max_repairs})"
    if group_label:
        title += f"   group={group_label}"
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 1.0), frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(batch_dir, "plot_stacked.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Log] 堆叠图已保存: {out}")

def make_cumulative_plot(batch_dir, group_label, action_label, stats, max_repairs):
    plt = _get_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))

    xs = stats["cumulative_k"]
    ys = [p * 100 for p in stats["cumulative_pass_at_k"]]

    ax.plot(xs, ys, marker="o", linewidth=2, color="#4C78A8")
    for x, y in zip(xs, ys):
        ax.text(x, y + 2, f"{y:.1f}%", ha="center", fontsize=9)

    if len(xs) >= 2:
        auc = sum((ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)) \
              / (100 * (len(xs) - 1))
    else:
        auc = ys[0] / 100 if ys else 0.0

    ax.set_xlabel("repair budget k")
    ax.set_ylabel("cumulative pass rate (%)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 110)
    title = f"Cumulative pass rate vs repair budget — {action_label}\n" \
            f"(runs={stats['n_runs']}, AUC={auc:.3f})"
    if group_label:
        title += f"   group={group_label}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(batch_dir, "plot_cumulative.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Log] 累计曲线已保存: {out}   AUC={auc:.4f}")

    stats["auc"] = auc

def main():
    import json
    from datetime import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="qwen3-vl-plus")
    ap.add_argument("-a", "--action", default="add_atom_action")
    ap.add_argument("--sample", type=int, default=0, help="起始样本索引")
    ap.add_argument("--n", type=int, default=1,
                    help="连续跑多少个样本（从 --sample 开始）")
    ap.add_argument("--samples", type=str, default=None,
                    help="逗号分隔的离散样本下标，优先于 --sample/--n")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每个样本独立重跑几次（画 pass@k 曲线用；师兄示例=5）")
    ap.add_argument("--group", default=None,
                    help="难度分组标签(hard/medium/simple)，写入存档 & 图表标题")
    ap.add_argument("--data-dir", default=None,
                    help="数据目录；不填用 src/data(v1: csv+hdf5)")
    ap.add_argument("--base-url",
                    default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-repairs", type=int, default=5,
                    help="单次运行内 submit FAIL 后最多再让 agent 修几次；0=不重试")
    ap.add_argument("--no-vision", action="store_true",
                    help="禁用渲图（消融，view 工具仍在）")
    ap.add_argument("--no-view", action="store_true",
                    help="移除 view 工具（更彻底的消融）")
    ap.add_argument("--no-plot", action="store_true",
                    help="跑完不自动画 plot.png")
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: 请先设置环境变量 DASHSCOPE_API_KEY (或 OPENAI_API_KEY)")
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    if args.samples:
        sample_indices = [int(x) for x in args.samples.split(",") if x.strip() != ""]
    else:
        sample_indices = list(range(args.sample, args.sample + args.n))

    if not sample_indices:
        print("ERROR: 没有要跑的样本下标（检查 --samples / --sample / --n）")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.samples:
        idx_tag = "samples_" + "-".join(str(i) for i in sample_indices)
        idx_tag = idx_tag[:60]
    else:
        idx_tag = f"s{args.sample}_n{args.n}"
    grp_tag = f"_{args.group}" if args.group else ""
    rep_tag = f"_x{args.repeat}" if args.repeat > 1 else ""
    batch_dir = os.path.join(
        SRC_DIR, "agentv2", "runs",
        f"{ts}_{args.model}_{args.action}{grp_tag}_{idx_tag}{rep_tag}")
    os.makedirs(batch_dir, exist_ok=True)
    print(f"[Log] batch dir: {batch_dir}")
    print(f"[Log] 计划样本下标: {sample_indices}  x{args.repeat} 次  group={args.group}")

    evaluator = build_evaluator(args.action, data_dir=args.data_dir)
    n_total = len(evaluator.data)

    summaries = []
    for idx in sample_indices:
        if idx < 0 or idx >= n_total:
            print(f"[WARN] sample {idx} 越界（共 {n_total} 条），跳过。")
            continue
        for rep in range(args.repeat):
            try:
                summaries.append(
                    run_one(client, args, evaluator, idx, batch_dir, rep))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[ERROR] sample {idx} rep{rep} 异常: {e}")
                print(tb)
                err_dir = os.path.join(batch_dir, f"s{idx}_r{rep}")
                os.makedirs(err_dir, exist_ok=True)
                try:
                    with open(os.path.join(err_dir, "runner_error.txt"), "w") as f:
                        f.write(tb)
                except Exception:
                    pass
                summaries.append({
                    "sample": idx,
                    "repeat": rep,
                    "group": args.group,
                    "passed": False,
                    "wrong_type": f"RunnerException:{type(e).__name__}:{e}",
                    "rmsd": None,
                    "max_diff": None,
                    "exact": False,
                    "repair_attempts": 0,
                    "repair_history": [f"RunnerException:{type(e).__name__}"],
                    "usage": {},
                    "run_dir": err_dir,
                })
    n_pass = sum(1 for s in summaries if s.get("passed"))
    n_exact = sum(1 for s in summaries if s.get("exact"))
    n_run = len(summaries)
    stats = compute_stats(summaries, args.repeat, args.max_repairs)
    batch = {
        "model": args.model, "action": args.action,
        "group": args.group, "repeat": args.repeat,
        "data_dir": args.data_dir,
        "no_vision": args.no_vision,
        "no_view": args.no_view,
        "sample_start": args.sample, "n": args.n,
        "sample_indices": sample_indices,
        "n_run": n_run, "n_pass": n_pass, "n_exact": n_exact,
        "stats": stats,
        "results": summaries,
    }
    with open(os.path.join(batch_dir, "batch.json"), "w") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)

    lines = [f"model : {args.model}",
             f"action: {args.action}",
             f"group : {args.group}   repeat: {args.repeat}",
             f"data  : {args.data_dir or os.path.join(SRC_DIR, 'data')}",
             f"vision: {'OFF' if args.no_vision else 'ON'}   "
             f"view: {'OFF (removed)' if args.no_view else 'ON'}",
             f"samples: {sample_indices}",
             f"repair: max_repairs={args.max_repairs}",
             "=" * 60]
    for s in summaries:
        rp = s.get("repair_attempts")
        tag_id = f"s{s['sample']}_r{s.get('repeat')}"
        if s.get("passed"):
            tag = "EXACT" if s.get("exact") else "PASS"
            lines.append(f"  {tag_id:<10} {tag:<5} "
                         f"rmsd={s.get('rmsd')} max_diff={s.get('max_diff')} "
                         f"repairs={rp}")
        else:
            lines.append(f"  {tag_id:<10} FAIL  ({s.get('wrong_type')}) "
                         f"repairs={rp} history={s.get('repair_history')}")
    lines.append("=" * 60)
    lines.append(f"PASSED {n_pass}/{n_run}   (exact match: {n_exact}/{n_run})")
    if stats:
        lines.append(f"single-step (first try) : "
                     f"{stats['single_rate']*100:6.2f}%  ({stats['n_single']}/{stats['n_runs']})")
        lines.append(f"multi-step  (with repair): "
                     f"{stats['multi_rate']*100:6.2f}%  ({stats['n_multi']}/{stats['n_runs']})")
        lines.append(f"fail        (out of budget): "
                     f"{stats['fail_rate']*100:6.2f}%  ({stats['n_fail']}/{stats['n_runs']})")
        lines.append(f"exact-match (rmsd=0)     : "
                     f"{stats['exact_rate']*100:6.2f}%  ({stats['n_exact']}/{stats['n_runs']})")
        lines.append("-" * 60)
        lines.append(f"cumulative pass rate by repair budget k:")
        lines.append(f"  k       : {stats['cumulative_k']}")
        lines.append(f"  pass(%) : "
                     f"{[round(p*100, 2) for p in stats['cumulative_pass_at_k']]}")
        if stats.get("auc") is not None:
            lines.append(f"  AUC (normalized [0,1]) : {stats['auc']:.4f}")
        if stats.get("pass_at_k_by_rep"):
            lines.append(f"pass@k over --repeat (k={stats['repeat_k']}): "
                         f"{[round(p*100, 2) for p in stats['pass_at_k_by_rep']]}")
    summary_txt = "\n".join(lines)
    with open(os.path.join(batch_dir, "summary.txt"), "w") as f:
        f.write(summary_txt)

    print("\n" + "=" * 60)
    print(summary_txt)
    print(f"\n[Log] 批次存档: {batch_dir}/batch.json 和 summary.txt")

    if not args.no_plot and stats:
        make_stacked_plot(batch_dir, args.group or "", args.action,
                          stats, args.max_repairs)
        make_cumulative_plot(batch_dir, args.group or "", args.action,
                             stats, args.max_repairs)

if __name__ == "__main__":
    main()
