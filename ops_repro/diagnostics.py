"""Render training metrics from completed or partial runs without loading a model."""
import argparse
import html
from pathlib import Path

from .common import read_jsonl, require


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def series(metrics, batches, datasets):
    """Produce global learning curves and dataset-level reward/teacher curves."""
    curves = [
        (key, [(row["updates"], row[key]) for row in metrics if key in row])
        for key in ("teacher_token_nll", "policy_token_loss", "gradient_norm",
                    "learning_rate", "max_logprob_drift", "teacher_lambda")
    ]
    curves.extend([
        ("rollout mean reward", [(row["batches"], row["mean_reward"]) for row in batches if "mean_reward" in row]),
        ("zero advantage group fraction", [(row["batches"], ratio(row["zero_advantage_groups"], row["groups"])) for row in batches]),
        ("active token fraction", [(row["batches"], ratio(row["active_tokens"], row["total_tokens"])) for row in batches if "active_tokens" in row]),
    ])
    for name in datasets:
        reward, inactive, all_wrong, all_correct = [], [], [], []
        nll, reweighted, deviation = [], [], []
        for row in batches:
            entry = row.get("by_dataset", {}).get(name)
            if entry:
                reward.append((row["batches"], ratio(entry["successes"], entry["episodes"])))
                inactive.append((row["batches"], ratio(entry["zero_advantage_groups"], entry["groups"])))
                all_wrong.append((row["batches"], ratio(entry["all_wrong_groups"], entry["groups"])))
                all_correct.append((row["batches"], ratio(entry["all_correct_groups"], entry["groups"])))
        for row in metrics:
            entry = row.get("by_dataset", {}).get(name, {})
            if "teacher_token_nll" in entry:
                nll.append((row["updates"], entry["teacher_token_nll"]))
            if "reweighted_tokens" in entry:
                reweighted.append((row["updates"], ratio(entry["reweighted_tokens"], entry["active_tokens"])))
                deviation.append((row["updates"], entry["mean_abs_weight_delta"]))
        curves.extend((f"{name} / {label}", points) for label, points in [
            ("teacher token NLL", nll), ("rollout reward", reward), ("zero advantage fraction", inactive),
            ("all wrong group fraction", all_wrong), ("all correct group fraction", all_correct),
            ("teacher reweighted fraction", reweighted), ("mean absolute weight deviation", deviation)])
    valid = [(label, [(x, y) for x, y in points if y is not None]) for label, points in curves]
    return [(label, points) for label, points in valid if points]


def chart(label, points):
    left, right = points[0][0], points[-1][0]
    lo, hi = min(y for _, y in points), max(y for _, y in points)
    xy = [(60+670*(x-left)/max(1,right-left), 110-90*(y-lo)/(hi-lo or 1)) for x,y in points]
    path = " ".join(f"{x:.1f},{y:.1f}" for x,y in xy)
    marker = f'<circle cx="{xy[0][0]}" cy="{xy[0][1]}" r="3" fill="#1455a5"/>' if len(xy)==1 else ""
    axis = "采样批次" if any(word in label for word in ("rollout", "zero advantage", "active token fraction", "group fraction")) else "优化步"
    return (f"<section><h2>{html.escape(label)}</h2>"
            f'<svg viewBox="0 0 755 145" role="img" aria-label="{html.escape(label, quote=True)}">'
            f'<path d="M60 15V115H735" fill="none" stroke="#687383"/>'
            f'<polyline points="{path}" fill="none" stroke="#1455a5" stroke-width="2"/>{marker}'
            f'<text x="0" y="23">{hi:.4g}</text><text x="0" y="111">{lo:.4g}</text>'
            f'<text x="60" y="135">{left}</text><text x="640" y="135">{axis} {right}</text></svg>'
            f"<p>{len(points)} 个记录点 · 首值 {points[0][1]:.5g} · 末值 {points[-1][1]:.5g}</p></section>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    def rows(name):
        path = args.run / name
        return read_jsonl(path) if path.is_file() else []
    metrics, batches = rows("metrics.jsonl"), rows("batch_metrics.jsonl")
    curves = series(metrics, batches, ("cloudops", "openrca", "cujbench", "itbench_aa"))
    require(curves, "No recorded training metrics to plot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        out.write("""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>SCVD / RLSD 训练诊断</title>
<style>body{max-width:900px;margin:32px auto;padding:0 16px;font:16px system-ui}svg{width:100%;font-size:12px}section{border-top:1px solid #dbe2ed}</style>
<h1>训练诊断</h1>
<p>SCVD NLL 描述训练样本拟合。RLSD policy loss 随在线批次变化，需结合验证集、奖励、零优势组和激活 token 一起看。训练奖励和教师调权幅度属于机制诊断，方法收益由同协议的 held-out 对照验证。缺失分组没有点。</p>
""")
        out.write("\n".join(chart(label, points) for label, points in curves))
        out.write("</html>\n")


if __name__ == "__main__":
    main()
