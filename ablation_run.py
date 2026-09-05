"""消融试验驱动器（审稿人混杂因素分解方案）：16 路并行跑 train.py，
日志写到 output/ablation/{dir}/train.log（--no-save，无需 checkpoint）。

网格（β = 主试验协议 {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}，a ∈ {1, 6, 12}，5 seeds）：
- mnist / imagenet100（MLP）：mlp(Base) 1 + ncm 1 + ncmo 3 + ceb 6 +
  ceb-energy 6 + opb(GPB-L) 18 + opb --energy-classifier(GPB) 18 +
  opb-free-scale --energy-classifier(OPB-FS) 18 = 71 配置
- housing（MLP）：mlp 1 + ceb 6 + ceb-tied 6 + opb(EPB-L) 18 +
  opb --tied-head(EPB) 18 + opb-free-scale --tied-head(EPB-FS) 18 = 67 配置

参考模型（mlp/ceb/opb×2）一并重训到 output/ablation 内，保证同 seed 配对
（seed 0–4 与主表同协议）与自包含。表格与配对 CI 由 ablation_table.py 生成。

用法：
    python ablation_run.py --shards 16     # 全部 209 配置
    python ablation_run.py --shards 1 --only 'mnist_mlp_ncm'   # 单个配置复跑
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "ablation"

BETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ANCHORS = [1, 6, 12]


def configs():
    """→ [(task, dataset, model, beta, anchor, extra_flags)]。"""
    out = []
    for task, ds in (("mnist", "mnist"), ("imagenet100", "imagenet100")):
        out.append((task, ds, "mlp", None, None, []))
        out.append((task, ds, "ncm", None, None, []))
        for a in ANCHORS:
            out.append((task, ds, "ncmo", None, a, []))
        for b in BETAS:
            out.append((task, ds, "ceb", b, None, []))
            out.append((task, ds, "ceb-energy", b, None, []))
        for b in BETAS:
            for a in ANCHORS:
                out.append((task, ds, "opb", b, a, []))  # GPB-L / EPB-L
                out.append((task, ds, "opb", b, a, ["--energy-classifier"]))  # GPB
                out.append((task, ds, "opb-free-scale", b, a, ["--energy-classifier"]))  # OPB-FS
    task, ds = "housing", "california"
    out.append((task, ds, "mlp", None, None, []))
    for b in BETAS:
        out.append((task, ds, "ceb", b, None, []))
        out.append((task, ds, "ceb-tied", b, None, []))
    for b in BETAS:
        for a in ANCHORS:
            out.append((task, ds, "opb", b, a, []))  # EPB-L
            out.append((task, ds, "opb", b, a, ["--tied-head"]))  # EPB
            out.append((task, ds, "opb-free-scale", b, a, ["--tied-head"]))  # EPB-FS
    return out


def dirname(ds, model, beta, anchor, flags=()):
    name = f"{ds}_mlp_{model}"
    if beta is not None:
        name += f"_beta_{beta:g}"
    if anchor is not None:
        name += f"_anchor_{anchor:g}"
    # 同模型不同读出头的配置必须分目录（opb 的 GPB-L/GPB/EPB-L/EPB）
    if "--energy-classifier" in flags:
        name += "_eg"
    if "--tied-head" in flags:
        name += "_th"
    return name


def command(cfg):
    task, ds, model, beta, anchor, flags = cfg
    d = dirname(ds, model, beta, anchor, flags)
    parts = [
        sys.executable, "train.py", "--task", task, "--model", model,
        "--runs", "5", "--seed", "0", "--no-save",
        "--log-path", str(OUT / d / "train.log"),
    ]
    if beta is not None:
        parts += ["--beta", f"{beta:g}"]
    if anchor is not None:
        parts += ["--anchor-scale", f"{anchor:g}"]
    return d, parts + flags


def run_shard(shard_id, shard, log_path):
    failed = []
    with open(log_path, "w", encoding="utf-8") as f:
        for i, cfg in enumerate(shard):
            d, parts = command(cfg)
            (OUT / d).mkdir(parents=True, exist_ok=True)  # train.py 不创建显式 log 目录
            f.write(f"=== [{shard_id}:{i}] {d} ===\n")
            f.flush()
            r = subprocess.run(parts, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                failed.append(d)
                f.write(f"!!! FAILED {d} (exit {r.returncode})\n")
            else:
                f.write(f"--- OK {d}\n")
            f.flush()
    return failed


def main():
    parser = argparse.ArgumentParser(description="消融试验驱动器")
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--only", default=None, help="只跑目录名完全等于该值的配置")
    args = parser.parse_args()

    cfgs = [c for c in configs()
            if args.only is None or args.only == dirname(c[1], c[2], c[3], c[4], c[5])]
    print(f"配置总数：{len(cfgs)}（--shards {args.shards}）")

    shards = [cfgs[i::args.shards] for i in range(args.shards)]
    procs = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        log = f"/tmp/ablation_s{i}.log"
        import multiprocessing
        p = multiprocessing.Process(target=run_shard, args=(i, shard, log))
        p.start()
        procs.append((i, p, log))

    all_failed = []
    for i, p, log in procs:
        p.join()
        # 失败列表由各分片日志解析
        with open(log, encoding="utf-8") as f:
            for line in f:
                if line.startswith("!!! FAILED"):
                    all_failed.append(line.split()[2])

    print(f"\n=== 结束：失败 {len(all_failed)} / {len(cfgs)} ===")
    for d in all_failed:
        print("  FAILED:", d)
    print(f"日志目录：{OUT}")


if __name__ == "__main__":
    main()
