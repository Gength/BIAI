#!/usr/bin/env python
"""One-command reproduction pipeline.

Runs the whole Order-Matters flow:
    normalize -> split -> pretrain -> Task1 finetune (O2/O3) + full-pool eval
                                       -> Task2 train (x64/arm64) + eval

Idempotent: a stage is skipped iff its artifact exists AND is newer than the
pre-trained checkpoint (so re-running after a pretrain re-run re-does every
downstream stage automatically). Failed stages exit non-zero and can be
resumed by re-running this script.

Usage:
    uv run python pipeline.py                 # run everything (resumable)
    uv run python pipeline.py --force pretrain  # force one stage
"""
import argparse
import glob
import json
import os

import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
LOG_DIR = os.path.join(OUT, "pipeline-logs")
RESULT_DIR = os.path.join(OUT, "results")

PRETRAIN_MARK = os.path.join(OUT, "bert4-pretrain-hf", "bert-best", "pytorch_model.bin")
FT_O2 = os.path.join(OUT, "bert4-finetune-hf-o2", "CFGFusion-best.pth")
FT_O3 = os.path.join(OUT, "bert4-finetune-hf-o3", "CFGFusion-best.pth")
T2_X64 = os.path.join(OUT, "bert4-task2-hf", "CFGFusion-task2-x64-best.pth")
T2_ARM64 = os.path.join(OUT, "bert4-task2-hf", "CFGFusion-task2-arm64-best.pth")

FINETUNE_EPOCHS = 8    # explicit reproduction setting; paper does not publish epochs


def log_path(name):
    return os.path.join(LOG_DIR, f"{name}.log")


def result_path(name):
    return os.path.join(RESULT_DIR, f"{name}.json")


def _newer_than_all(path, dependencies):
    """Return whether an artifact exists and is newer than every dependency."""
    if not path or not os.path.exists(path):
        return False
    if any(not os.path.exists(dep) for dep in dependencies):
        return False
    return os.path.getmtime(path) > max(os.path.getmtime(dep)
                                        for dep in dependencies)


def _completed_after(artifact, marker, dependencies):
    """Require both a fresh artifact and a fresh completion marker."""
    return (_newer_than_all(artifact, dependencies)
            and _newer_than_all(marker, dependencies))


def stage_ready(name):
    """Artifact check per stage (None = not runnable yet, see stages below)."""
    baseline_files = glob.glob(os.path.join(ROOT, "baseline", "*.pkl"))
    normalize_ready = (
        len(baseline_files) >= 1000
        and _newer_than_all(
            os.path.join(ROOT, "baseline", "normalize-done.json"),
            [os.path.join(ROOT, "normalize_instr.py")])
    )
    split_marker = os.path.join(OUT, "baseline-vocab.txt")
    split_dependencies = [os.path.join(ROOT, "split_dataset.py")]
    if baseline_files:
        split_dependencies.append(max(
            baseline_files, key=os.path.getmtime))
    split_ready = (
        _newer_than_all(split_marker, split_dependencies)
        and os.path.exists(os.path.join(OUT, "binary-project-map.json"))
    )
    model_sources = [
        os.path.join(ROOT, "models", name) for name in
        ("bert.py", "checkpoint_utils.py", "collatefn.py", "dataset.py",
         "model.py", "retrieval.py", "tokenizer.py", "trainer.py")
    ]
    pretrain_dependencies = model_sources + [
        os.path.join(ROOT, "bert4_pretrain.py"),
        split_marker,
        os.path.join(OUT, "baseline-train.jsonl"),
        os.path.join(OUT, "baseline-val.jsonl"),
    ]
    pretrain_done = os.path.join(OUT, "bert4-pretrain-hf", "train-done.json")
    pretrain_ready = _completed_after(
        PRETRAIN_MARK, pretrain_done, pretrain_dependencies)

    finetune_dependencies = model_sources + [
        os.path.join(ROOT, "bert4_finetune.py"), PRETRAIN_MARK]
    task2_dependencies = model_sources + [
        os.path.join(ROOT, "bert4_task2.py"), PRETRAIN_MARK]
    eval_dependencies = model_sources + [
        os.path.join(ROOT, "bert_caculate_similarity.py")]
    checks = {
        "normalize": normalize_ready,
        "split": split_ready,
        "pretrain": pretrain_ready,
        "finetune-o2": _completed_after(
            FT_O2, os.path.join(OUT, "bert4-finetune-hf-o2", "train-done.json"),
            finetune_dependencies),
        "finetune-o3": _completed_after(
            FT_O3, os.path.join(OUT, "bert4-finetune-hf-o3", "train-done.json"),
            finetune_dependencies),
        "eval-o2": _newer_than_all(
            result_path("task1-o2"), eval_dependencies + [FT_O2]),
        "eval-o3": _newer_than_all(
            result_path("task1-o3"), eval_dependencies + [FT_O3]),
        "task2-x64": _completed_after(
            T2_X64, os.path.join(OUT, "bert4-task2-hf", "train-done-x64.json"),
            task2_dependencies),
        "task2-arm64": _completed_after(
            T2_ARM64, os.path.join(OUT, "bert4-task2-hf", "train-done-arm64.json"),
            task2_dependencies),
        "eval-task2-x64": _newer_than_all(
            result_path("task2-x64"), task2_dependencies + [T2_X64]),
        "eval-task2-arm64": _newer_than_all(
            result_path("task2-arm64"), task2_dependencies + [T2_ARM64]),
    }
    return checks[name]


def run_stage(name, cmd, force=False):
    if stage_ready(name) and not force:
        print(f"[pipeline] skip {name} (artifact up to date)")
        return True
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[pipeline] === {name} ===", flush=True)
    t0 = time.time()
    with open(log_path(name), "w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f"[pipeline] FAILED {name} after {dt:.1f} min — see {log_path(name)}")
        sys.exit(1)
    print(f"[pipeline] done {name} ({dt:.1f} min)", flush=True)
    return True


def run_eval(name, result_name, cmd, force=False):
    if stage_ready(name) and not force:
        print(f"[pipeline] skip {name} (artifact up to date)")
        return
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"[pipeline] === {name} ===", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        with open(log_path(name), "w") as f:
            f.write(text)
        print(f"[pipeline] FAILED {name} — see {log_path(name)}")
        sys.exit(1)
    metrics = {}
    for pat, key in [
        (r"MRR10:\s*([\d.]+)", "mrr10"),
        (r"Rank1:\s*([\d.]+)", "rank1"),
        (r"[Aa]ccuracy(?:\s*[:(]\s*|\s+)([\d.]+)", "accuracy"),
    ]:
        m = re.search(pat, text)
        if m:
            metrics[key] = float(m.group(1))
    expected = ({"mrr10", "rank1"} if result_name.startswith("task1-")
                else {"accuracy"})
    if not expected.issubset(metrics):
        with open(log_path(name), "w") as f:
            f.write(text)
        missing = ", ".join(sorted(expected - metrics.keys()))
        print(f"[pipeline] FAILED {name}: missing metrics {missing} — "
              f"see {log_path(name)}")
        sys.exit(1)
    with open(result_path(result_name), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[pipeline] {name}: {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="One-command reproduction pipeline")
    parser.add_argument("--force", default=None,
                        help="force a single stage to re-run (e.g. pretrain)")
    args = parser.parse_args()

    stages = [
        ("normalize", ["uv", "run", "python", "normalize_instr.py", "--force"]),
        ("split", ["uv", "run", "python", "split_dataset.py"]),
        ("pretrain", ["uv", "run", "python", "bert4_pretrain.py"]),
        ("finetune-o2", ["uv", "run", "python", "bert4_finetune.py",
                         "--task1_opt", "o2", "--epochs", str(FINETUNE_EPOCHS)]),
        ("eval-o2", None),  # filled below
        ("finetune-o3", ["uv", "run", "python", "bert4_finetune.py",
                         "--task1_opt", "o3", "--epochs", str(FINETUNE_EPOCHS)]),
        ("eval-o3", None),
        ("task2-x64", ["uv", "run", "python", "bert4_task2.py", "--platform", "x64"]),
        ("eval-task2-x64", None),
        ("task2-arm64", ["uv", "run", "python", "bert4_task2.py", "--platform", "arm64"]),
        ("eval-task2-arm64", None),
    ]

    def eval_cmd(opt, checkpoint):
        return ["uv", "run", "python", "bert_caculate_similarity.py",
                "--checkpoint", checkpoint,
                "--pretrained", os.path.join(OUT, "bert4-pretrain-hf", "bert-best"),
                "--pool", os.path.join(OUT, f"task1-{opt}-test-function_pool.csv"),
                "--device", "cuda"]

    eval_cmds = {
        "eval-o2": ("task1-o2", eval_cmd("o2", FT_O2)),
        "eval-o3": ("task1-o3", eval_cmd("o3", FT_O3)),
        "eval-task2-x64": (
            "task2-x64", ["uv", "run", "python", "bert4_task2.py",
                           "--platform", "x64", "--eval"]),
        "eval-task2-arm64": (
            "task2-arm64", ["uv", "run", "python", "bert4_task2.py",
                             "--platform", "arm64", "--eval"]),
    }

    for name, cmd in stages:
        force = (args.force == name)
        if name in eval_cmds:
            result_name, eval_command = eval_cmds[name]
            run_eval(name, result_name, eval_command, force=force)
        elif cmd is not None:
            run_stage(name, cmd, force=force)

    # Summary
    print("\n[pipeline] ================= SUMMARY =================")
    for name in ["task1-o2", "task1-o3", "task2-x64", "task2-arm64"]:
        path = result_path(name)
        if os.path.exists(path):
            print(f"[pipeline] {name}: {json.load(open(path))}")
    print("[pipeline] all stages done — update RESULTS.md with the numbers above")


if __name__ == "__main__":
    main()
