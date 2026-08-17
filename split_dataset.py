"""Dataset preparation following the Order-Matters paper (Table 1).

Data: gcc-compiled binaries from Dataset-1, architectures {x64, arm64},
optimization levels {O0, O1, O2, O3} (paper Task 2 uses O0-O3; Task 1 uses
O2/O3 cross-platform pairs). Os and clang are excluded, matching the paper
("we choose x86-64 and ARM as the two platforms, and compile on gcc").

Outputs (in ./outputs):
- baseline-{train,val,test}.jsonl      : all functions (pretraining + finetune)
- {split}-function-idx-mapping.pkl     : key -> line number
- task1-o{2,3}-{split}-function_pool.csv : cross-platform (x64<->arm64)
                                          same-function pairs, label 1/0
- task2-{x64,arm64}-{split}-functions.pkl : function lists for O0-O3
                                          classification (paper reports the
                                          two platforms separately)
- baseline-vocab.txt                   : token vocab (built from the JSONL)
"""
import os
import re
import json
import pickle
import random
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from models.tokenizer import AsmTokenizer

BASELINE_DIR = os.path.join(".", "baseline")
OUTPUT_DIR = os.path.join(".", "outputs")
DATASET_DIR = os.path.join(".", "data", "Dataset-1")

ARCHS = ["x64", "arm64"]
OPTS = ["O0", "O1", "O2", "O3"]


def binary_target(file_name):
    """Executable/library identity after the compiler metadata prefix."""
    if "_" not in file_name:
        raise ValueError(f"unexpected Dataset-1 binary name: {file_name!r}")
    return file_name.split("_", 1)[1]


def build_binary_project_map(dataset_dir=DATASET_DIR):
    """Map each flattened binary name back to its source project directory."""
    mapping = {}
    for project in sorted(os.listdir(dataset_dir)):
        project_dir = os.path.join(dataset_dir, project)
        if not os.path.isdir(project_dir) or project == "z3":
            continue
        for binary_name in sorted(os.listdir(project_dir)):
            previous = mapping.get(binary_name)
            if previous is not None and previous != project:
                raise ValueError(
                    f"binary name {binary_name!r} occurs in both "
                    f"{previous!r} and {project!r}; flattened baseline names "
                    "are ambiguous"
                )
            mapping[binary_name] = project
    return mapping


def parse_bin_info(file_name, project=None):
    """Extract (project, compiler, version, opt, arch, file_name) from a pkl name."""
    parts = re.split(r"[-_]", file_name)
    project = project or parts[-1]
    compiler = next((c for c in ["gcc", "clang"] if c in parts), "unknown")
    raw_version = next((v for v in parts if re.fullmatch(r"\d+(\.\d+)?", v)), None)
    version = str(float(raw_version)) if raw_version is not None else "unknown"
    opt = next((o for o in ["O0", "O1", "O2", "O3", "Os"] if o in parts), "unknown")
    arch = next((a for a in ["x86", "x64", "arm32", "arm64", "mips32", "mips64"]
                 if a in parts), "unknown")
    return project, compiler, version, opt, arch, file_name


def collect_functions():
    """Collect all functions with their (project, compiler, version, opt, arch)."""
    project_by_binary = build_binary_project_map()
    function_list = []  # (function_name, project, compiler, version, opt, arch, file_name)
    for f in sorted(os.listdir(BASELINE_DIR)):
        if not (f.endswith(".pkl") and f.startswith("output_")):
            continue
        file_name = f.replace("output_", "").replace(".pkl", "")
        if file_name not in project_by_binary:
            raise ValueError(f"cannot resolve source project for {file_name!r}")
        project, compiler, version, opt, arch, _ = parse_bin_info(
            file_name, project=project_by_binary[file_name])
        if compiler != "gcc" or arch not in ARCHS or opt not in OPTS:
            continue  # paper: gcc, x86-64/ARM, O0-O3
        try:
            with open(os.path.join(BASELINE_DIR, f), "rb") as fh:
                data = pickle.load(fh)
        except Exception:
            continue
        for func_name in sorted(data.keys()):
            function_list.append((func_name, project, compiler, version, opt,
                                  arch, file_name))
    return function_list


def group_split(functions, test_size=0.1, val_size=0.1, seed=42):
    """Group-level split: same-source functions (same project+name) stay together."""
    groups = defaultdict(list)
    for fn in functions:
        groups[(fn[0], fn[1])].append(fn)  # (function_name, project)

    group_keys = sorted(groups.keys())
    random.seed(seed)
    train_val, test = train_test_split(group_keys, test_size=test_size,
                                       random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=seed)
    splits = {"train": [], "val": [], "test": []}
    for split, keys in zip(splits, [train, val, test]):
        for k in keys:
            splits[split].extend(groups[k])
    return splits


def generate_jsonl(splits):
    """Write per-function JSONL + idx mapping for all three splits."""
    for split, functions in splits.items():
        function_name_idx_map = {}
        count = 0
        output_path = os.path.join(OUTPUT_DIR, f"baseline-{split}.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        # Group by file to load each pkl once.
        by_file = defaultdict(list)
        for fn in functions:
            by_file[fn[6]].append(fn)
        with open(output_path, "a", encoding="utf-8") as out_file:
            for file_name, funcs in by_file.items():
                with open(os.path.join(BASELINE_DIR, f"output_{file_name}.pkl"),
                          "rb") as fh:
                    binary_data = pickle.load(fh)
                for (func_name, project, compiler, version, opt, arch,
                     fname) in funcs:
                    function_data = binary_data[func_name]
                    addr_to_idx = function_data["addr_to_idx"]
                    block_addr = sorted(addr_to_idx, key=addr_to_idx.get)
                    instruction_blocks = [
                        " ".join(function_data[a]) for a in block_addr
                    ]
                    adj = function_data["adjacency_matrix"]
                    output_obj = {
                        "instruction_blocks": instruction_blocks,
                        "adjacency_matrix": {
                            "row": adj.row.tolist(),
                            "col": adj.col.tolist(),
                            "data": adj.data.tolist(),
                            "shape": adj.shape,
                        },
                        "opt": opt,
                        "arch": arch,
                    }
                    key = (func_name, compiler, version, opt, arch, file_name)
                    function_name_idx_map[key] = count
                    count += 1
                    out_file.write(json.dumps(output_obj) + "\n")
        with open(os.path.join(OUTPUT_DIR,
                               f"{split}-function-idx-mapping.pkl"), "wb") as fh:
            pickle.dump(function_name_idx_map, fh)
        print(f"{split}: {count} functions -> {output_path}")


def generate_task1_pools(splits, seed=42):
    """Task 1: cross-platform (x64 <-> arm64) same-function pairs, per opt."""
    rng = random.Random(seed)
    for opt_level in ["O2", "O3"]:
        for split, functions in splits.items():
            # anchor = x64, target = arm64 (same function name & project, same opt)
            by_name = defaultdict(lambda: {"x64": [], "arm64": []})
            for (func_name, project, compiler, version, opt, arch,
                 file_name) in functions:
                if opt == opt_level:
                    # A symbol such as `main` can refer to unrelated source in
                    # two executables from the same top-level project. Pair
                    # only matching symbol + binary target; compiler versions
                    # and architectures remain the varying dimensions.
                    by_name[(func_name, binary_target(file_name))][arch].append(
                        (version, file_name))
            pairs = []
            for (func_name, target_name), groups in by_name.items():
                x64_list, arm64_list = groups["x64"], groups["arm64"]
                if not x64_list or not arm64_list:
                    continue
                # Positive pairs: every x64/ARM64 compiler-version combination
                # compiled from the same symbol in the same binary target.
                pos = []
                for a in x64_list:
                    for b in arm64_list:
                        pos.append((a, b))
                for (a_ver, a_file), (b_ver, b_file) in pos:
                    pairs.append({
                        "anchor_function_file": a_file,
                        "anchor_function_name": func_name,
                        "anchor_compiler": "gcc", "anchor_version": a_ver,
                        "anchor_opt": opt_level, "anchor_arch": "x64",
                        "target_function_file": b_file,
                        "target_function_name": func_name,
                        "target_compiler": "gcc", "target_version": b_ver,
                        "target_opt": opt_level, "target_arch": "arm64",
                        "label": 1,
                    })
                # Negative pairs: random different-name functions.
                neg_pool = [
                    k for k, value in by_name.items()
                    if k != (func_name, target_name) and value["arm64"]
                ]
                rng.shuffle(neg_pool)
                n_neg = min(len(pos), len(neg_pool))
                for i in range(n_neg):
                    other = by_name[neg_pool[i]]
                    if not other["arm64"]:
                        continue
                    a = rng.choice(x64_list)
                    b = rng.choice(other["arm64"])
                    pairs.append({
                        "anchor_function_file": a[1],
                        "anchor_function_name": func_name,
                        "anchor_compiler": "gcc", "anchor_version": a[0],
                        "anchor_opt": opt_level, "anchor_arch": "x64",
                        "target_function_file": b[1],
                        "target_function_name": neg_pool[i][0],
                        "target_compiler": "gcc", "target_version": b[0],
                        "target_opt": opt_level, "target_arch": "arm64",
                        "label": 0,
                    })
            pd.DataFrame(pairs).to_csv(
                os.path.join(OUTPUT_DIR,
                             f"task1-{opt_level.lower()}-{split}-function_pool.csv"),
                index=False)
            print(f"task1-{opt_level.lower()}-{split}: {len(pairs)} pairs")


def generate_task2_splits(splits):
    """Task 2: per-platform function lists for O0-O3 classification."""
    for arch in ARCHS:
        for split, functions in splits.items():
            # 6-tuple keys, matching the JSONL idx mapping (no project).
            funcs = [(fn, cv, v, o, ar, fname)
                     for (fn, pr, cv, v, o, ar, fname) in functions
                     if ar == arch]
            with open(os.path.join(OUTPUT_DIR,
                                   f"task2-{arch}-{split}-functions.pkl"),
                      "wb") as fh:
                pickle.dump(funcs, fh)
            print(f"task2-{arch}-{split}: {len(funcs)} functions")


def build_vocab():
    """Build the token vocab from the train split JSONL."""
    from datasets import load_dataset
    tokenizer = AsmTokenizer()  # starts from the base special-token vocab
    dataset = load_dataset(
        "json",
        data_files=os.path.join(OUTPUT_DIR, "baseline-train.jsonl"),
        split="train",
        streaming=True,
    )
    for data in dataset:
        tokenizer.build_vocab(data["instruction_blocks"])
    tokenizer.save_vocab(os.path.join(OUTPUT_DIR, "baseline-vocab.txt"))
    print(f"Vocab size: {len(tokenizer.vocab)}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "binary-project-map.json"),
              "w", encoding="utf-8") as fh:
        json.dump(build_binary_project_map(), fh, indent=2, sort_keys=True)
    functions = collect_functions()
    print(f"Collected {len(functions)} functions "
          f"({len(set((f[0], f[1]) for f in functions))} same-source groups)")
    splits = group_split(functions)
    generate_jsonl(splits)
    generate_task1_pools(splits)
    generate_task2_splits(splits)
    build_vocab()
