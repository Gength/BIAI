import os
import re
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from models.tokenizer import AsmTokenizer
from itertools import combinations
import random
BASELINE_DIR = os.path.join('.','baseline')
OUTPUT_DIR = os.path.join('.','outputs')

def split_functions():
    # Step 1: recursively collect all pkl files from subfolders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_files = []

    for file in os.listdir(BASELINE_DIR):
        if file.endswith(".pkl") and file.startswith("output_"):
            pkl_files.append(os.path.join(BASELINE_DIR, file))

    # Step 2: extract meta info (project, compiler, version, opt level) from file name

    def parse_bin_info(file_path):
        file_name = os.path.basename(file_path).replace("output_", "").replace(".pkl", "")
        parts = re.split(r"[-_]", file_name)
        bin_name = parts[-1]
        compiler = next((c for c in ["gcc", "clang"] if c in parts), "unknown")
        version = next((v for v in parts if re.match(r"\d+(\.\d+)?", v)), "unknown")
        opt = next((o for o in ["O0", "O1", "O2", "O3", "Os"] if o in parts), "unknown")
        return compiler, version, opt, file_name

    function_list = []

    # Step 3: iterate over all pkl files and gather function info
    for file_path in pkl_files:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except:
            continue
        compiler, version, opt, file_name = parse_bin_info(file_path)
        for func_name in data.keys():
            entry = (func_name, compiler, version, opt, file_name)
            function_list.append(entry)

    # Step 4: split into train/val/test (function-level split)
    train_val, test = train_test_split(function_list, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.3, random_state=42)


    for split, data in zip(["train", "val", "test"], [train, val, test]):
        with open(os.path.join(OUTPUT_DIR, f"baseline-{split}-functions.pkl"), "wb") as f:
            pickle.dump(data, f)

    # Step 6: generate function_pool.csv (pairwise combinations in test set)
    pairs = []
    # Convert test set to list of dicts for easier handling
    test_dicts = [
        {
            "function_name": fn,
            "compiler": comp,
            "version": ver,
            "opt": opt,
            "file_name": file_name
        }
        for fn, comp, ver, opt, file_name in test
    ]
    # Group by (function_name)
    grouped = defaultdict(list)
    for item in test_dicts:
        grouped[item["function_name"]].append(item)
    # delete items with less than 2 functions
    grouped = {k: v for k, v in grouped.items() if len(v) >= 2}

    for a_func, a_group in grouped.items():
        # Generate positive pairs: all pairs within the group
        positive_pairs_count = 0
        negative_pairs_count = 0
        for a, b in combinations(a_group, 2):
            pairs.append({
                "anchor_function_file": a["file_name"],
                "anchor_function_name": a["function_name"],
                "anchor_compiler": a["compiler"],
                "anchor_version": a["version"],
                "anchor_opt": a["opt"],
                "target_function_file": b["file_name"],
                "target_function_name": b["function_name"],
                "target_compiler": b["compiler"],
                "target_version": b["version"],
                "target_opt": b["opt"],
                "label": 1  # Positive pair
            })
            positive_pairs_count += 1
        # Generate negative pairs: randomly pick one from this group, one from each other group
        b_funcs = [k for k in grouped.keys() if k != a_func]
        random.shuffle(b_funcs)  # Shuffle to ensure randomness
        while negative_pairs_count < positive_pairs_count and len(b_funcs) > 0:
            a_idx = random.randint(0, len(a_group) - 1)
            a = a_group[a_idx]
            b_func = b_funcs.pop()
            b_group = grouped[b_func]
            b_idx = random.randint(0, len(b_group) - 1)
            b = b_group[b_idx]
            pairs.append({
                "anchor_function_file": a["file_name"],
                "anchor_function_name": a["function_name"],
                "anchor_compiler": a["compiler"],
                "anchor_version": a["version"],
                "anchor_opt": a["opt"],
                "target_function_file": b["file_name"],
                "target_function_name": b["function_name"],
                "target_compiler": b["compiler"],
                "target_version": b["version"],
                "target_opt": b["opt"],
                "label": 0  # Negative pair
            })
            negative_pairs_count += 1

    # Step 7: save CSV
    pd.DataFrame(pairs).to_csv(os.path.join(OUTPUT_DIR, "function_pool.csv"), index=False)

import json
# Define the function to process a single split
def process_split_datasets(split):
    if split == "test":
        function_name_idx_map = {}
        count = 0
    output_path = os.path.join(OUTPUT_DIR, f"baseline-{split}.jsonl")
    if(os.path.exists(output_path)):
        os.remove(output_path)
    with open(os.path.join(OUTPUT_DIR, f"baseline-{split}-functions.pkl"), "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data, columns=['function_name', 'compiler', 'version', 'opt', 'file_name'])
    for file_name, group in df.groupby('file_name'):
        load_path = os.path.join(BASELINE_DIR, 'output_'+file_name+'.pkl')
        with open(load_path, "rb") as f:
            binary_data = pickle.load(f)
        
        with open(output_path, 'a', encoding='utf-8') as out_file:
            for row in group.itertuples(index=False):
                function_name = row.function_name
                compiler = row.compiler
                version = row.version
                opt = row.opt
                function_data = binary_data[function_name]
                addr_to_idx = function_data['addr_to_idx']
                output_obj = {}
                # Only keep keys of integer type
                block_addr = [k for k in addr_to_idx.keys()]
                block_addr = sorted(block_addr, key=lambda x: addr_to_idx[x])
                instruction_blocks = []
                for instr_key in block_addr:
                    block = function_data[instr_key]
                    instructions = " <SEP> ".join(block)
                    instruction_blocks.append(instructions)
                output_obj['instruction_blocks'] = instruction_blocks
                
                adj = function_data['adjacency_matrix']
                output_obj['adjacency_matrix'] = {
                    'row': adj.row.tolist(),
                    'col': adj.col.tolist(),
                    'data': adj.data.tolist(),
                    'shape': adj.shape
                }
                if split != "test":
                    out_file.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                else:
                    key = (function_name, compiler, version, opt, file_name)
                    function_name_idx_map[key] = count
                    count += 1
                    out_file.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
    if split == "test":
        with open(os.path.join(OUTPUT_DIR, f"baseline-{split}-function_name_idx_map.pkl"), "wb") as f:
            pickle.dump(function_name_idx_map, f)

from multiprocessing import Process
if __name__ == '__main__':
    # Split into three datasets based on project and function name
    split_functions()
    # Create three processes to generate datasets separately
    splits = ["train", "val", "test"]
    processes = []
    
    for split in splits:
        p = Process(target=process_split_datasets, args=(split,))
        processes.append(p)
        p.start()
    
    # synchronize processes
    for p in processes:
        p.join()
    
    # build vocab
    vocab = {"<PAD>": 0, "<CLS>": 1, "<SEP>": 2, "<MASK>": 3, "<UNK>": 4, "<const>": 5}
    from datasets import load_dataset
    tokenizer = AsmTokenizer(vocab_file=os.path.join(OUTPUT_DIR, "baseline-vocab.txt"))
    tokenizer.vocab = vocab  # Use the predefined vocab
    for dataset_name in ["baseline-train", "baseline-val", "baseline-test"]:
        dataset_path = os.path.join(".", "outputs", f"{dataset_name}.jsonl")
        dataset = load_dataset('json', data_files=dataset_path, split="train", streaming=True)
        for data in dataset:
                tokenizer.build_vocab(data['instruction_blocks'])
    # Save vocab to file
    tokenizer.save_vocab(os.path.join(OUTPUT_DIR, "baseline-vocab.txt"))
