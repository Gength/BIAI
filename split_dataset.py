import os
import re
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from models.tokenizer import AsmTokenizer
BASELINE_DIR = os.path.join('.','baseline')
OUTPUT_DIR = os.path.join('.','outputs')

def split_functions():
    # Step 1: recursively collect all pkl files from subfolders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_files = []
    for root, _, files in os.walk(BASELINE_DIR):
        for f in files:
            if f.endswith(".pkl") and f.startswith("output_"):
                pkl_files.append(os.path.join(root, f))

    # Step 2: extract meta info (project, compiler, version, opt level) from file name

    def parse_bin_info(file_path):
        file_name = os.path.basename(file_path).replace("output_", "").replace(".pkl", "")
        project = os.path.basename(os.path.dirname(file_path)).lower()
        parts = re.split(r"[-_]", file_name)
        compiler = next((c for c in ["gcc", "clang"] if c in parts), "unknown")
        version = next((v for v in parts if re.match(r"\d+(\.\d+)?", v)), "unknown")
        opt = next((o for o in ["O0", "O1", "O2", "O3", "Os"] if o in parts), "unknown")
        return project, compiler, version, opt

    function_list = []
    function_pool = defaultdict(list)

    # Step 3: iterate over all pkl files and gather function info
    for file_path in pkl_files:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except:
            continue
        project, compiler, version, opt = parse_bin_info(file_path)
        bin_name = os.path.basename(file_path).replace("output_", "").replace(".pkl", "")
        for func_name in data:
            entry = {
                "function_name": func_name,
                "bin": bin_name,
                "project": project,
                "compiler": compiler,
                "version": version,
                "opt": opt
            }
            function_list.append(entry)
            function_pool[(project, func_name)].append(entry)

    # Step 4: split into train/val/test (function-level split)
    all_func_names = list(set((f["project"], f["function_name"]) for f in function_list))
    train_val, test = train_test_split(all_func_names, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.3, random_state=42)

    split_map = {}
    for proj, func in train:
        split_map[(proj, func)] = "train"
    for proj, func in val:
        split_map[(proj, func)] = "val"
    for proj, func in test:
        split_map[(proj, func)] = "test"

    # Step 5: write pickle file per split
    splits = {"train": [], "val": [], "test": []}
    for f in function_list:
        key = (f["project"], f["function_name"])
        split = split_map.get(key)
        if split:
            splits[split].append(f)

    for split in ["train", "val", "test"]:
        with open(os.path.join(OUTPUT_DIR, f"baseline-{split}-functions.pkl"), "wb") as f:
            pickle.dump(splits[split], f)

    # Step 6: generate function_pool.csv (pairwise combinations in test set)
    pairs = []
    test_funcs = splits["test"]
    grouped = defaultdict(list)
    for item in test_funcs:
        grouped[(item["project"], item["function_name"])].append(item)

    for key, group in grouped.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                pairs.append({
                    "anchor_function_bin": a["bin"],
                    "anchor_function_name": a["function_name"],
                    "anchor_compiler": a["compiler"],
                    "anchor_version": a["version"],
                    "anchor_opt": a["opt"],
                    "target_function_bin": b["bin"],
                    "target_function_name": b["function_name"],
                    "target_compiler": b["compiler"],
                    "target_version": b["version"],
                    "target_opt": b["opt"]
                })

    # Step 7: save CSV
    pd.DataFrame(pairs).to_csv(os.path.join(OUTPUT_DIR, "function_pool.csv"), index=False)






import json
# 定义处理单个split的函数
def process_split_datasets(split):
    if split == "test":
        function_name_idx_map = {}
        count = 0
    output_path = os.path.join(OUTPUT_DIR, f"baseline-{split}.jsonl")
    if(os.path.exists(output_path)):
        os.remove(output_path)
    with open(os.path.join(OUTPUT_DIR, f"baseline-{split}-functions.pkl"), "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    for name, group in df.groupby('bin'):
        project = group['project'].iloc[0]
        load_path = os.path.join(BASELINE_DIR, project, 'output_'+name+'.pkl')
        with open(load_path, "rb") as f:
            binary_data = pickle.load(f)
        
        with open(output_path, 'a', encoding='utf-8') as out_file:
            for row in group.itertuples(index=False):
                function_name = row.function_name
                function_data = binary_data[function_name]
                addr_to_idx = function_data['addr_to_idx']
                output_obj = {}
                # 只保留整数类型的键
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
                    function_name_idx_map[function_name] = count
                    count += 1
                    output_obj['function_name'] = function_name
                    output_obj['project'] = project
                    out_file.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
    if split == "test":
        # 保存function_name_idx_map到文件
        with open(os.path.join(OUTPUT_DIR, "function_name_idx_map.json"), "w") as f:
            json.dump(function_name_idx_map, f, indent=4)

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
