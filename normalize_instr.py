import multiprocessing
import argparse
import hashlib
import os
import pickle
import re
import time
import capstone
from scipy.sparse import lil_matrix
import json

MAX_INSTRUCTION_BLOCK_SIZE = 1000  # Maximum number of instructions per block
MIN_INSTRUCTION_BLOCK_SIZE = 6  # Minimum number of instructions per block

# Fallback branch/call mnemonic set (capstone groups are preferred).
JUMP_MNEMONICS = {
    # x86
    "jmp", "je", "jne", "ja", "jb", "jae", "jbe", "jg", "jge", "jl", "jle",
    "call", "loop", "loope", "loopne", "jz", "jnz", "js", "jns", "jo", "jno",
    "jc", "jnc", "jp", "jnp", "jcxz", "jecxz", "jrcxz", "ljmp", "lcall",
    "retn", "retf",
    # ARM (32/64)
    "b", "bl", "bx", "blx", "br", "blr", "cbz", "cbnz", "tbz", "tbnz",
    "beq", "bne", "bcs", "bhs", "bcc", "blo", "bmi", "bpl", "bvs", "bvc",
    "bhi", "bls", "bge", "blt", "bgt", "ble", "bal", "svc", "hvc", "smc",
    "bxj", "eret",
}

def normalize_instruction_assembly_code(instruction, arch_bits, Unknown=None):
    """Lightweight tokenization of one assembly instruction.

    Follows the paper's "tokens in the CFG blocks are regarded as words"
    idea: keep the mnemonic and register names verbatim, and only replace
    unbounded values (immediates, branch/call targets, memory displacements)
    with unified placeholder tokens (<IMM> / <TARGET>). No hand-crafted
    semantic categories (unlike the previous course pipeline which mapped
    opcodes to categories and registers to <REG:gpr>, collapsing the vocab
    to ~36 tokens and defeating the semantic-aware BERT pre-training).
    """
    mnemonic = instruction.mnemonic.lower()

    # Branch/call targets are <TARGET>; other immediates are <IMM>.
    is_branch = False
    try:
        is_branch = (instruction.group(capstone.CS_GRP_JUMP)
                     or instruction.group(capstone.CS_GRP_CALL))
    except Exception:
        pass
    if not is_branch:
        is_branch = mnemonic in JUMP_MNEMONICS

    operands = []
    for op in instruction.operands:
        if op.type == capstone.CS_OP_REG:  # 1
            operands.append(instruction.reg_name(op.reg).lower())
        elif op.type == capstone.CS_OP_IMM:  # 2
            operands.append("<TARGET>" if is_branch else "<IMM>")
        elif op.type == capstone.CS_OP_MEM:  # 3
            mem = op.mem
            mem_parts = []
            if mem.base != 0:
                mem_parts.append(instruction.reg_name(mem.base).lower())
            if mem.index != 0:
                index_name = instruction.reg_name(mem.index).lower()
                mem_parts.append(f"{index_name}*{mem.scale}" if mem.scale != 1
                                 else index_name)
            if mem.disp != 0:
                mem_parts.append("<IMM>")
            if not mem_parts:
                mem_parts.append("<ABS_MEM>")
            operands.append("[" + "+".join(mem_parts) + "]")
        else:  # FP immediates, etc.
            operands.append("<IMM>")

    normalized = mnemonic
    if operands:
        normalized += " " + ", ".join(operands)
    return normalized

import angr


def _is_elf_file(path):
    """Return True only for regular ELF files (including symlinks to files)."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def discover_binary_paths(binary_folder, architecture, compiler):
    """Discover the actual Dataset-1 binaries selected by this experiment."""
    binary_paths = []
    seen_binary_names = {}
    for subfolder in sorted(os.listdir(binary_folder)):
        subfolder_path = os.path.join(binary_folder, subfolder)
        if not os.path.isdir(subfolder_path) or subfolder == "z3":
            continue
        for bin_name in sorted(os.listdir(subfolder_path)):
            bin_path = os.path.join(subfolder_path, bin_name)
            key_words = set(re.split("[-_]", bin_name))
            selected = (bool(key_words & set(architecture))
                        and bool(key_words & set(compiler)))
            # angr may place directories such as ``*_angr_rtdb`` beside a
            # binary. Filename matching alone must never treat those sidecars
            # (or any other regular non-ELF file) as extraction targets.
            if not selected or not _is_elf_file(bin_path):
                continue
            previous_project = seen_binary_names.get(bin_name)
            if previous_project is not None and previous_project != subfolder:
                raise ValueError(
                    f"binary name {bin_name!r} occurs in both "
                    f"{previous_project!r} and {subfolder!r}"
                )
            seen_binary_names[bin_name] = subfolder
            binary_paths.append(bin_path)
    return binary_paths


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(temporary, path)

def disassemble_function(binary_path, function_address=None, function_name=None):
    """
    Disassemble a function in a binary file using angr.
    :param binary_path: Path to the binary file
    :param function_address: Target function address (hexadecimal)
    :param function_name: Target function name
    """
    # Load the binary file
    project = angr.Project(binary_path, auto_load_libs=False)
    
    # Build the Control Flow Graph (CFG)
    cfg = project.analyses.CFGFast()
    
    # Get the target function
    target_func = None
    if function_address:
        target_func = cfg.functions.get(int(function_address, 16))
    elif function_name:
        for addr, func in cfg.functions.items():
            if func.name == function_name:
                target_func = func
                break
    
    if not target_func:
        print("[-] Target function not found")
        return
    
    print(f"[+] Found target function: {target_func.name} @ 0x{target_func.addr:x}")
    print(f"    Function size: {target_func.size} bytes")
    print(f"    Number of basic blocks: {len(target_func.block_addrs_set)}")
    

    # get assembly code for the function
    print("\n======= assembly code (original => normalized) =======")
    

    # sort the basic blocks by their address
    blocks = sorted(target_func.blocks, key=lambda b: b.addr)
    normalized_blocks = []
    for block in blocks:
        print(f"\n; basic block 0x{block.addr:x} - 0x{block.addr + block.size:x}")
        
        # disassemble each instruction in the block
        capstone = project.arch.capstone
        normalized_instrs = []
        for instruction in capstone.disasm(block.bytes, block.addr):
            # original instruction 
            address = f"0x{instruction.address:x}"
            mnemonic = instruction.mnemonic
            op_str = instruction.op_str
            raw_inst = f"{mnemonic} {op_str}" if op_str else mnemonic
            
            # normalized instruction
            normalized = normalize_instruction_assembly_code(
                instruction, 
                project.arch.bits
            )
            print(f"{address}: {raw_inst.ljust(30)} => {normalized}")
            normalized_instrs.append((address, raw_inst, normalized))
        normalized_blocks.append(normalized_instrs)
            
    return normalized_blocks

def process_binary_file(args):
    Unknown = {
        "unkown_opcode": set(),
        "unknown_reg": set(),
        "unknown_operand": set()
    }
    binary_path, save_path = args
    binary_name = os.path.split(binary_path)[-1]
    try:
        results = dict()
        proj = angr.Project(binary_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()
        
        for addr, func in cfg.functions.items():
            if func.name in ['UnresolvableCallTarget', 'UnresolvableJumpTarget']:
                continue
            # First, check the number of basic blocks
            blocks = list(func.blocks)
            n_blocks = len(blocks)
            if n_blocks < MIN_INSTRUCTION_BLOCK_SIZE or n_blocks > MAX_INSTRUCTION_BLOCK_SIZE:
                continue

            # Handle duplicate function names (two different addresses with the
            # same name): keep them separate to avoid polluting each other.
            target_key = func.name
            if func.name not in results:
                results[func.name] = dict()
            elif addr not in results[func.name].get('addr_to_idx', {}):
                name = f"{func.name}@{addr:#x}"
                if name in results:
                    continue
                results[name] = dict()
                target_key = name

            # Create address-to-index mapping
            # Sort basic blocks by address to ensure consistent index order
            sorted_blocks = sorted(blocks, key=lambda b: b.addr)

            # 1. Normalize every block first, tolerating per-instruction and
            #    per-block failures (a single bad instruction must not drop
            #    the whole function, which previously left addr_to_idx
            #    inconsistent with the block data).
            block_data = {}
            for block in sorted_blocks:
                try:
                    instrs = []
                    for insn in block.capstone.insns:
                        try:
                            instrs.append(normalize_instruction_assembly_code(
                                insn, proj.arch.bits, Unknown=Unknown))
                        except Exception:
                            instrs.append("invalid")
                    block_data[block.addr] = instrs
                except Exception:
                    continue  # skip this block entirely

            good_addrs = sorted(block_data.keys())
            if len(good_addrs) < MIN_INSTRUCTION_BLOCK_SIZE:
                continue
            addr_to_idx = {a: i for i, a in enumerate(good_addrs)}

            # 2. Write the function entry only now that the block data is
            #    complete and consistent.
            entry = results[target_key]
            entry['addr_to_idx'] = addr_to_idx
            adj_matrix = lil_matrix((len(good_addrs), len(good_addrs)), dtype=int)
            for block_addr in good_addrs:
                node = cfg.model.get_any_node(block_addr)
                succ = getattr(node, 'successors', [])
                for succ_node in succ:
                    if succ_node.addr in addr_to_idx:
                        adj_matrix[addr_to_idx[block_addr],
                                   addr_to_idx[succ_node.addr]] = 1
                entry[block_addr] = block_data[block_addr]
            entry['adjacency_matrix'] = adj_matrix.tocoo()

    except Exception as e:
        print(f"Error processing {binary_name}: {e}")
    if any(results.values()):
        with open(os.path.join(save_path, "output_"+binary_name+".pkl"), "wb") as f:
            pickle.dump(results, f)
    else:
        print(f"Warning: No results for {binary_name}, skipping save.")
    return Unknown

from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract native-size CFGs with angr")
    parser.add_argument(
        "--force", action="store_true",
        help="re-extract and overwrite every selected local baseline pickle",
    )
    args = parser.parse_args()
    Unknown = {
        "unkown_opcode": set(),
        "unknown_reg": set(),
        "unknown_operand": set()
    }
    data_dir = '.'
    binary_folder = os.path.join(data_dir, "data", "Dataset-1")

    save_root = "baseline"
    os.makedirs(save_root, exist_ok=True)
    architecture = ['x64', 'arm64']  # paper Task 1: x86-64 <-> ARM
    compiler = ['gcc']
    tasks = []
    selected_paths = discover_binary_paths(
        binary_folder, architecture, compiler)
    selected_names = [os.path.basename(path) for path in selected_paths]
    extractor_sha256 = _file_sha256(__file__)
    progress_path = os.path.join(save_root, "normalize-in-progress.json")
    run_started_ns = time.time_ns()
    resumed_force = False

    if args.force and os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            if (progress.get("extractor_sha256") == extractor_sha256
                    and progress.get("selected_binaries") == selected_names):
                run_started_ns = int(progress["run_started_ns"])
                resumed_force = True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            resumed_force = False

    if args.force and not resumed_force:
        _write_json_atomic(progress_path, {
            "run_started_ns": run_started_ns,
            "extractor_sha256": extractor_sha256,
            "selected_binaries": selected_names,
        })

    for bin_path in selected_paths:
        output_path = os.path.join(
            save_root, "output_" + os.path.basename(bin_path) + ".pkl")
        output_is_current_force_run = (
            os.path.exists(output_path)
            and os.stat(output_path).st_mtime_ns >= run_started_ns
        )
        if ((not args.force and os.path.exists(output_path))
                or (resumed_force and output_is_current_force_run)):
            continue
        tasks.append((bin_path, save_root))

    if resumed_force:
        print(f"Resuming forced extraction: {len(tasks)} of "
              f"{len(selected_paths)} binaries remain.")

    # num_processes = min(multiprocessing.cpu_count(), len(tasks))
    num_processes = 6  # keep memory usage bounded (15GB host)
    unknown_list = []
    if tasks:
        with multiprocessing.Pool(processes=num_processes) as pool:
            unknown_list = list(tqdm(
                pool.imap_unordered(process_binary_file, tasks),
                total=len(tasks)))
    for unk in unknown_list:
        Unknown['unkown_opcode'].update(unk.get('unkown_opcode', set()))
        Unknown['unknown_reg'].update(unk.get('unknown_reg', set()))
        Unknown['unknown_operand'].update(unk.get('unknown_operand', set()))
    incomplete = []
    processed_paths = {binary_path for binary_path, _ in tasks}
    for binary_path in selected_paths:
        output_path = os.path.join(
            save_root, "output_" + os.path.basename(binary_path) + ".pkl")
        if (not os.path.exists(output_path)
                or (binary_path in processed_paths
                    and os.stat(output_path).st_mtime_ns < run_started_ns)
                or (args.force
                    and os.stat(output_path).st_mtime_ns < run_started_ns)):
            incomplete.append(binary_path)
    if incomplete:
        raise RuntimeError(
            f"CFG extraction failed or stayed stale for {len(incomplete)} "
            f"binaries; first entries: {incomplete[:5]}"
        )
    Unknown['unkown_opcode'] = list(Unknown['unkown_opcode'])
    Unknown['unknown_reg'] = list(Unknown['unknown_reg'])
    Unknown['unknown_operand'] = list(Unknown['unknown_operand'])
    if tasks:
        with open(os.path.join(save_root, "unknown_opcode.json"), "w") as f:
            json.dump(Unknown, f, indent=4)
    _write_json_atomic(os.path.join(save_root, "normalize-done.json"), {
        "selected_binaries": len(selected_paths),
        "processed_binaries": len(tasks),
        "skipped_existing": len(selected_paths) - len(tasks),
        "force": args.force,
        "resumed_force": resumed_force,
        "extractor_mtime_ns": os.stat(__file__).st_mtime_ns,
        "extractor_sha256": extractor_sha256,
    })
    if os.path.exists(progress_path):
        os.unlink(progress_path)
