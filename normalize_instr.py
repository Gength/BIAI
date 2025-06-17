import multiprocessing
import os
import pickle
import re
from scipy.sparse import lil_matrix
import json

with open("opcode_categories.json", "r") as f:
    OPCODE_CATEGORIES = json.load(f)
with open("register_types.json", "r") as f:
    REGISTER_TYPES = json.load(f)

def normalize_instruction_assembly_code(instruction, arch_bits, Unknown=None):
    """
    Compatible normalization function for angr/capstone.
    :param instruction: capstone instruction object
    :param arch_bits: architecture bit width (32 or 64)
    """
    # Get mnemonic and convert to lowercase
    inst = instruction.mnemonic.lower()
    
    # Lookup opcode category
    opcode_category = "unknown"
    for prefix, category in OPCODE_CATEGORIES.items():
        if inst.startswith(prefix):
            opcode_category = category
            break
    
    if opcode_category == "unknown":
        if Unknown is not None and inst not in Unknown['unkown_opcode']:
            print(f"Warning: Unknown Opcode '{inst}', using 'unknown' category.")
            Unknown['unkown_opcode'].add(inst)
    
    normalized_inst = f"{opcode_category}:{inst}"
    
    # Process each operand
    operands = []
    for i, op in enumerate(instruction.operands):
        # Register operand
        if op.type == 1:  # CS_OP_REG
            reg_name = instruction.reg_name(op.reg).lower()
            reg_type = REGISTER_TYPES.get(reg_name, "unknown")
            
            if reg_type == "unknown":
                # Infer type based on architecture bit width and register name
                if arch_bits == 32:
                    # 32-bit architecture register handling
                    if reg_name.startswith('e'):
                        reg_type = 'gpr'  # 32-bit general purpose register (eax, ebx, etc.)
                    elif reg_name in ['ax', 'bx', 'cx', 'dx']:
                        reg_type = 'gpr16'  # 16-bit general purpose register
                    elif reg_name in ['ah', 'al', 'bh', 'bl']:
                        reg_type = 'gpr8'   # 8-bit general purpose register
                    else:
                        reg_type = 'gpr'    # Default as general purpose register
                else:  # 64-bit architecture
                    if reg_name.startswith('r') and reg_name[1:].isdigit():
                        reg_type = 'gpr'   # 64-bit general purpose register (r8-r15)
                    elif reg_name.startswith('r'):
                        reg_type = 'gpr64' # Traditional 64-bit registers (rax, rbx, etc.)
                    elif reg_name in ['eax', 'ebx']:
                        reg_type = 'gpr32' # 32-bit mode registers
                    else:
                        reg_type = 'gpr'   # Default as general purpose register
                
                # Update register type mapping to avoid future warnings
                REGISTER_TYPES[reg_name] = reg_type
            
            operands.append(f"<REG:{reg_type}>")
            
        # Immediate operand
        elif op.type == 2:  # CS_OP_IMM:
            # Determine max bits based on architecture
            max_bits = 32 if arch_bits == 32 else 64
            
            # Special handling for jump instructions
            jump_mnemonics = {'jmp', 'je', 'jne', 'call', 'ja', 'jb', 'jae', 'jbe', 'jg', 'jge', 'jl', 'jle'}
            if instruction.mnemonic.lower() in jump_mnemonics:
                operands.append("<TARGET>")
            else:
                abs_imm = abs(op.imm)
                
                # Classify immediate value based on architecture and size
                if abs_imm == 0:
                    operands.append("<IMM:zero>")
                elif abs_imm < 2**8:
                    operands.append("<IMM:8bit>")
                elif abs_imm < 2**16:
                    operands.append("<IMM:16bit>")
                elif abs_imm < 2**32:
                    operands.append("<IMM:32bit>")
                elif arch_bits == 64 and abs_imm < 2**64:
                    operands.append("<IMM:64bit>")
                else:
                    # Special case handling
                    operands.append("<IMM:oversized>")
                    
        # Memory operand (consider architecture bit width)
        elif op.type == 3:  # CS_OP_MEM
            mem = op.mem
            mem_parts = []
            
            # Base register
            if mem.base != 0:
                base_reg = instruction.reg_name(mem.base).lower()
                base_type = REGISTER_TYPES.get(base_reg, "unknown")
                if base_type == "unknown":
                    # Infer based on architecture
                    if arch_bits == 32 and base_reg.startswith('e'):
                        base_type = "gpr"
                    elif arch_bits == 64 and (base_reg.startswith('r') or base_reg in ['rax', 'rbx']):
                        base_type = "gpr"
                    else:
                        base_type = "gpr"  # Default as general purpose register
                mem_parts.append(f"<BASE:{base_type}>")
            
            # Index register
            if mem.index != 0:
                index_reg = instruction.reg_name(mem.index).lower()
                index_type = REGISTER_TYPES.get(index_reg, "unknown")
                if index_type == "unknown":
                    # Infer based on architecture
                    if arch_bits == 32 and index_reg.startswith('e'):
                        index_type = "gpr"
                    elif arch_bits == 64 and (index_reg.startswith('r') or index_reg in ['rax', 'rbx']):
                        index_type = "gpr"
                    else:
                        index_type = "gpr"  # Default as general purpose register
                scale = mem.scale
                mem_parts.append(f"<INDEX:{index_type}:{scale}>")
            
            # Displacement value classification (consider architecture)
            if mem.disp != 0:
                abs_disp = abs(mem.disp)
                # Adjust classification thresholds based on architecture
                if arch_bits == 32:
                    if abs_disp < 2**8:
                        mem_parts.append("<DISP:small>")
                    elif abs_disp < 2**16:
                        mem_parts.append("<DISP:medium>")
                    else:
                        mem_parts.append("<DISP:large>")
                else:  # 64-bit
                    if abs_disp < 2**8:
                        mem_parts.append("<DISP:small>")
                    elif abs_disp < 2**16:
                        mem_parts.append("<DISP:medium>")
                    elif abs_disp < 2**32:
                        mem_parts.append("<DISP:large>")
                    else:
                        mem_parts.append("<DISP:huge>")
            
            # Special handling for cases without base/index
            if not mem_parts:
                mem_parts.append("<ABS_MEM>")
            
            operands.append("[" + "+".join(mem_parts) + "]")
            
        # Other operand types
        else:
            operands.append("<UNK_OP>")
            if Unknown is not None and op.type not in Unknown['unknown_operand']:
                print(f"Warning: Unknown Operand Type '{op.type}', using 'unknown_operand' type.")
                Unknown['unknown_operand'].add(op.type)
    
    # Build normalized instruction
    if operands:
        normalized_inst += " " + ", ".join(operands)
    
    return normalized_inst

import angr
import capstone

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
    
    for block in blocks:
        print(f"\n; basic block 0x{block.addr:x} - 0x{block.addr + block.size:x}")
        
        # disassemble each instruction in the block
        capstone = project.arch.capstone
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
            if n_blocks <= 5 or n_blocks > 1000:
                continue
            
            # Initialize function entry
            if func.name not in results:
                results[func.name] = dict()

            # Create address-to-index mapping
            # Sort basic blocks by address to ensure consistent index order
            sorted_blocks = sorted(blocks, key=lambda b: b.addr)
            block_addrs = [block.addr for block in sorted_blocks]
            addr_to_idx = {addr: idx for idx, addr in enumerate(block_addrs)}
            # Save address-to-index mapping
            results[func.name]['addr_to_idx'] = addr_to_idx
            # Initialize adjacency matrix
            adj_matrix = lil_matrix((n_blocks, n_blocks), dtype=int)

            for block in sorted_blocks:
                # Generate adjacency matrix
                node = cfg.model.get_any_node(block.addr)
                succ = getattr(node, 'successors', [])
                for succ_node in succ:
                    if succ_node.addr in block_addrs:
                        src_idx = addr_to_idx[block.addr]
                        dest_idx = addr_to_idx[succ_node.addr]
                        adj_matrix[src_idx, dest_idx] = 1

                results[func.name][block.addr] = []
                # Normalize instructions
                for insn in block.capstone.insns:
                    normalized_insn = normalize_instruction_assembly_code(
                        insn, 
                        proj.arch.bits,
                        Unknown=Unknown
                    )
                    results[func.name][block.addr].append(normalized_insn)
            # Convert sparse matrix to a serializable format (such as COO format)
            results[func.name]['adjacency_matrix'] = adj_matrix.tocoo()

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
    Unknown = {
        "unkown_opcode": set(),
        "unknown_reg": set(),
        "unknown_operand": set()
    }
    data_dir = '.'
    binary_folder = os.path.join(data_dir, "Dataset-1")

    save_root = "baseline"
    os.makedirs(save_root, exist_ok=True)
    architecture = ['x86', 'x64']
    compiler = ['gcc']
    tasks = []
    for subfolder in os.listdir(binary_folder):
        subfolder_path = os.path.join(binary_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for bin_name in os.listdir(subfolder_path):
            bin_path = os.path.join(subfolder_path, bin_name)
            if(os.path.exists(os.path.join(save_root, "output_"+bin_name+".pkl"))):
                continue
            key_words = re.split('[-_]', bin_name)
            intersection = len(set(key_words) & set(architecture)) > 0 and len(set(key_words) & set(compiler)) > 0
            # Skip if the binary file name does not contain required keywords or architecture information
            if intersection:
                tasks.append((bin_path, save_root))



    # num_processes = min(multiprocessing.cpu_count(), len(tasks))
    num_processes = 12
    with multiprocessing.Pool(processes=num_processes) as pool:
        unknown_list = list(tqdm(pool.imap_unordered(process_binary_file, tasks), total=len(tasks)))
    # wait for all processes to finish
    pool.close()
    pool.join()
    for unk in unknown_list:
        Unknown['unkown_opcode'].update(unk.get('unkown_opcode', set()))
        Unknown['unknown_reg'].update(unk.get('unknown_reg', set()))
        Unknown['unknown_operand'].update(unk.get('unknown_operand', set()))
    Unknown['unkown_opcode'] = list(Unknown['unkown_opcode'])
    Unknown['unknown_reg'] = list(Unknown['unknown_reg'])
    Unknown['unknown_operand'] = list(Unknown['unknown_operand'])
    with open(os.path.join(".", "unknown_opcode.json"), "w") as f:
        # Merging Unknown information from all processes requires separate handling; here, only the main process's Unknown is saved
        json.dump(Unknown, f, indent=4)