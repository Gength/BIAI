import multiprocessing
import os
import pickle
import re
from scipy.sparse import lil_matrix
def filter_near(op):
    return '<NEAR>'

def filter_far(op):
    return '<FAR>'

def filter_void(op):
    return '<VOID>'

def filter_special(op):
    return '<SPECIAL>'

def filter_reg(op):
    return op

def filter_disp_in_displacement(displacement):
    if displacement > 0:
        return '<POSITIVE>'
    elif displacement < 0:
        return '<NEGATIVE>'
    else:
        return '<ZERO>'

def unsigned2signed(number, width):
    if number > 2**(width-1) - 1:
        number = number - 2**width
    return number

def normalize_instruction_assembly_code(instruction, arch_bits):
    """
    兼容angr/capstone的归一化函数
    :param instruction: capstone指令对象
    :param arch_bits: 架构位宽 (32 or 64)
    """
    cpu_width = arch_bits  # 使用实际架构位宽
    
    # 获取助记符
    inst = instruction.mnemonic
    
    # 跳转/调用指令列表
    jump_mnemonics = {
        'jmp', 'je', 'jne', 'jz', 'jnz', 'jb', 'jbe', 'ja', 'jae', 
        'jl', 'jle', 'jg', 'jge', 'jo', 'jno', 'js', 'jns', 'jp', 
        'jnp', 'jecxz', 'jrcxz', 'loop', 'loope', 'loopne', 'call'
    }
    
    # 处理每个操作数
    operands = []
    for i, op in enumerate(instruction.operands):
        # 寄存器操作数
        if op.type == 1:  # CS_OP_REG
            reg_name = instruction.reg_name(op.reg)
            operands.append(filter_reg(reg_name))
            
        # 立即数操作数
        elif op.type == 2:  # CS_OP_IMM
            # 跳转指令特殊处理
            if instruction.mnemonic in jump_mnemonics:
                operands.append(filter_near(op))
            else:
                imm = op.imm
                # 处理负数的补码表示
                if imm < 0:
                    imm = (1 << cpu_width) + imm
                
                if unsigned2signed(imm, cpu_width) > 0:
                    operands.append('<POSITIVE>')
                elif unsigned2signed(imm, cpu_width) < 0:
                    operands.append('<NEGATIVE>')
                else:
                    operands.append('<ZERO>')
                    
        # 内存操作数
        elif op.type == 3:  # CS_OP_MEM
            mem = op.mem
            parts = []
            
            # 基址寄存器
            if mem.base != 0:
                parts.append(instruction.reg_name(mem.base))
            
            # 索引寄存器
            if mem.index != 0:
                index_part = instruction.reg_name(mem.index)
                if mem.scale > 1:
                    index_part += f"*{mem.scale}"
                parts.append(index_part)
            
            # 位移值
            disp_str = filter_disp_in_displacement(mem.disp)
            if mem.disp != 0 or not parts:  # 只有位移或位移+寄存器
                parts.append(disp_str)
            
            operands.append(f"[{'+'.join(parts)}]")
            
        # 其他类型操作数
        else:
            operands.append(f'<UNSUPPORTED_OP:{op.type}>')
    
    # 拼接操作数
    if operands:
        inst += ' ' + ', '.join(operands)
    
    return inst

import angr
import capstone

def disassemble_function(binary_path, function_address=None, function_name=None):
    """
    使用 angr 反汇编二进制文件中的函数
    :param binary_path: 二进制文件路径
    :param function_address: 目标函数地址(十六进制)
    :param function_name: 目标函数名
    """
    # 加载二进制文件
    project = angr.Project(binary_path, auto_load_libs=False)
    
    # 构建控制流图(CFG)
    cfg = project.analyses.CFGFast()
    
    # 获取目标函数
    target_func = None
    if function_address:
        target_func = cfg.functions.get(int(function_address, 16))
    elif function_name:
        for addr, func in cfg.functions.items():
            if func.name == function_name:
                target_func = func
                break
    
    if not target_func:
        print("[-] 未找到目标函数")
        return
    
    print(f"[+] 找到目标函数: {target_func.name} @ 0x{target_func.addr:x}")
    print(f"    函数大小: {target_func.size} 字节")
    print(f"    基本块数量: {len(target_func.block_addrs_set)}")
    
    # 获取函数的汇编代码
    print("\n======= 汇编代码 (原始 => 归一化) =======")
    
    # 按地址排序基本块
    blocks = sorted(target_func.blocks, key=lambda b: b.addr)
    
    for block in blocks:
        print(f"\n; 基本块 0x{block.addr:x} - 0x{block.addr + block.size:x}")
        
        # 反汇编基本块中的每条指令
        capstone = project.arch.capstone
        for instruction in capstone.disasm(block.bytes, block.addr):
            # 原始指令
            address = f"0x{instruction.address:x}"
            mnemonic = instruction.mnemonic
            op_str = instruction.op_str
            raw_inst = f"{mnemonic} {op_str}" if op_str else mnemonic
            
            # 归一化指令
            normalized = normalize_instruction_assembly_code(
                instruction, 
                project.arch.bits
            )
            
            print(f"{address}: {raw_inst.ljust(30)} => {normalized}")





def normalize_instruction(inst_str):
    """
    归一化单条汇编指令
    :param inst_str: 汇编指令字符串
    :return: 归一化后的指令字符串
    """
    CPU_WIDTH = 64  # 假设使用64位架构
    def unsigned2signed(number, width):
        """将无符号数转换为有符号数"""
        if number >= 2**(width-1):
            return number - 2**width
        return number
    # 指令前缀列表
    prefixes = {
        'lock', 'rep', 'repe', 'repne', 'repz', 'repnz', 
        'bnd', 'xacquire', 'xrelease', 'notrack'
    }
    
    # 首先将整个指令按空格分割成令牌
    tokens = inst_str.split()
    if not tokens:
        return ""
    
    # 处理指令前缀和主操作码
    prefix_parts = []
    mnemonic = None
    operand_tokens = []
    
    # 收集所有前缀
    i = 0
    while i < len(tokens) and tokens[i].lower() in prefixes:
        prefix_parts.append(tokens[i])
        i += 1
    
    # 下一个令牌是主操作码
    if i < len(tokens):
        mnemonic = tokens[i]
        i += 1
    
    # 剩余的是操作数部分
    operand_tokens = tokens[i:]
    
    # 构建完整助记符（前缀 + 主指令）
    if prefix_parts and mnemonic:
        full_mnemonic = " ".join(prefix_parts + [mnemonic]).lower()
    elif mnemonic:
        full_mnemonic = mnemonic.lower()
    else:
        full_mnemonic = " ".join(prefix_parts).lower()
    
    # 如果没有操作数，直接返回
    if not operand_tokens:
        return full_mnemonic
    
    # 将操作数令牌组合回字符串（保留原始格式）
    operands_str = " ".join(operand_tokens)
    operands = [op.strip() for op in operands_str.split(',')]
    normalized_ops = []
    
    # 跳转指令列表（包括带前缀的变体）
    jump_mnemonics = {
        'jmp', 'je', 'jne', 'jz', 'jnz', 'jb', 'jbe', 'ja', 'jae', 
        'jl', 'jle', 'jg', 'jge', 'jo', 'jno', 'js', 'jns', 'jp', 
        'jnp', 'jecxz', 'jrcxz', 'loop', 'loope', 'loopne', 'call'
    }
    
    # 寄存器列表（x86_64）
    registers = {
        # 64位
        'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp', 
        'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
        # 32位
        'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
        # 16位
        'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
        # 8位
        'al', 'bl', 'cl', 'dl', 'sil', 'dil', 'bpl', 'spl',
        'ah', 'bh', 'ch', 'dh'
    }
    
    for op in operands:
        # 处理内存操作数 (如 "qword ptr [rbp - 8]")
        if 'ptr' in op or '[' in op:
            # 提取方括号内的表达式
            start = op.find('[')
            end = op.find(']')
            if start != -1 and end != -1:
                expr = op[start+1:end].strip()
                
                # 分割表达式中的组件
                components = expr.replace('+', ' + ').replace('-', ' - ').split()
                normalized_components = []
                sign = 1
                
                # 处理每个组件
                for comp in components:
                    comp = comp.strip()
                    if comp == '+':
                        sign = 1
                        continue
                    elif comp == '-':
                        sign = -1
                        continue
                    
                    # 寄存器组件
                    if comp.lower() in registers:
                        normalized_components.append(comp)
                    # 数值组件
                    else:
                        try:
                            # 处理带符号的数值
                            if comp.startswith('-0x'):
                                num = -int(comp[3:], 16)
                            elif comp.startswith('+0x'):
                                num = int(comp[3:], 16)
                            elif comp.startswith('0x'):
                                num = int(comp[2:], 16)
                            elif comp.endswith('h'):
                                # 处理带符号的后缀格式
                                if comp[0] == '-':
                                    num = -int(comp[1:-1], 16)
                                elif comp[0] == '+':
                                    num = int(comp[1:-1], 16)
                                else:
                                    num = int(comp[:-1], 16)
                            else:
                                # 尝试解析为整数（可能带符号）
                                num = int(comp)
                            
                            # 归一化位移值
                            signed_num = unsigned2signed(sign * abs(num), CPU_WIDTH)
                            if signed_num > 0:
                                normalized_components.append('<POSITIVE>')
                            elif signed_num < 0:
                                normalized_components.append('<NEGATIVE>')
                            else:
                                normalized_components.append('<ZERO>')
                        except:
                            normalized_components.append(comp)
                
                # 重新构建内存表达式
                normalized_mem = '[' + '+'.join(normalized_components) + ']'
                normalized_ops.append(normalized_mem)
            else:
                normalized_ops.append('<MEM>')
        
        # 处理跳转目标 - 检查实际指令（不带前缀）是否为跳转指令
        elif mnemonic and mnemonic.lower() in jump_mnemonics:
            normalized_ops.append('<NEAR>')
        
        # 处理立即数（包括负数）
        else:
            try:
                # 处理带符号的十六进制格式
                if op.startswith('-0x'):
                    num = -int(op[3:], 16)
                elif op.startswith('+0x'):
                    num = int(op[3:], 16)
                elif op.startswith('0x'):
                    num = int(op[2:], 16)
                elif op.endswith('h'):
                    # 处理带符号的后缀格式
                    if op[0] == '-':
                        num = -int(op[1:-1], 16)
                    elif op[0] == '+':
                        num = int(op[1:-1], 16)
                    else:
                        num = int(op[:-1], 16)
                else:
                    # 尝试解析为整数（可能带符号）
                    num = int(op)
                
                # 归一化立即数
                signed_num = unsigned2signed(num, CPU_WIDTH)
                if signed_num > 0:
                    normalized_ops.append('<POSITIVE>')
                elif signed_num < 0:
                    normalized_ops.append('<NEGATIVE>')
                else:
                    normalized_ops.append('<ZERO>')
            except:
                # 不是数值，检查是否是寄存器
                if op.lower() in registers:
                    normalized_ops.append(op)
                else:
                    normalized_ops.append(op)
    
    # 构建归一化后的指令
    normalized_inst = full_mnemonic
    if normalized_ops:
        normalized_inst += ' ' + ', '.join(normalized_ops)
    
    return normalized_inst

def load_assembly_data(data):
    data_pairs = []
    for item in data:
        if item[0].startswith('0x'):
            temp = []
            for datum in item:
                instr=str.split(datum, '\t')[1:]
                instr = ' '.join(instr)
                temp.append(instr)
            item = temp
        normalized = [normalize_instruction(inst) for inst in item]
        if len(normalized) % 2 != 0:
            normalized.append('nop')
        for i in range(0, len(normalized)-1, 2):
            # Skip if either instruction is empty after normalization
            
                # Ensure both instructions are not empty
            data_pairs.append((normalized[i].strip(), normalized[i+1].strip()))
    return data_pairs

def process_binary_file(args):
        binary_path, save_path = args
        binary_name = os.path.split(binary_path)[-1]
        try:
            results = dict()
            proj = angr.Project(binary_path, auto_load_libs=False)
            cfg = proj.analyses.CFGFast()
            
            for addr, func in cfg.functions.items():
                if func.name in ['UnresolvableCallTarget', 'UnresolvableJumpTarget']:
                    continue
                # 先检查基本块数量
                blocks = list(func.blocks)
                n_blocks = len(blocks)
                if n_blocks <= 5:
                    continue
                
                # 初始化函数条目
                if func.name not in results:
                    results[func.name] = dict()

                # 创建地址到索引的映射
                # 按地址排序基本块以确保一致的索引顺序
                sorted_blocks = sorted(blocks, key=lambda b: b.addr)
                block_addrs = [block.addr for block in sorted_blocks]
                addr_to_idx = {addr: idx for idx, addr in enumerate(block_addrs)}
                # 保存地址到索引的映射
                results[func.name]['addr_to_idx'] = addr_to_idx
                # 初始化邻接矩阵
                adj_matrix = lil_matrix((n_blocks, n_blocks), dtype=int)

                for block in sorted_blocks:
                    # 生成邻接矩阵
                    node = cfg.model.get_any_node(block.addr)
                    succ = getattr(node, 'successors', [])
                    for succ_node in succ:
                        if succ_node.addr in block_addrs:
                            src_idx = addr_to_idx[block.addr]
                            dest_idx = addr_to_idx[succ_node.addr]
                            adj_matrix[src_idx, dest_idx] = 1

                    results[func.name][block.addr] = []
                    # 归一化指令
                    for insn in block.capstone.insns:
                        normalized_insn = normalize_instruction_assembly_code(
                            insn, 
                            proj.arch.bits
                        )
                        results[func.name][block.addr].append(normalized_insn)
                # 将稀疏矩阵转换为可序列化的格式（如COO格式）
                results[func.name]['adjacency_matrix'] = adj_matrix.tocoo()

        except Exception as e:
            print(f"Error processing {binary_name}: {e}")
        if any(results.values()):
            with open(os.path.join(save_path, "output_"+binary_name+".pkl"), "wb") as f:
                pickle.dump(results, f)
        else:
            print(f"Warning: No results for {binary_name}, skipping save.")

from tqdm import tqdm
if __name__ == "__main__":
    data_dir = '.'
    binary_folder = os.path.join(data_dir, "Dataset-1")

    save_root = "baseline"
    os.makedirs(save_root, exist_ok=True)
    architecture = ['x86', 'x64']
    compiler = ['gcc']
    tasks = []
    for subfolder in os.listdir(binary_folder):
        subfolder_path = os.path.join(binary_folder, subfolder)
        # if not os.path.isdir(subfolder_path):
        #     continue
        save_path = os.path.join(save_root, subfolder)
        os.makedirs(save_path, exist_ok=True)
        for bin_name in os.listdir(subfolder_path):
            bin_path = os.path.join(subfolder_path, bin_name)
            if(os.path.isfile(os.path.join(save_path, "output_"+bin_name+".pkl"))):
                continue
            key_words = re.split('[-_]', bin_name)
            intersection = len(set(key_words) & set(architecture)) > 0 and len(set(key_words) & set(compiler)) > 0
            # 如果二进制文件名中没有关键字或架构信息，则跳过
            if intersection:
                tasks.append((bin_path, save_path))

    # num_processes = min(multiprocessing.cpu_count(), len(tasks))
    num_processes = 12
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_binary_file, tasks), total=len(tasks)))