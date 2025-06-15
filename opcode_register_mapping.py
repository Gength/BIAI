import json
# Opcode category mapping - extended version
OPCODE_CATEGORIES = json.load(open("opcode_categories.json", "r"))

# Register type mapping - extended version
REGISTER_TYPES = json.load(open("register_types.json", "r"))

# Load additional opcode categories
def load_extended_opcode_categories():
    # Try to load additional opcode categories from a JSON file
    try:
        with open("unknown_opcode.json", "r") as f:
            unknown_data = json.load(f)
        
        # Add default categories for unknown opcodes
        for opcode in unknown_data.get("unkown_opcode", []):
            # Determine category based on opcode prefix
            if opcode.startswith('v'):
                OPCODE_CATEGORIES[opcode] = 'vector'
            elif opcode.startswith('f'):
                OPCODE_CATEGORIES[opcode] = 'fpu'
            elif opcode.startswith('p'):
                OPCODE_CATEGORIES[opcode] = 'vector'
            elif opcode.startswith('aes'):
                OPCODE_CATEGORIES[opcode] = 'crypto'
            elif opcode.startswith('sha'):
                OPCODE_CATEGORIES[opcode] = 'crypto'
            elif opcode.startswith('cmov'):
                OPCODE_CATEGORIES[opcode] = 'data_transfer'
            elif opcode.startswith('set'):
                OPCODE_CATEGORIES[opcode] = 'flag_operation'
            elif opcode.startswith('bnd'):
                OPCODE_CATEGORIES[opcode] = 'bounds_prefix'
            elif opcode.startswith('lock'):
                OPCODE_CATEGORIES[opcode] = 'lock_prefix'
            elif opcode.startswith('rep'):
                OPCODE_CATEGORIES[opcode] = 'rep_prefix'
            elif opcode in ['nop', 'pause']:
                OPCODE_CATEGORIES[opcode] = 'nop'
            else:
                # Default category is system instruction
                OPCODE_CATEGORIES[opcode] = 'system'
        
        # Add default types for unknown registers
        for reg in unknown_data.get("unknown_reg", []):
            reg_lower = reg.lower()
            if reg_lower.startswith('xmm') or reg_lower.startswith('ymm') or reg_lower.startswith('zmm'):
                REGISTER_TYPES[reg_lower] = 'vector'
            elif reg_lower.startswith('mm'):
                REGISTER_TYPES[reg_lower] = 'mmx'
            elif reg_lower.startswith('st'):
                REGISTER_TYPES[reg_lower] = 'fpu'
            elif reg_lower.startswith('r') and reg_lower[1:].isdigit():
                REGISTER_TYPES[reg_lower] = 'gpr'
            elif reg_lower in ['rip', 'eip']:
                REGISTER_TYPES[reg_lower] = 'ip'
            else:
                # Default type is general purpose register
                REGISTER_TYPES[reg_lower] = 'gpr'
    
    except FileNotFoundError:
        print("unknown_opcode.json file not found, using default categories")
    except Exception as e:
        print(f"Error loading additional opcode categories: {e}")
load_extended_opcode_categories()
json.dump(OPCODE_CATEGORIES, open("opcode_categories.json", "w"), indent=4)
json.dump(REGISTER_TYPES, open("register_types.json", "w"), indent=4)