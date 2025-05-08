#!/usr/bin/env python3
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Constants from the C header files
# Operation flags
OP_NONE = 0x00
OP_INVALID = 0x80
OP_DATA_I8 = 0x01
OP_DATA_I16 = 0x02
OP_DATA_I16_I32 = 0x04
OP_DATA_I16_I32_I64 = 0x08
OP_EXTENDED = 0x10
OP_RELATIVE = 0x20
OP_MODRM = 0x40
OP_PREFIX = 0x80

# Instruction flags
F_INVALID = 0x01
F_PREFIX = 0x02
F_REX = 0x04
F_MODRM = 0x08
F_SIB = 0x10
F_DISP = 0x20
F_IMM = 0x40
F_RELATIVE = 0x80

# Instruction Control Flow Kind enumeration from FlowKind.h
# Architecture-agnostic categorization of instructions for traversing the
# control flow of a trace.
INSTRUCTION_CONTROL_FLOW_KIND = {
    'UNKNOWN': 0,      # The instruction could not be classified
    'OTHER': 1,        # Sequential instruction not affecting control flow
    'CALL': 2,         # Near (function) call
    'RETURN': 3,       # Near (function) return
    'JUMP': 4,         # Near unconditional jump
    'COND_JUMP': 5,    # Near conditional jump
    'FAR_CALL': 6,     # Call-like far transfer (SYSCALL, SYSENTER, FAR CALL)
    'FAR_RETURN': 7,   # Return-like far transfer (SYSRET, SYSEXIT, IRET, FAR RET)
    'FAR_JUMP': 8      # Jump-like far transfer (FAR JMP)
}

# Convert to JAX array for faster lookups
# instruction_control_flow_kinds = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.uint8)

def get_flow_kind_name(kind):
    """Get the string name for an instruction control flow kind value."""
    names = ["unknown", "other", "call", "return", "jump", 
            "cond jump", "far call", "far return", "far jump"]
    return names[kind] if kind < len(names) else "unknown"


flags_table = [
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_NONE,
    OP_NONE,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_NONE,
    OP_NONE,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_PREFIX,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_PREFIX,
    OP_NONE,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_PREFIX,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_PREFIX,
    OP_NONE,

    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_PREFIX,
    OP_PREFIX,
    OP_PREFIX,
    OP_PREFIX,
    OP_DATA_I16_I32,
    OP_MODRM | OP_DATA_I16_I32,
    OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,

    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I16_I32,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_DATA_I16 | OP_DATA_I16_I32,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_DATA_I8,
    OP_DATA_I16_I32,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,
    OP_DATA_I16_I32_I64,

    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_DATA_I16,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I16_I32,
    OP_DATA_I8 | OP_DATA_I16,
    OP_NONE,
    OP_DATA_I16,
    OP_NONE,
    OP_NONE,
    OP_DATA_I8,
    OP_NONE,
    OP_NONE,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_DATA_I8,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_DATA_I16 | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I8,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_PREFIX,
    OP_NONE,
    OP_PREFIX,
    OP_PREFIX,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM]
flags_table_ex = [
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_INVALID,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_INVALID,
    OP_NONE,
    OP_INVALID,
    OP_MODRM,
    OP_INVALID,
    OP_MODRM | OP_DATA_I8,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,
    OP_NONE,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM | OP_EXTENDED,
    OP_INVALID,
    OP_MODRM,
    OP_INVALID,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_INVALID,
    OP_NONE,
    OP_MODRM | OP_EXTENDED,
    OP_INVALID,
    OP_MODRM | OP_EXTENDED | OP_DATA_I8,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,
    OP_INVALID,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_NONE,
    OP_MODRM,
    OP_MODRM,
    OP_INVALID,
    OP_INVALID,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,
    OP_RELATIVE | OP_DATA_I16_I32,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_INVALID,
    OP_INVALID,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_MODRM,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM | OP_DATA_I8,
    OP_MODRM,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,
    OP_NONE,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,

    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_MODRM,
    OP_INVALID,
]

# Define flags tables as JAX arrays


def create_flags_tables():

    return jnp.array(flags_table, dtype=jnp.uint8), jnp.array(flags_table_ex, dtype=jnp.uint8)


def get_byte_at(code_bytes, position):
    """Safe way to get a byte at a position with bounds checking."""
    valid_pos = position < code_bytes.shape[0]
    return jnp.where(valid_pos, code_bytes[position], 0)


@jax.jit
def jax_ldasm(code_bytes, use_64_bit):
    """JAX implementation of the ldisasm function.

    Args:
        code_bytes: A JAX array containing the instruction bytes
        use_64bit: Boolean indicating if 64-bit mode is used

    Returns:
        A JAX array containing the disassembly information (12 bytes)
        [0]: instruction length
        [1]: flags
        [2]: rex
        [3]: modrm
        [4]: sib
        [5]: opcode_offset
        [6]: opcode_size
        [7]: disp_offset
        [8]: disp_size
        [9]: imm_offset
        [10]: imm_size
        [11]: primary_opcode (equivalent to FlowKind.h's primary_opcode)
    """
    # Create the flags tables
    flags_table, flags_table_ex = create_flags_tables()
    # Create a struct-like array to hold the ldasm_data
    result = jnp.zeros(12, dtype=jnp.uint8)

    # Initialize all variables
    position = jnp.array(0, jnp.uint8)
    s = jnp.array(0, jnp.uint8)
    flags = jnp.array(0, jnp.uint8)
    rex = jnp.array(0, jnp.uint8)
    modrm = jnp.array(0, jnp.uint8)
    sib = jnp.array(0, jnp.uint8)
    opcd_offset = jnp.array(0, jnp.uint8)
    opcd_size = jnp.array(0, jnp.uint8)
    disp_offset = jnp.array(0, jnp.uint8)
    disp_size = jnp.array(0, jnp.uint8)
    imm_offset = jnp.array(0, jnp.uint8)
    imm_size = jnp.array(0, jnp.uint8)
    primary_opcode = jnp.array(0, jnp.uint8)  # Added for compatibility with FlowKind.h

    # Prefix tracking
    rexw = jnp.array(0, jnp.uint8)
    pr_66 = jnp.array(0, jnp.uint8)
    pr_67 = jnp.array(0, jnp.uint8)

    # Phase 1: Parse prefixes
    def handle_prefix(state):
        pos, size, flg, pr66, pr67 = state
        # Get current byte
        current_byte = get_byte_at(code_bytes, pos)
        pr66_new = jnp.where(current_byte == 0x66, 1, pr66)
        pr67_new = jnp.where(current_byte == 0x67, 1, pr67)
        
        return (pos + 1, size + 1, flg | F_PREFIX, pr66_new, pr67_new)
        
    
    def cond(state):
        pos, size, flg, pr66, pr67 = state
        # Get current byte
        current_byte = get_byte_at(code_bytes, pos)
        
        # Check if it's a prefix
        is_prefix = (flags_table[current_byte] & OP_PREFIX) != 0
        return is_prefix
    
    position, s, flags, pr_66, pr_67 = lax.while_loop(
        cond,
        handle_prefix,
        (position, s, flags, pr_66, pr_67)
    )
    
    # Phase 2: Parse REX prefix (64-bit mode only)
    # Check if we have a REX prefix
    current_byte = get_byte_at(code_bytes, position)
    
    has_rex = use_64_bit & ((current_byte >> 4) == 4)
    
    # Update based on REX
    def handle_rex():
        rex_val = current_byte
        rexw_val = (rex_val >> 3) & 1
        return rex_val, rexw_val, position + 1, s + 1, flags | F_REX
    
    def skip_rex():
        return jnp.array(0, jnp.uint8), jnp.array(0, jnp.uint8), position, s, flags
    
    rex, rexw, position, s, flags = lax.cond(
        has_rex,
        handle_rex,
        skip_rex
    )

    # Check for invalid double REX
    current_byte = get_byte_at(code_bytes, position)
    invalid_rex = use_64_bit & ((current_byte >> 4) == 4)
    
    def handle_invalid_rex():
        return position + 1, s + 1, flags | F_INVALID
    
    def keep_position():
        return position, s, flags
    
    position, s, flags = lax.cond(
        (invalid_rex & has_rex),
        handle_invalid_rex,
        keep_position
    )

    # Phase 3: Parse opcode
    opcd_offset = position
    
    # Get first opcode byte
    op = get_byte_at(code_bytes, position)
    position = position + 1
    s = s + 1
    opcd_size = 1
    
    # Initial primary_opcode is the first opcode byte
    primary_opcode = op
    
    # Check for 2-byte opcode (0F)
    is_2byte = op == 0x0F
    
    def handle_2byte_opcode():
        nonlocal position, s, opcd_size
        op_2byte = get_byte_at(code_bytes, position)
        position_2byte = position + 1
        s_2byte = s + 1
        opcd_size_2byte = 2
        
        # For 2-byte opcodes, the primary_opcode is the second byte
        primary_opcode_2byte = op_2byte
        
        # Get flags for 2-byte opcode
        f_2byte = flags_table_ex[op_2byte]
        
        # Check for extended opcode (SSE)
        has_extended = f_2byte & OP_EXTENDED
        
        def handle_extended_op():
            op_ext = get_byte_at(code_bytes, position_2byte)
            # For 3-byte opcodes, the primary_opcode remains the second byte
            return (op_ext, 
                    position_2byte + 1, 
                    s_2byte + 1, 
                    opcd_size_2byte + 1, 
                    f_2byte, 
                    f_2byte & OP_INVALID,
                    primary_opcode_2byte)
        
        def keep_regular_op():
            return (op_2byte, 
                    position_2byte, 
                    s_2byte, 
                    opcd_size_2byte, 
                    f_2byte, 
                    f_2byte & OP_INVALID,
                    primary_opcode_2byte)
        
        op_final, pos_final, s_final, size_final, f_final, is_invalid, primary_op_final = lax.cond(
            has_extended,
            handle_extended_op,
            keep_regular_op
        )
        
        return op_final, pos_final, s_final, size_final, f_final, is_invalid, primary_op_final
    
    def keep_1byte_opcode():
        return op, position, s, opcd_size, flags_table[op], jnp.array(0, jnp.uint8), primary_opcode
    
    op, position, s, opcd_size, f, is_invalid, primary_opcode = lax.cond(
        is_2byte,
        handle_2byte_opcode,
        keep_1byte_opcode
    )
    
    # Special case for opcodes A0-A3
    is_special = (op >= 0xA0) & (op <= 0xA3)
    
    def update_pr66():
        return pr_67
    
    def keep_pr66():
        return pr_66
    
    pr_66 = lax.cond(
        is_special & (~is_2byte),
        update_pr66,
        keep_pr66
    )
    
    # Update flags for invalid opcode
    flags = flags | (F_INVALID * is_invalid)

    # Phase 4: Parse ModR/M, SIB, and displacement
    has_modrm = f & OP_MODRM

    def handle_modrm():
        nonlocal position, s, flags
        # Get ModR/M byte
        modrm_val = get_byte_at(code_bytes, position)
        
        # Extract mod, reg, rm fields
        mod_val = modrm_val >> 6
        ro_val = (modrm_val & 0x38) >> 3
        rm_val = modrm_val & 0x7
        
        # Special cases for F6/F7 opcodes
        is_f6 = (op == 0xF6) & ((ro_val == 0) | (ro_val == 1))
        is_f7 = (op == 0xF7) & ((ro_val == 0) | (ro_val == 1))
        f_val = f | (OP_DATA_I8 * is_f6) | (OP_DATA_I16_I32_I64 * is_f7)
        
        return (modrm_val, mod_val, ro_val, rm_val, position + 1, s + 1, 
                flags | F_MODRM, f_val)
    
    def skip_modrm():
        return jnp.array(0, jnp.uint8), jnp.array(0, jnp.uint8), jnp.array(0, jnp.uint8), jnp.array(0, jnp.uint8), position, s, flags, f
    
    modrm, mod, ro, rm, position, s, flags, f = lax.cond(
        has_modrm,
        handle_modrm,
        skip_modrm
    )
    
    # Check for SIB byte
    has_sib = has_modrm & (mod != 3) & (rm == 4) & (use_64_bit | (~pr_67))
    
    def handle_sib():
        nonlocal position, s, flags, disp_size
        # Get SIB byte
        sib_val = get_byte_at(code_bytes, position)
        # Check for special case with base=5 and mod=0
        is_special_sib = (sib_val & 7) == 5 & (mod == 0)
        # Update displacement size if special case
        disp_size_val = jnp.where(is_special_sib, 4, disp_size)
        
        return sib_val, position + 1, s + 1, flags | F_SIB, disp_size_val
    
    def skip_sib():
        return jnp.array(0, jnp.uint8), position, s, flags, disp_size
    
    sib, position, s, flags, disp_size = lax.cond(
        has_sib,
        handle_sib,
        skip_sib
    )

    # Displacement handling for each mod value using lax.cond
    # mod = 0
    def handle_mod0_disp():
        nonlocal disp_size
        
        # Different handling based on architecture and prefixes
        def handle_64bit():
            return jnp.where(rm == 5, 4, disp_size)
        
        def handle_32bit():
            def handle_pr67():
                return jnp.where(rm == 6, 2, disp_size)
            
            def handle_no_pr67():
                return jnp.where(rm == 5, 4, disp_size)
            
            return lax.cond(
                pr_67,
                handle_pr67,
                handle_no_pr67
            )
        
        return lax.cond(
            use_64_bit,
            handle_64bit,
            handle_32bit
        )
    
    # mod = 2
    def handle_mod2_disp():
        def handle_64bit():
            return jnp.array(4, jnp.uint8)
        
        def handle_32bit():
            return jnp.where(pr_67, jnp.array(2, jnp.uint8), jnp.array(4, jnp.uint8))
        
        return lax.cond(
            use_64_bit,
            handle_64bit,
            handle_32bit
        )
        
    disp_size = lax.cond(
        has_modrm,
        lambda: lax.switch(mod,
            [handle_mod0_disp, lambda: jnp.array(1, jnp.uint8), handle_mod2_disp],),
        lambda: disp_size
    )

    # Record displacement offset and adjust position
    has_disp = disp_size > 0
    disp_offset = jnp.where(has_disp, position, 0)
    position = position + disp_size
    s = s + disp_size
    flags = flags | (F_DISP * has_disp)

    # Phase 5: Parse immediate data
    # Determine immediate size based on operation type and prefixes
    has_imm64 = (rexw | (use_64_bit & (op >= 0xA0) & (op <= 0xA3))) & (f & OP_DATA_I16_I32_I64)
    has_imm32_16 = (f & OP_DATA_I16_I32) | (f & OP_DATA_I16_I32_I64)
    
    def determine_imm_size():
        def handle_imm64():
            return jnp.array(8, jnp.uint8)
        
        def handle_imm32_16():
            return jnp.array(4, jnp.uint8) - (pr_66 << 1)
        
        def handle_no_imm():
            return jnp.array(0, jnp.uint8)
        
        # First check for 64-bit immediate
        return lax.cond(
            has_imm64,
            handle_imm64,
            lambda: lax.cond(
                has_imm32_16,
                handle_imm32_16,
                handle_no_imm
            )
        )
    
    imm_size = determine_imm_size()
    
    # Add sizes for OP_DATA_I8 and OP_DATA_I16
    imm_size = imm_size + (f & 0x3)
    
    # Record immediate offset and adjust position
    has_imm = imm_size > 0
    
    def handle_imm():
        nonlocal flags
        imm_off = position
        flg = flags | F_IMM
        # Check for relative flag
        flg = jnp.where(f & OP_RELATIVE, flg | F_RELATIVE, flg)
        return imm_off, flg
    
    def no_imm():
        return jnp.array(0, jnp.uint8), flags
    
    imm_offset, flags = lax.cond(
        has_imm,
        handle_imm,
        no_imm
    )
    
    s = s + imm_size

    # Check for too long instruction
    flags = flags | (F_INVALID * (s > 15))

    # Fill result array
    result = result.at[0].set(s)                # instruction length
    result = result.at[1].set(flags)            # flags
    result = result.at[2].set(rex)              # REX prefix
    result = result.at[3].set(modrm)            # ModR/M byte
    result = result.at[4].set(sib)              # SIB byte
    result = result.at[5].set(opcd_offset)      # opcode offset
    result = result.at[6].set(opcd_size)        # opcode size
    result = result.at[7].set(disp_offset)      # displacement offset
    result = result.at[8].set(disp_size)        # displacement size
    result = result.at[9].set(imm_offset)       # immediate offset
    result = result.at[10].set(imm_size)        # immediate size
    result = result.at[11].set(primary_opcode)  # primary opcode byte for flow kind analysis

    return result

@partial(jax.jit, static_argnums=(1,))
def byte_sequence_to_instr_bytes(byte_sequence, max_instr_len=15):
    """Convert a byte sequence to instruction bytes.

    Args:
        byte_sequence: A JAX array of bytes (uint8) with shape [L]
        max_instr_len: Maximum instruction length to process

    Returns:
        A JAX array of shape [L, 15] containing instruction bytes
    """
    # Pad the byte sequence
    padded_bytes = jnp.pad(byte_sequence, (0, max_instr_len - 1), constant_values=0x90)

    # Create a function to extract instruction bytes
    def extract_instr_bytes(i):
        return lax.dynamic_slice_in_dim(padded_bytes, start_index=i, slice_size=max_instr_len, axis=0)

    # Create positions array
    positions = jnp.arange(byte_sequence.shape[0])

    # Map the extraction function over all positions
    instr_bytes = jax.vmap(extract_instr_bytes)(positions)

    return instr_bytes

@jax.jit
def disasm_jax(instr_bytes, use_64_bit):
    """JAX implementation to disassemble a byte sequence.

    Args:
        byte_sequence: A JAX array of bytes (uint8) with shape [L, 15]
    Returns:
        A JAX array of shape [L, 12] containing disassembly info for each position
    """
    # disassemble each instruction
    result = jax.vmap(jax_ldasm, in_axes=(0, None))(instr_bytes, use_64_bit)
    return result

@jax.jit
def disasm_info_to_opcode_and_modrm(disasm_info):
    """Convert disassembly info to opcode and modrm.

    Args:
        disasm_info: JAX array of shape [L, 12] containing disassembly info
    Returns:
        JAX array of shape [L, 3] containing [primary_opcode, opcode_len, modrm]
    """
    # Extract relevant fields
    primary_opcode = disasm_info[:, 11]  # primary opcode
    opcode_len = disasm_info[:, 6]       # opcode size
    modrm = disasm_info[:, 3]            # ModR/M byte

    # Create the output array
    opcode_and_modrm = jnp.stack((primary_opcode, opcode_len, modrm), axis=-1)

    return opcode_and_modrm

@jax.jit
def map_opcode_into_control_flow_kind(opcode_and_modrm):
    """JAX implementation of MapOpcodeIntoControlFlowKind from FlowKind.h.
    
    Args:
        opcode_and_modrm: JAX array [primary_opcode, opcode_len, modrm]
        
    Returns:
        JAX uint8 representing the InstructionControlFlowKind
    """
    primary_opcode = opcode_and_modrm[0]
    opcode_len = opcode_and_modrm[1]
    modrm = opcode_and_modrm[2]
    
    # Default to OTHER
    flow_kind = 1  # OTHER
    
    # Check if opcode_len > 2, return OTHER
    flow_kind = jnp.where(opcode_len > 2, 
                          1,  # OTHER
                          flow_kind)
    
    # Check for conditional jumps (0x70-0x7F)
    is_70_7F = (primary_opcode >= 0x70) & (primary_opcode <= 0x7F)
    flow_kind = jnp.where(is_70_7F & (opcode_len == 1),
                          5,  # COND_JUMP
                          flow_kind)
    
    # Check for conditional jumps (0x80-0x8F)
    is_80_8F = (primary_opcode >= 0x80) & (primary_opcode <= 0x8F)
    flow_kind = jnp.where(is_80_8F & (opcode_len == 2),
                          5,  # COND_JUMP
                          flow_kind)
    
    # Handle specific opcodes
    flow_kind = jnp.where((primary_opcode == 0x9A) & (opcode_len == 1),
                          6,  # FAR_CALL
                          flow_kind)
    
    # Handle 0xFF opcode with different modrm.reg values
    is_FF = (primary_opcode == 0xFF) & (opcode_len == 1)
    modrm_reg = (modrm >> 3) & 7
    
    flow_kind = jnp.where(is_FF & (modrm_reg == 2),
                          2,  # CALL
                          flow_kind)
    
    flow_kind = jnp.where(is_FF & (modrm_reg == 3),
                          6,  # FAR_CALL
                          flow_kind)
    
    flow_kind = jnp.where(is_FF & (modrm_reg == 4),
                          4,  # JUMP
                          flow_kind)
    
    flow_kind = jnp.where(is_FF & (modrm_reg == 5),
                          8,  # FAR_JUMP
                          flow_kind)
    
    # Check for CALL instructions
    flow_kind = jnp.where((primary_opcode == 0xE8) & (opcode_len == 1),
                          2,  # CALL
                          flow_kind)
    
    # Check for FAR_CALL instructions
    is_far_call = ((primary_opcode == 0xCD) | (primary_opcode == 0xCC) | 
                   (primary_opcode == 0xCE) | (primary_opcode == 0xF1)) & (opcode_len == 1)
    flow_kind = jnp.where(is_far_call,
                          6,  # FAR_CALL
                          flow_kind)
    
    # Check for FAR_RETURN instructions
    flow_kind = jnp.where((primary_opcode == 0xCF) & (opcode_len == 1),
                          7,  # FAR_RETURN
                          flow_kind)
    
    # Check for JUMP instructions
    is_jump = ((primary_opcode == 0xE9) | (primary_opcode == 0xEB)) & (opcode_len == 1)
    flow_kind = jnp.where(is_jump,
                          4,  # JUMP
                          flow_kind)
    
    # Check for FAR_JUMP instructions
    flow_kind = jnp.where((primary_opcode == 0xEA) & (opcode_len == 1),
                          8,  # FAR_JUMP
                          flow_kind)
    
    # Check for COND_JUMP instructions
    is_cond_jump = ((primary_opcode == 0xE3) | (primary_opcode == 0xE0) | 
                   (primary_opcode == 0xE1) | (primary_opcode == 0xE2)) & (opcode_len == 1)
    flow_kind = jnp.where(is_cond_jump,
                          5,  # COND_JUMP
                          flow_kind)
    
    # Check for RETURN instructions
    is_return = ((primary_opcode == 0xC3) | (primary_opcode == 0xC2)) & (opcode_len == 1)
    flow_kind = jnp.where(is_return,
                          3,  # RETURN
                          flow_kind)
    
    # Check for FAR_RETURN instructions
    is_far_return = ((primary_opcode == 0xCB) | (primary_opcode == 0xCA)) & (opcode_len == 1)
    flow_kind = jnp.where(is_far_return,
                          7,  # FAR_RETURN
                          flow_kind)
    
    # Check for additional FAR_CALL instructions
    is_far_call_2 = ((primary_opcode == 0x05) | (primary_opcode == 0x34)) & (opcode_len == 2)
    flow_kind = jnp.where(is_far_call_2,
                          6,  # FAR_CALL
                          flow_kind)
    
    # Check for additional FAR_RETURN instructions
    is_far_return_2 = ((primary_opcode == 0x35) | (primary_opcode == 0x07)) & (opcode_len == 2)
    flow_kind = jnp.where(is_far_return_2,
                          7,  # FAR_RETURN
                          flow_kind)
    
    # Handle opcode 0x01 with specific modrm values
    is_01 = (primary_opcode == 0x01) & (opcode_len == 2)
    
    flow_kind = jnp.where(is_01 & (modrm == 0xc1),
                          6,  # FAR_CALL
                          flow_kind)
    
    is_far_return_3 = is_01 & ((modrm == 0xc2) | (modrm == 0xc3))
    flow_kind = jnp.where(is_far_return_3,
                          7,  # FAR_RETURN
                          flow_kind)
    
    return flow_kind

@jax.jit
def extract_number(instr_bytes, offset, size):
    """Extract a number from instruction bytes.

    Args:
        instr_bytes: A JAX array of shape [15] containing instruction bytes
        offset: Offset to start extracting
        size: Size of the number to extract

    Returns:
        A number extracted from the instruction bytes
    """    
    # Use dynamic_slice to get the relevant bytes
    padded_bytes = jnp.pad(instr_bytes, (0, 8))
    sliced_bytes = lax.dynamic_slice(padded_bytes, (offset,), (8,))
    
    indices = jnp.arange(0, 8)
    mask = (indices < size)
    # Create a mask to select the relevant bytes
    selected_bytes = jnp.where(mask, sliced_bytes, 0)
    # Convert to a number
    result = jnp.sum(selected_bytes * (1 << (indices * 8)), axis=0, dtype=jnp.int32)
    return result

@jax.jit
def parse_disasm(instr_bytes, disasm, flow_kind):
    '''Calculate the control flow for the instruction.
    
    Args:
        instr_bytes: A JAX array of shape [L, 15] containing instruction bytes
        disasm: A JAX array of shape [L, 12] containing disassembly info
        flow_kind: A JAX array of shape [L] containing InstructionControlFlowKind
    Returns:
        A tuple containing:
        instruction: A JAX array of shape [L, 5], containing:
            - rex
            - opcode
            - modrm
            - sib
            - instruction length
        control flow: A JAX array of shape [L, 18] containing the control flow
        The four dimensions are:
            - must target * 1, For those unconditional junmps or natural fallthrough
            - may targets * 2, For those conditional jumps
            - next instruction address * 1, To mark the location of instruction boundaries
            - overlapping addresses * 14, To mark the overlapping instructions
    '''
    # Extract necessary fields from disasm
    instr_len = disasm[:, 0]
    flags = disasm[:, 1]
    rex, modrm, sib = disasm[:, 2], disasm[:, 3], disasm[:, 4]
    # opcode_offset, opcode_size = disasm[:, 5], disasm[:, 6]
    # disp_offset, disp_size = disasm[:, 7], disasm[:, 8]
    imm_offset, imm_size = disasm[:, 9], disasm[:, 10]
    primary_opcode = disasm[:, 11]
    
    # Extract the numbers from instruction bytes
    # opcode = jax.vmap(extract_number)(instr_bytes, opcode_offset, opcode_size)
    # disp = jax.vmap(extract_number)(instr_bytes, disp_offset, disp_size)
    imm = jax.vmap(extract_number)(instr_bytes, imm_offset, imm_size)
    
    offset = jnp.arange(0, instr_bytes.shape[0])
    # next instruction address
    next_instr_addr = jnp.where(instr_len > 0, offset + instr_len, -1)
    # overlapping addresses
    ranges = jnp.arange(1, 15)
    overlapping = jnp.where(instr_len[:, jnp.newaxis] > ranges[jnp.newaxis, :], offset[:, jnp.newaxis] + ranges[jnp.newaxis, :], -1)
    # must target, based on flow kind, next for other, -1 for unconditional jumps, next + imm for unconditional jumps and calls
    must_target = jnp.where(flow_kind == 1, next_instr_addr, -1)
    must_target = jnp.where((((flow_kind == 2) | (flow_kind == 4)) & (imm_size != 0)), next_instr_addr + imm, must_target)
    # may target, based on flow kind, only for conditional jumps
    may_target = jnp.where((flow_kind == 5)[:, jnp.newaxis], jnp.stack((next_instr_addr, next_instr_addr + imm), axis=-1), -1)
    
    # Add base to each of the representative bytes
    # instruction = jnp.stack((rex, primary_opcode, modrm, sib, flags, instr_len), axis=-1, dtype=jnp.int32)
    # instr_len = jnp.ones(instruction.shape[0], dtype=jnp.int32) * instruction.shape[1]
    
    instruction = instr_bytes.astype(jnp.int32)
    idx = jnp.arange(0, instruction.shape[1])
    instruction = instruction + (idx * 256)
    control_flow = jnp.full((must_target.shape[0], 18), -1, dtype=jnp.int32)
    control_flow = control_flow.at[:, 0].set(must_target)
    control_flow = control_flow.at[:, 1:3].set(may_target)
    control_flow = control_flow.at[:, 3].set(next_instr_addr)
    control_flow = control_flow.at[:, 4:].set(overlapping)
    control_flow = jnp.where((control_flow < 0) | (control_flow >= instr_bytes.shape[0]), -1, control_flow)
    return instruction, control_flow, instr_len
    
@jax.jit
def preprocess_binary(byte_sequence, use_64_bit):
    """Preprocess the byte sequence for further process.

    Args:
        byte_sequence: A JAX array of bytes (uint8) with shape [L]

    Returns:
        A tuple containing:
            - A JAX array of shape [L, 12] containing disassembly info for each position
            - A JAX array of shape [L, 3] containing [primary_opcode, opcode_len, modrm]
            - A JAX array of shape [L] containing the InstructionControlFlowKind
    """
    instr_bytes = byte_sequence_to_instr_bytes(byte_sequence)
    disasm_result = disasm_jax(instr_bytes, use_64_bit)
    opcode_modrm = disasm_info_to_opcode_and_modrm(disasm_result)
    flow_kind = jax.vmap(map_opcode_into_control_flow_kind)(opcode_modrm)
    instruction, control_flow, instr_len = parse_disasm(instr_bytes, disasm_result, flow_kind)

    return instruction, control_flow, instr_len

def interpret_disasm_result(disasm_array, include_flow_kind=True):
    """Interpret the disassembly result array into a human-readable dictionary.

    This function converts the raw numeric values from disassembly into a structured
    dictionary with decoded flags and other helpful information for debugging and
    visualization purposes.

    Args:
        disasm_array: A JAX array of shape [L, 12] from disasm_jax
        include_flow_kind: If True, calculate and include flow kind information

    Returns:
        A dictionary with human-readable interpretation of each instruction
    """
    result = {}
    
    # Flag names for readable output
    flag_names = {
        F_INVALID: "INVALID",
        F_PREFIX: "PREFIX",
        F_REX: "REX",
        F_MODRM: "MODRM",
        F_SIB: "SIB",
        F_DISP: "DISP",
        F_IMM: "IMM",
        F_RELATIVE: "RELATIVE"
    }
    
    # Opcode flag names for readable output
    opcode_flag_names = {
        OP_NONE: "NONE",
        OP_INVALID: "INVALID",
        OP_DATA_I8: "DATA_I8",
        OP_DATA_I16: "DATA_I16",
        OP_DATA_I16_I32: "DATA_I16_I32",
        OP_DATA_I16_I32_I64: "DATA_I16_I32_I64", 
        OP_EXTENDED: "EXTENDED",
        OP_RELATIVE: "RELATIVE",
        OP_MODRM: "MODRM",
        OP_PREFIX: "PREFIX"
    }
    
    for i in range(len(disasm_array)):
        data = disasm_array[i]

        inst_length = int(data[0])
        if inst_length == 0:
            continue  # Skip empty entries

        # Decode instruction flags to readable strings
        flag_value = int(data[1])
        flag_strings = [name for flag, name in flag_names.items() if flag_value & flag]
        
        # Decode primary opcode
        primary_opcode = int(data[11])
        
        entry = {
            "length": inst_length,
            "flags": flag_value,
            "flags_decoded": " | ".join(flag_strings) if flag_strings else "NONE",
            "rex": int(data[2]),
            "modrm": int(data[3]),
            "sib": int(data[4]),
            "opcode_offset": int(data[5]),
            "opcode_size": int(data[6]),
            "disp_offset": int(data[7]),
            "disp_size": int(data[8]),
            "imm_offset": int(data[9]),
            "imm_size": int(data[10]),
            "primary_opcode": f"0x{primary_opcode:02X}"
        }
        
        # Calculate ModR/M components if present
        if flag_value & F_MODRM:
            modrm_byte = int(data[3])
            entry["modrm_decoded"] = {
                "mod": (modrm_byte >> 6) & 0x3,
                "reg": (modrm_byte >> 3) & 0x7,
                "rm": modrm_byte & 0x7
            }
        
        # Include flow kind if requested
        if include_flow_kind:
            # Convert to the format expected by map_opcode_into_control_flow_kind
            opcode_and_modrm = jnp.array([
                primary_opcode,
                int(data[6]),  # opcode_size
                int(data[3])   # modrm
            ])
            
            flow_kind = map_opcode_into_control_flow_kind(opcode_and_modrm)
            flow_kind_value = int(flow_kind)
            entry["flow_kind"] = flow_kind_value
            entry["flow_kind_name"] = get_flow_kind_name(flow_kind_value)
        
        result[f"pos_{i}"] = entry

    return result


# Example usage
if __name__ == "__main__":
    # Example byte sequence (x86 instructions)
    example_bytes = jnp.array(np.random.randint(0, 256, size=(32, 1024 * 1024), dtype=np.uint8), dtype=jnp.uint8)
    # example_bytes = jnp.array([
    #     0xE8, 0x12, 0x34, 0x56, 0x78,  # CALL instruction
    #     0xFF, 0xD0,                     # CALL rax
    #     0xC3,                           # RET
    #     0xEB, 0x10,                     # JMP short
    #     0x74, 0x05                      # JE short
    # ], dtype=jnp.uint8)
    # Disassemble
    # disasm_result, opcode_modrm, flow_kind = preprocess_binary(example_bytes)
    # print("Disassembly Result:\n", disasm_result)
    # print("Opcode and ModRM:\n", opcode_modrm)
    # print("Control Flow Kind:\n", flow_kind)
    
    instruction, control_flow = jax.vmap(preprocess_binary)(example_bytes)
    for i in range(10):
        instruction, control_flow = jax.vmap(preprocess_binary)(example_bytes)
    # print("Instruction:\n", instruction)
    # print("Control Flow:\n", control_flow)

