#!/usr/bin/env python3
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from flax import nnx
from tady.model.disasm_table import *

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
    
    done = jnp.array(False, dtype=jnp.bool_)

    # Default to OTHER
    flow_kind = 1  # OTHER

    # Check if opcode_len > 2, return OTHER
    flow_kind = jnp.where(opcode_len > 2,
                          1,  # OTHER
                          flow_kind)
    done = opcode_len > 2

    # Check for conditional jumps (0x70-0x7F)
    is_70_7F = (primary_opcode >= 0x70) & (primary_opcode <= 0x7F) 
    flow_kind = jnp.where(is_70_7F & (opcode_len == 1) & ~done,
                          5,  # COND_JUMP
                          flow_kind)
    done = done | is_70_7F

    # Check for conditional jumps (0x80-0x8F)
    is_80_8F = (primary_opcode >= 0x80) & (primary_opcode <= 0x8F)
    flow_kind = jnp.where(is_80_8F & (opcode_len == 2) & ~done,
                          5,  # COND_JUMP
                          flow_kind)
    done = done | is_80_8F

    # Handle specific opcodes
    flow_kind = jnp.where((primary_opcode == 0x9A) & (opcode_len == 1) & ~done,
                          6,  # FAR_CALL
                          flow_kind)
    done = done | ((primary_opcode == 0x9A) & (opcode_len == 1))

    # Handle 0xFF opcode with different modrm.reg values
    is_FF = (primary_opcode == 0xFF) & (opcode_len == 1)
    modrm_reg = (modrm >> 3) & 7

    flow_kind = jnp.where(is_FF & (modrm_reg == 2) & ~done,
                          2,  # CALL
                          flow_kind)
    done = done | (is_FF & (modrm_reg == 2))
    
    flow_kind = jnp.where(is_FF & (modrm_reg == 3) & ~done,
                          6,  # FAR_CALL
                          flow_kind)
    done = done | (is_FF & (modrm_reg == 3))

    flow_kind = jnp.where(is_FF & (modrm_reg == 4) & ~done,
                          4,  # JUMP
                          flow_kind)
    done = done | (is_FF & (modrm_reg == 4))

    flow_kind = jnp.where(is_FF & (modrm_reg == 5) & ~done,
                          8,  # FAR_JUMP
                          flow_kind)
    done = done | (is_FF & (modrm_reg == 5))

    # Check for CALL instructions
    flow_kind = jnp.where((primary_opcode == 0xE8) & (opcode_len == 1) & ~done,
                          2,  # CALL
                          flow_kind)
    done = done | ((primary_opcode == 0xE8) & (opcode_len == 1))

    # Check for FAR_CALL instructions
    is_far_call = ((primary_opcode == 0xCD) | (primary_opcode == 0xCC) |
                   (primary_opcode == 0xCE) | (primary_opcode == 0xF1)) & (opcode_len == 1)
    flow_kind = jnp.where(is_far_call & ~done,
                          6,  # FAR_CALL
                          flow_kind)
    done = done | is_far_call

    # Check for FAR_RETURN instructions
    flow_kind = jnp.where((primary_opcode == 0xCF) & (opcode_len == 1) & ~done,
                          7,  # FAR_RETURN
                          flow_kind)
    done = done | ((primary_opcode == 0xCF) & (opcode_len == 1))

    # Check for JUMP instructions
    is_jump = ((primary_opcode == 0xE9) | (
        primary_opcode == 0xEB)) & (opcode_len == 1)
    flow_kind = jnp.where(is_jump & ~done,
                          4,  # JUMP
                          flow_kind)
    done = done | is_jump

    # Check for FAR_JUMP instructions
    flow_kind = jnp.where((primary_opcode == 0xEA) & (opcode_len == 1) & ~done,
                          8,  # FAR_JUMP
                          flow_kind)
    done = done | ((primary_opcode == 0xEA) & (opcode_len == 1))

    # Check for COND_JUMP instructions
    is_cond_jump = ((primary_opcode == 0xE3) | (primary_opcode == 0xE0) |
                    (primary_opcode == 0xE1) | (primary_opcode == 0xE2)) & (opcode_len == 1)
    flow_kind = jnp.where(is_cond_jump & ~done,
                          5,  # COND_JUMP
                          flow_kind)
    done = done | is_cond_jump

    # Check for RETURN instructions
    is_return = ((primary_opcode == 0xC3) | (
        primary_opcode == 0xC2)) & (opcode_len == 1)
    flow_kind = jnp.where(is_return & ~done,
                          3,  # RETURN
                          flow_kind)
    done = done | is_return

    # Check for FAR_RETURN instructions
    is_far_return = ((primary_opcode == 0xCB) | (
        primary_opcode == 0xCA)) & (opcode_len == 1)
    flow_kind = jnp.where(is_far_return & ~done,
                          7,  # FAR_RETURN
                          flow_kind)
    done = done | is_far_return

    # Check for additional FAR_CALL instructions
    is_far_call_2 = ((primary_opcode == 0x05) | (
        primary_opcode == 0x34)) & (opcode_len == 2)
    flow_kind = jnp.where(is_far_call_2 & ~done,
                          6,  # FAR_CALL
                          flow_kind)
    done = done | is_far_call_2

    # Check for additional FAR_RETURN instructions
    is_far_return_2 = ((primary_opcode == 0x35) | (
        primary_opcode == 0x07)) & (opcode_len == 2)
    flow_kind = jnp.where(is_far_return_2 & ~done,
                          7,  # FAR_RETURN
                          flow_kind)
    done = done | is_far_return_2

    # Handle opcode 0x01 with specific modrm values
    is_01 = (primary_opcode == 0x01) & (opcode_len == 2)

    flow_kind = jnp.where(is_01 & (modrm == 0xc1) & ~done,
                          6,  # FAR_CALL
                          flow_kind)
    done = done | (is_01 & (modrm == 0xc1))

    is_far_return_3 = is_01 & ((modrm == 0xc2) | (modrm == 0xc3))
    flow_kind = jnp.where(is_far_return_3 & ~done,
                          7,  # FAR_RETURN
                          flow_kind)
    done = done | is_far_return_3

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
    result = jnp.sum(selected_bytes * (1 << (indices * 8)),
                     axis=0, dtype=jnp.int32)
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
    overlapping = jnp.where(instr_len[:, jnp.newaxis] > ranges[jnp.newaxis, :],
                            offset[:, jnp.newaxis] + ranges[jnp.newaxis, :], -1)
    # must target, based on flow kind, next for other, -1 for unconditional jumps, next + imm for unconditional jumps and calls
    must_target = jnp.where(flow_kind == 1, next_instr_addr, -1)
    must_target = jnp.where((((flow_kind == 2) | (flow_kind == 4)) & (
        imm_size != 0)), next_instr_addr + imm, must_target)
    # may target, based on flow kind, only for conditional jumps
    may_target = jnp.where((flow_kind == 5)[:, jnp.newaxis], jnp.stack(
        (next_instr_addr, next_instr_addr + imm), axis=-1), -1)

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
    control_flow = jnp.where((control_flow < 0) | (
        control_flow >= instr_bytes.shape[0]), -1, control_flow)
    return instruction, control_flow, instr_len

class DisasmJax(nnx.Module):
    def __init__(self):
        super().__init__()
        self.flags_table = nnx.Variable(jnp.array(flags_table, dtype=jnp.uint8))
        self.flags_table_ex = nnx.Variable(
            jnp.array(flags_table_ex, dtype=jnp.uint8))

    def parse_prefix(self, code_byte, state, p, s, pr_66, pr_67):
        """Parse the prefix of the instruction.

        Args:
            code_byte: The current byte of the instruction
            state: The current state of the disassembly
        """
        def cond_fn(state_tuple, code_byte):
            state, p, s, pr_66, pr_67 = state_tuple
            code = code_byte[p]
            return ((self.flags_table[code] & OP_PREFIX) != 0) & (s < 15)

        def body_fn(state_tuple, code_byte):
            state, p, s, pr_66, pr_67 = state_tuple
            code_byte = get_byte_at(code_byte, p)
            pr_66_new = jnp.where(
                code_byte == 0x66, jnp.array(1, dtype=jnp.uint8), pr_66)
            pr_67_new = jnp.where(
                code_byte == 0x67, jnp.array(1, dtype=jnp.uint8), pr_67)
            # flags is at state[1]
            new_state = state.at[1].set(state[1] | F_PREFIX)
            new_state = jnp.where(s == 15, new_state.at[0].set(
                new_state[1] | F_INVALID), new_state)
            return new_state, p + 1, s + 1, pr_66_new, pr_67_new

        state, p, s, pr_66, pr_67 = lax.while_loop(
            lambda state_tuple: cond_fn(state_tuple, code_byte),
            lambda state_tuple: body_fn(state_tuple, code_byte),
            (state, p, s, pr_66, pr_67))
        return state, p, s, pr_66, pr_67

    def parse_rex(self, code_byte, state, p, s, rexw):
        """Parse the REX prefix of the instruction.

        Args:
            code_byte: The current byte of the instruction
            state: The current state of the disassembly
        """
        code = code_byte[p]
        cond = (code >> 4) == 4
        p = jnp.where(cond, p + 1, p)
        s = jnp.where(cond, s + 1, s)
        state = jnp.where(cond, state.at[1].set(state[1] | F_REX), state)
        state = jnp.where(cond, state.at[2].set(code), state)
        rexw = jnp.where(cond, (code >> 3) & 1, rexw)

        code = code_byte[p]
        state = jnp.where((code >> 4) == 4, state.at[1].set(
            state[1] | F_INVALID), state)
        s = jnp.where((code >> 4) == 4, s+1, s)
        return state, p, s, rexw

    def parse_opcode(self, code_byte, state, p, s, pr_66, pr_67):
        """Parse the opcode of the instruction.

        Args:
            code_byte: The current byte of the instruction
            state: The current state of the disassembly
            p: The current position in the instruction
            s: The current size of the instruction
            pr_66: The current value of pr_66
            pr_67: The current value of pr_67
        """
        # opcd_offset is at state[5]
        state = state.at[5].set(p)
        # opcd_size is at state[6]
        state = state.at[6].set(1)
        op = code_byte[p]
        p += 1
        s += 1

        def handle_2byte_opcode(code_byte, state, p, s, op, pr_66):
            op = code_byte[p]
            p += 1
            s += 1
            state = state.at[6].set(2)
            f = self.flags_table_ex[op]
            state = jnp.where((f & OP_INVALID) != 0, state.at[1].set(
                state[1] | F_INVALID), state)
            valid = (f & OP_INVALID) == 0
            has_extended = (f & OP_EXTENDED) != 0
            cond = valid & has_extended
            op = jnp.where(cond, code_byte[p], op)
            p = jnp.where(cond, p + 1, p)
            s = jnp.where(cond, s + 1, s)
            state = jnp.where(cond, state.at[6].set(3), state)
            return state, op, f, p, s, pr_66

        def handle_others(code_byte, state, p, s, op, pr_66):
            f = self.flags_table[op]
            pr_66 = jnp.where((op >= np.array(0xA0, dtype=jnp.uint8)) & (
                op <= np.array(0xA3, dtype=jnp.uint8)), pr_67, pr_66)
            return state, op, f, p, s, pr_66

        return lax.cond(op == 0x0F,
                        lambda: handle_2byte_opcode(
                            code_byte, state, p, s, op, pr_66),
                        lambda: handle_others(code_byte, state, p, s, op, pr_66))

    def parse_modrm(self, code_byte, state, op, f, p, s, pr_67, is64):
        """Parse the ModR/M byte of the instruction.

        Args:
            code_byte: The current byte of the instruction
            state: The current state of the disassembly
        """
        code = code_byte[p]
        mod = (code >> 6)
        ro = (code & 0x38) >> 3
        rm = code & 0x7

        # modrm is at state[3]
        state = state.at[3].set(code)
        p += 1
        s += 1
        # flags is at state[1]
        state = state.at[1].set(state[1] | F_MODRM)
        f = jnp.where(((op == 0xF6) & ((ro == 0) | (ro == 1))),
                      f | OP_DATA_I8, f)
        f = jnp.where(((op == 0xF7) & ((ro == 0) | (ro == 1))),
                      f | OP_DATA_I16_I32_I64, f)

        cond = (mod != 3) & (rm == 4) & ((pr_67 == 0) | is64)

        def handle_sib(state, p, s):
            # sib is at state[4]
            code = code_byte[p]
            state = state.at[4].set(code)
            p += 1
            s += 1
            # flags is at state[1]
            state = state.at[1].set(state[1] | F_SIB)
            disp_size = jnp.where(((state[4] & 7) == 5) & (mod == 0), 4, 0)
            state = state.at[8].set(disp_size)
            return state, p, s

        state, p, s = lax.cond(cond, lambda: handle_sib(
            state, p, s), lambda: (state, p, s))

        def handle_mod0(state):
            # disp_size is at state[8]
            cond = is64 & (rm == 5)
            state = jnp.where(cond, state.at[8].set(4), state)
            state = jnp.where(cond, state.at[1].set(
                state[1] | F_RELATIVE), state)
            cond = (~is64) & pr_67 & (rm == 6)
            state = jnp.where(cond, state.at[8].set(2), state)
            cond = (~is64) & (pr_67 == 0) & (rm == 5)
            state = jnp.where(cond, state.at[8].set(4), state)
            return state

        def handle_mod1(state):
            state = state.at[8].set(1)
            return state

        def handle_mod2(state):
            cond = is64
            state = jnp.where(cond, state.at[8].set(4), state)
            cond = (~is64) & pr_67
            state = jnp.where(cond, state.at[8].set(2), state)
            cond = (~is64) & (pr_67 == 0)
            state = jnp.where(cond, state.at[8].set(4), state)
            return state

        state = lax.switch(mod, [
            lambda: handle_mod0(state),
            lambda: handle_mod1(state),
            lambda: handle_mod2(state),
            lambda: state
        ])
        # disp_offset is at state[7]
        state = jnp.where(state[8] == 0, state, state.at[7].set(p))
        p += jnp.where(state[8] == 0, 0, state[8])
        s += jnp.where(state[8] == 0, 0, state[8])
        state = jnp.where(state[8] == 0, state,
                          state.at[1].set(state[1] | F_DISP))
        return state, p, s, f

    def parse_immediate(self, state, p, s, f, rexw, is64, op, pr_66):
        """Parse the immediate data of the instruction.

        Args:
            state: The current state of the disassembly
            p: The current position in the instruction
            s: The current size of the instruction
            f: The current flags
            rexw: The current value of rexw
            is64: The current value of is64
            op: The current opcode
        """
        # imm_size is at state[10]
        cond_imm8 = ((rexw != 0) | (is64 & (op >= np.array(0xA0, dtype=jnp.uint8)) & (
            op <= np.array(0xA3, dtype=jnp.uint8)))) & ((f & OP_DATA_I16_I32_I64) != 0)
        cond_imm4 = ((f & OP_DATA_I16_I32) != 0) | (
            (f & OP_DATA_I16_I32_I64) != 0)
        state = lax.cond(
            cond_imm8,
            lambda: state.at[10].set(8),
            lambda: lax.cond(
                cond_imm4,
                lambda: state.at[10].set(4 - (pr_66 << 1)),
                lambda: state,
            ))
        state = state.at[10].set(state[10] + (f & 3))
        has_imm = state[10] > 0
        s = jnp.where(has_imm, s + state[10], s)
        # imm_offset is at state[9]
        state = jnp.where(has_imm, state.at[9].set(p), state)
        state = jnp.where(has_imm, state.at[1].set(state[1] | F_IMM), state)
        state = jnp.where(has_imm & (f & OP_RELATIVE != 0),
                          state.at[1].set(state[1] | F_RELATIVE), state)

        # instruction is too long
        state = jnp.where(s > 15, state.at[1].set(state[1] | F_INVALID), state)
        return state, s

    def disasm(self, code_byte, use_64_bit):
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
        # Create a struct-like array to hold the ldasm_data
        # We will use this state to accumulate the disassembly information
        state = jnp.zeros(13, dtype=jnp.uint8)

        p = jnp.array(0, dtype=jnp.uint8)
        s = jnp.array(0, dtype=jnp.uint8)  # instruction size accumulator
        op = jnp.array(0, dtype=jnp.uint8)
        f = jnp.array(0, dtype=jnp.uint8)
        rexw = jnp.array(0, dtype=jnp.uint8)
        pr_66 = jnp.array(0, dtype=jnp.uint8)
        pr_67 = jnp.array(0, dtype=jnp.uint8)

        # Phase 1: Parse prefixes
        state, p, s, pr_66, pr_67 = self.parse_prefix(
            code_byte, state, p, s, pr_66, pr_67)

        # Phase 2: Parse REX prefix
        valid = (state[1] & F_INVALID) == 0
        state, p, s, rexw = lax.cond(
            valid & use_64_bit,
            lambda: self.parse_rex(code_byte, state, p, s, rexw),
            lambda: (state, p, s, rexw))

        # Phase 3: Parse opcode
        valid = (state[1] & F_INVALID) == 0
        state, op, f, p, s, pr_66 = lax.cond(
            valid,
            lambda: self.parse_opcode(code_byte, state, p, s, pr_66, pr_67),
            lambda: (state, op, f, p, s, pr_66))

        # Phase 4: Parse ModR/M
        valid = (state[1] & F_INVALID) == 0
        state, p, s, f = lax.cond(
            valid & (f & OP_MODRM != 0),
            lambda: self.parse_modrm(
                code_byte, state, op, f, p, s, pr_67, use_64_bit),
            lambda: (state, p, s, f))

        # Phase 5: Parse immediate data
        valid = (state[1] & F_INVALID) == 0
        state, s = lax.cond(
            valid,
            lambda: self.parse_immediate(
                state, p, s, f, rexw, use_64_bit, op, pr_66),
            lambda: (state, s))

        state = state.at[0].set(s)
        state = state.at[11].set(op)
        return state

    def instr_len_decode(self, code_byte, use_64_bit):
        """Decode the instruction length from the code byte for flow kind.

        Args:
            code_byte: A JAX array containing the instruction bytes
            use_64bit: Boolean indicating if 64-bit mode is used

        Returns:
            A JAX array containing primary_opcode, opcode_len and modrm  
            state[0]: primary_opcode
            state[1]: opcode_len
            state[2]: modrm
        """
        state = jnp.zeros(3, dtype=jnp.uint8)
        p = jnp.array(0, dtype=jnp.uint8)
        done = jnp.array(0, dtype=jnp.bool)
        prefix_done = jnp.array(0, dtype=jnp.bool)

        def cond_fn(state_tuple):
            state, p, done, prefix_done = state_tuple
            return ~prefix_done & ~done

        def body_fn(state_tuple):
            state, p, done, prefix_done = state_tuple
            op = get_byte_at(code_byte, p)
            state = state.at[0].set(op)
            cond0 = (op == 0x26) | (op == 0x2E) | (op == 0x36) | (op == 0x3E) | (op == 0x64) | (
                op == 0x65) | (op == 0x66) | (op == 0x67) | (op == 0xF0) | (op == 0xF2) | (op == 0xF3)
            cond1 = (op >= 0x40) & (op <= 0x4F)
            cond2 = (op == 0xc5)
            cond3 = (op == 0xc4)
            cond4 = (op == 0x62)
            cond5 = ~cond0 & ~cond1 & ~cond2 & ~cond3 & ~cond4
            cond_idx = jnp.where(cond0, 0, jnp.where(cond1, 1, jnp.where(
                cond2, 2, jnp.where(cond3, 3, jnp.where(cond4, 4, 5)))))

            def handle_cond0(p, prefix_done, done, state, use_64_bit):
                return p + 1, prefix_done, done, state

            def handle_cond1(p, prefix_done, done, state, use_64_bit):
                p = jnp.where(use_64_bit, p + 1, p)
                prefix_done = jnp.where(use_64_bit, prefix_done, True)
                return p, prefix_done, done, state

            def handle_cond2(p, prefix_done, done, state, use_64_bit):
                cond = ~use_64_bit & ((code_byte[p + 1] & 0xc0) != 0xc0)

                def update_state(state):
                    state = state.at[0].set(code_byte[p + 2])
                    state = state.at[1].set(2)
                    state = state.at[2].set(code_byte[p + 3])
                    return state
                state = lax.cond(cond, lambda: state,
                                 lambda: update_state(state))
                prefix_done = jnp.where(cond, True, prefix_done)
                done = jnp.where(cond, done, True)
                return p, prefix_done, done, state

            def handle_cond3(p, prefix_done, done, state, use_64_bit):
                cond = ~use_64_bit & ((code_byte[p + 1] & 0xc0) != 0xc0)

                def update_state(state):
                    state = state.at[0].set(code_byte[p + 3])
                    state = state.at[1].set(code_byte[p+1] & 0x1f)
                    state = state.at[2].set(code_byte[p + 4])
                    return state
                state = lax.cond(cond, lambda: state,
                                 lambda: update_state(state))
                prefix_done = jnp.where(cond, True, prefix_done)
                done = jnp.where(cond, done, True)
                return p, prefix_done, done, state

            def handle_cond4(p, prefix_done, done, state, use_64_bit):
                cond = ~use_64_bit & ((code_byte[p + 1] & 0xc0) != 0xc0)

                def update_state(state):
                    state = state.at[0].set(code_byte[p + 4])
                    state = state.at[1].set(code_byte[p+1] & 0x03)
                    state = state.at[2].set(code_byte[p + 5])
                    return state
                state = lax.cond(cond, lambda: state,
                                 lambda: update_state(state))
                prefix_done = jnp.where(cond, True, prefix_done)
                done = jnp.where(cond, done, True)
                return p, prefix_done, done, state

            def handle_cond5(p, prefix_done, done, state, use_64_bit):
                prefix_done = jnp.array(True, dtype=jnp.bool)
                return p, prefix_done, done, state

            p, prefix_done, done, state = lax.switch(cond_idx,
                                                     [
                                                         lambda: handle_cond0(
                                                             p, prefix_done, done, state, use_64_bit),
                                                         lambda: handle_cond1(
                                                             p, prefix_done, done, state, use_64_bit),
                                                         lambda: handle_cond2(
                                                             p, prefix_done, done, state, use_64_bit),
                                                         lambda: handle_cond3(
                                                             p, prefix_done, done, state, use_64_bit),
                                                         lambda: handle_cond4(
                                                             p, prefix_done, done, state, use_64_bit),
                                                         lambda: handle_cond5(
                                                             p, prefix_done, done, state, use_64_bit),
                                                     ])
            return (state, p, done, prefix_done)
        (state, p, done, prefix_done) = lax.while_loop(
            cond_fn, body_fn, (state, p, done, prefix_done))

        def handle_others(p, state):
            state = state.at[0].set(code_byte[p])
            state = state.at[1].set(1)
            state = state.at[2].set(code_byte[p+1])

            def handle_op_0f(p, state):
                p += 1
                op = code_byte[p]
                state = state.at[0].set(code_byte[p])
                cond0 = op == 0x38
                cond1 = op == 0x3a
                cond2 = (op & 0xf8) == 0x38
                cond3 = op == 0x0f
                cond4 = ~cond0 & ~cond1 & ~cond2 & ~cond3
                cond_idx = jnp.where(cond0, 0, jnp.where(cond1, 1, jnp.where(
                    cond2, 2, jnp.where(cond3, 3, 4))))

                def handle_cond0(p, state):
                    p += 1
                    state = state.at[0].set(code_byte[p])
                    state = state.at[1].set(3)
                    state = state.at[2].set(code_byte[p+1])
                    return state

                def handle_cond1(p, state):
                    p += 1
                    state = state.at[0].set(code_byte[p])
                    state = state.at[1].set(3)
                    state = state.at[2].set(code_byte[p+1])
                    return state

                def handle_cond2(p, state):
                    p += 1
                    state = state.at[0].set(code_byte[p])
                    state = state.at[1].set(0)
                    state = state.at[2].set(code_byte[p+1])
                    return state

                def handle_cond3(p, state):
                    state = state.at[1].set(3)
                    state = state.at[2].set(code_byte[p+1])
                    return state

                def handle_cond4(p, state):
                    state = state.at[1].set(2)
                    state = state.at[2].set(code_byte[p+1])
                    return state
                state = lax.switch(cond_idx,
                                   [
                                       lambda: handle_cond0(p, state),
                                       lambda: handle_cond1(p, state),
                                       lambda: handle_cond2(p, state),
                                       lambda: handle_cond3(p, state),
                                       lambda: handle_cond4(p, state)
                                   ])
                return state
            state = lax.cond(state[0] == 0x0f,
                             lambda: handle_op_0f(p, state), lambda: state)
            return state
        state = lax.cond(done, lambda: state, lambda: handle_others(p, state))
        return state

    def seq_to_instr_bytes(self, code_bytes):
        """Convert a sequence of bytes to an instruction bytes array.

        Args:
            code_bytes: A JAX array containing the instruction bytes of length L
        """
        # Pad the byte sequence
        padded_bytes = jnp.pad(
            code_bytes, (0, 15 - 1), constant_values=0x90)

        # Create a function to extract instruction bytes
        def extract_instr_bytes(i):
            return lax.dynamic_slice_in_dim(padded_bytes, start_index=i, slice_size=15, axis=0)

        # Create positions array
        positions = jnp.arange(code_bytes.shape[0])

        # Map the extraction function over all positions
        instr_bytes = nnx.vmap(extract_instr_bytes)(positions)

        return instr_bytes

    def __call__(self, code_bytes, use_64_bit):
        """Disassemble a sequence of bytes.

        Args:
            code_bytes: A JAX array containing the instruction bytes
            use_64bit: Boolean indicating if 64-bit mode is used
        """
        instr_bytes = self.seq_to_instr_bytes(code_bytes)
        opcode_modrm = nnx.vmap(self.instr_len_decode, in_axes=(0, None))(instr_bytes, use_64_bit)
        disasm = nnx.vmap(self.disasm, in_axes=(0, None))(instr_bytes, use_64_bit)
        flow_kind = nnx.vmap(map_opcode_into_control_flow_kind)(opcode_modrm)
        instruction, control_flow, instr_len = parse_disasm(instr_bytes, disasm, flow_kind)
        return instruction, control_flow, instr_len, flow_kind


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
    # Renamed to avoid conflict if passed
    flags_table_static, flags_table_ex_static = create_flags_tables()
    # Create a struct-like array to hold the ldasm_data
    result = jnp.zeros(12, dtype=jnp.uint8)

    # Initialize all variables
    position = jnp.array(0, dtype=jnp.uint8)
    s = jnp.array(0, dtype=jnp.uint8)  # instruction size accumulator
    flags = jnp.array(0, dtype=jnp.uint8)
    rex = jnp.array(0, dtype=jnp.uint8)
    modrm = jnp.array(0, dtype=jnp.uint8)
    sib = jnp.array(0, dtype=jnp.uint8)
    opcd_offset = jnp.array(0, dtype=jnp.uint8)
    opcd_size = jnp.array(0, dtype=jnp.uint8)
    disp_offset = jnp.array(0, dtype=jnp.uint8)
    disp_size = jnp.array(0, dtype=jnp.uint8)  # This will be calculated
    imm_offset = jnp.array(0, dtype=jnp.uint8)
    imm_size = jnp.array(0, dtype=jnp.uint8)
    primary_opcode = jnp.array(0, dtype=jnp.uint8)  # For FlowKind

    # Prefix tracking
    rexw = jnp.array(0, dtype=jnp.uint8)
    pr_66 = jnp.array(0, dtype=jnp.uint8)
    pr_67 = jnp.array(0, dtype=jnp.uint8)

    # Phase 1: Parse prefixes
    def prefix_cond_fn(state_tuple):
        pos, _, _, _, _ = state_tuple
        current_byte_val = get_byte_at(code_bytes, pos)
        # Check if it's a prefix using the static table
        return (flags_table_static[current_byte_val] & OP_PREFIX) != 0

    def prefix_body_fn(state_tuple):
        pos, sz, flgs, p66, p67 = state_tuple
        current_byte_val = get_byte_at(code_bytes, pos)
        p66_new = jnp.where(current_byte_val == 0x66,
                            jnp.array(1, dtype=jnp.uint8), p66)
        p67_new = jnp.where(current_byte_val == 0x67,
                            jnp.array(1, dtype=jnp.uint8), p67)
        # Max prefix check (s == 15 in C)
        # This check is complex inside while_loop condition for early exit.
        # C code has `if (s == 15u) { ld->flags |= F_INVALID; ld->instr_len = s; return s; }`
        # JAX while_loop doesn't directly support early exit with setting flags like this easily.
        # The final check `s > 15` will catch it.
        return pos + 1, sz + 1, flgs | F_PREFIX, p66_new, p67_new

    position, s, flags, pr_66, pr_67 = lax.while_loop(
        prefix_cond_fn,
        prefix_body_fn,
        (position, s, flags, pr_66, pr_67)
    )

    # Phase 2: Parse REX prefix (64-bit mode only)
    current_byte_for_rex = get_byte_at(code_bytes, position)
    has_rex_cond = use_64_bit & ((current_byte_for_rex >> 4) == 4)

    def handle_rex_branch_fn(cb_rex, pos, sz, flgs):
        rex_val = cb_rex
        rexw_val = (rex_val >> 3) & 1
        return rex_val, rexw_val, pos + 1, sz + 1, flgs | F_REX

    def skip_rex_branch_fn(cb_rex, pos, sz, flgs):
        return jnp.array(0, dtype=jnp.uint8), jnp.array(0, dtype=jnp.uint8), pos, sz, flgs

    rex, rexw, position, s, flags = lax.cond(
        has_rex_cond,
        handle_rex_branch_fn,
        skip_rex_branch_fn,
        current_byte_for_rex, position, s, flags
    )

    # Check for invalid double REX
    current_byte_for_invalid_rex = get_byte_at(code_bytes, position)
    is_invalid_double_rex_cond = use_64_bit & has_rex_cond & (
        (current_byte_for_invalid_rex >> 4) == 4)

    def apply_invalid_rex_penalty_fn(pos, sz, flgs):
        return pos + 1, sz + 1, flgs | F_INVALID

    def no_penalty_fn(pos, sz, flgs):
        return pos, sz, flgs

    position, s, flags = lax.cond(
        is_invalid_double_rex_cond,
        apply_invalid_rex_penalty_fn,
        no_penalty_fn,
        position, s, flags
    )

    # Phase 3: Parse opcode
    opcd_offset = position
    op_byte1 = get_byte_at(code_bytes, position)

    # Operands for opcode parsing paths
    # (code_bytes_op, pos_op, s_op, opcd_sz_op, primary_op_val, ft_std_op, ft_ex_op, first_op_byte)

    pos_after_op1 = position + 1
    s_after_op1 = s + 1
    # current opcode size before decision
    opcd_size_val = jnp.array(1, dtype=jnp.uint8)

    # This primary_opcode is the candidate, will be updated by the cond
    current_primary_opcode = op_byte1

    is_2byte_cond = op_byte1 == 0x0F

    def handle_2byte_opcode_path_fn(cb_op, pos_op, s_op, opcd_sz_op, p_op_val, ft_ex_op):
        op_2nd_byte = get_byte_at(cb_op, pos_op)
        pos_after_op2 = pos_op + 1
        s_after_op2 = s_op + 1
        opcd_sz_after_op2 = jnp.array(2, dtype=jnp.uint8)

        # For 2-byte opcodes, the primary_opcode is the second byte
        p_op_for_2byte = op_2nd_byte

        f_from_ex_table = ft_ex_op[op_2nd_byte]
        has_extended_cond = (f_from_ex_table & OP_EXTENDED) != 0

        def handle_extended_op_branch(cb_ext, pos_ext, s_ext, opcd_sz_ext, f_val_ext, p_op_ext):
            # This is the 'op' for flags
            op_3rd_byte = get_byte_at(cb_ext, pos_ext)
            return (op_3rd_byte,
                    pos_ext + 1,
                    s_ext + 1,
                    opcd_sz_ext + 1,  # 3 bytes total
                    f_val_ext,       # Flags from 2nd byte (flags_table_ex)
                    (f_val_ext & OP_INVALID) != 0,
                    p_op_ext)        # Primary opcode remains 2nd byte

        def keep_regular_2byte_op_branch(cb_ext, pos_ext, s_ext, opcd_sz_ext, f_val_ext, p_op_ext):
            # op_2nd_byte is the 'op' for flags
            return (op_2nd_byte,
                    pos_ext,
                    s_ext,
                    opcd_sz_ext,     # 2 bytes total
                    f_val_ext,       # Flags from 2nd byte
                    (f_val_ext & OP_INVALID) != 0,
                    p_op_ext)        # Primary opcode is 2nd byte

        # Operands for inner cond: cb_op, pos_after_op2, s_after_op2, opcd_sz_after_op2, f_from_ex_table, p_op_for_2byte
        return lax.cond(
            has_extended_cond,
            handle_extended_op_branch,
            keep_regular_2byte_op_branch,
            cb_op, pos_after_op2, s_after_op2, opcd_sz_after_op2, f_from_ex_table, p_op_for_2byte
        )

    def keep_1byte_opcode_path_fn(cb_op, pos_op, s_op, opcd_sz_op, p_op_val, ft_std_op, first_op_b):
        # p_op_val is first_op_b
        return (first_op_b,  # 'op' for flags is the first byte
                pos_op,
                s_op,
                opcd_sz_op,  # 1 byte for opcode
                ft_std_op[first_op_b],
                # No OP_INVALID from flags_table for 1-byte opcodes
                jnp.array(False, dtype=jnp.bool_),
                p_op_val)  # Primary opcode is the first byte

    # Unpack: op_final, position_final, s_final, opcd_size_final, f_final, is_invalid_from_opcode, primary_opcode_final
    op, position, s, opcd_size, f_opcode_flags, is_invalid_from_opcode, primary_opcode = lax.cond(
        is_2byte_cond,
        lambda: handle_2byte_opcode_path_fn(
            code_bytes, pos_after_op1, s_after_op1, opcd_size_val, current_primary_opcode, flags_table_ex_static),
        lambda: keep_1byte_opcode_path_fn(code_bytes, pos_after_op1, s_after_op1,
                                          opcd_size_val, current_primary_opcode, flags_table_static, op_byte1)
    )

    flags = flags | (F_INVALID * is_invalid_from_opcode.astype(jnp.uint8))

    # Special case for opcodes A0-A3 (pr_66 = pr_67)
    # C uses `op` which is the first byte, or second if 0F. Here `op_byte1` is first byte.
    is_special_A0_A3_cond = (op_byte1 >= np.array(0xA0, dtype=jnp.uint8)) & (
        op_byte1 <= np.array(0xA3, dtype=jnp.uint8)) & (~is_2byte_cond)
    pr_66 = lax.cond(
        is_special_A0_A3_cond,
        lambda p66, p67: p67,
        lambda p66, p67: p66,
        pr_66, pr_67
    )

    # Phase 4: Parse ModR/M, SIB, and displacement
    has_modrm_cond = (f_opcode_flags & OP_MODRM) != 0

    # Initial disp_size before ModR/M logic
    current_disp_size = jnp.array(0, dtype=jnp.uint8)

    def handle_modrm_path_fn(cb_mod, pos_mod, s_mod, flgs_mod, f_op_flgs, op_val_mod):
        modrm_byte_val = get_byte_at(cb_mod, pos_mod)
        mod_val = modrm_byte_val >> 6
        ro_val = (modrm_byte_val & 0x38) >> 3
        rm_val = modrm_byte_val & 0x7

        is_f6_cond = (op_val_mod == 0xF6) & ((ro_val == 0) | (ro_val == 1))
        is_f7_cond = (op_val_mod == 0xF7) & ((ro_val == 0) | (ro_val == 1))

        f_updated = f_op_flgs | \
            (OP_DATA_I8 * is_f6_cond.astype(jnp.uint8)) | \
            (OP_DATA_I16_I32_I64 * is_f7_cond.astype(jnp.uint8))

        new_flgs = flgs_mod | F_MODRM
        new_pos = pos_mod + 1
        new_s = s_mod + 1

        # SIB byte logic
        needs_sib_check = (mod_val != 3) & (rm_val == 4)
        is_64_or_no_pr67_sib = use_64_bit | (pr_67 == 0)
        has_sib_sub_cond = needs_sib_check & is_64_or_no_pr67_sib

        # Initial disp_size for SIB/Disp block is 0
        disp_sz_before_sib = jnp.array(0, dtype=jnp.uint8)

        def handle_sib_sub_path(cb_sib, pos_sib, s_sib, flgs_sib, d_sz_sib, mod_for_sib):
            sib_byte_val = get_byte_at(cb_sib, pos_sib)
            is_special_sib_cond = ((sib_byte_val & 7) ==
                                   5) & (mod_for_sib == 0)
            disp_sz_updated_sib = jnp.where(
                is_special_sib_cond, jnp.array(4, dtype=jnp.uint8), d_sz_sib)
            return sib_byte_val, pos_sib + 1, s_sib + 1, flgs_sib | F_SIB, disp_sz_updated_sib

        def skip_sib_sub_path(cb_sib, pos_sib, s_sib, flgs_sib, d_sz_sib, mod_for_sib):
            return jnp.array(0, dtype=jnp.uint8), pos_sib, s_sib, flgs_sib, d_sz_sib

        sib_val, pos_after_sib, s_after_sib, flgs_after_sib, disp_sz_after_sib = lax.cond(
            has_sib_sub_cond,
            lambda: handle_sib_sub_path(
                cb_mod, new_pos, new_s, new_flgs, disp_sz_before_sib, mod_val),
            lambda: skip_sib_sub_path(
                cb_mod, new_pos, new_s, new_flgs, disp_sz_before_sib, mod_val)
        )

        # Displacement logic
        # Operands: disp_s_in, use_64b_in, rm_in, pr_67_in, current_flgs_in
        def mod0_disp_branch(d_sz, u64, rm_v_disp, pr67_disp, flgs_d_in):
            s0_res = d_sz
            s0_64 = jnp.where(rm_v_disp == 5, jnp.array(4, jnp.uint8), s0_res)
            flags_delta_64 = jnp.where(
                rm_v_disp == 5, F_RELATIVE, jnp.array(0, jnp.uint8))

            s0_32_pr67 = jnp.where(
                rm_v_disp == 6, jnp.array(2, jnp.uint8), s0_res)
            s0_32_nopr67 = jnp.where(
                rm_v_disp == 5, jnp.array(4, jnp.uint8), s0_res)
            s0_32 = jnp.where(pr67_disp == 1, s0_32_pr67, s0_32_nopr67)

            final_s = jnp.where(u64, s0_64, s0_32)
            final_fd = jnp.where(u64, flags_delta_64, jnp.array(0, jnp.uint8))
            return final_s, flgs_d_in | final_fd

        def mod1_disp_branch(d_sz, u64, rm_v_disp, pr67_disp, flgs_d_in):
            return jnp.array(1, jnp.uint8), flgs_d_in

        def mod2_disp_branch(d_sz, u64, rm_v_disp, pr67_disp, flgs_d_in):
            s2 = jnp.where(u64, jnp.array(4, jnp.uint8),
                           jnp.where(pr67_disp == 1, jnp.array(2, jnp.uint8), jnp.array(4, jnp.uint8)))
            return s2, flgs_d_in

        def mod3_disp_branch(d_sz, u64, rm_v_disp, pr67_disp, flgs_d_in):
            return d_sz, flgs_d_in

        disp_size_final_modrm, flgs_after_disp_calc = lax.switch(
            mod_val,
            [mod0_disp_branch, mod1_disp_branch,
                mod2_disp_branch, mod3_disp_branch],
            # Operands:
            disp_sz_after_sib, use_64_bit, rm_val, pr_67, flgs_after_sib
        )

        has_disp_final_cond = disp_size_final_modrm > 0
        disp_off_final = jnp.where(
            has_disp_final_cond, pos_after_sib, jnp.array(0, dtype=jnp.uint8))
        pos_final_modrm = pos_after_sib + disp_size_final_modrm
        s_final_modrm = s_after_sib + disp_size_final_modrm
        flgs_final_modrm = flgs_after_disp_calc | (
            F_DISP * has_disp_final_cond.astype(jnp.uint8))

        return (modrm_byte_val, sib_val, pos_final_modrm, s_final_modrm, flgs_final_modrm,
                f_updated, disp_off_final, disp_size_final_modrm)

    def skip_modrm_path_fn(cb_mod, pos_mod, s_mod, flgs_mod, f_op_flgs, op_val_mod):
        # No ModR/M: modrm, sib, disp_offset, disp_size remain 0. Flags, f, position, s are unchanged from this block.
        return (jnp.array(0, dtype=jnp.uint8),  # modrm
                jnp.array(0, dtype=jnp.uint8),  # sib
                pos_mod, s_mod, flgs_mod, f_op_flgs,
                jnp.array(0, dtype=jnp.uint8),  # disp_offset
                jnp.array(0, dtype=jnp.uint8))  # disp_size

    # modrm, sib, position, s, flags, f_opcode_flags, disp_offset, disp_size
    modrm, sib, position, s, flags, f_opcode_flags, disp_offset, disp_size = lax.cond(
        has_modrm_cond,
        lambda: handle_modrm_path_fn(
            code_bytes, position, s, flags, f_opcode_flags, op),
        lambda: skip_modrm_path_fn(
            code_bytes, position, s, flags, f_opcode_flags, op)
    )

    # Phase 5: Parse immediate data
    # Use op_byte1 for A0-A3 check for immediate sizing as per C context.
    cond_op_A0_A3_for_imm_size = use_64_bit & (
        op_byte1 >= 0xA0) & (op_byte1 <= 0xA3) & (~is_2byte_cond)

    has_imm64_sub_cond = (rexw.astype(jnp.bool_) | cond_op_A0_A3_for_imm_size) & \
                         ((f_opcode_flags & OP_DATA_I16_I32_I64) != 0)
    has_imm32_16_sub_cond = ((f_opcode_flags & OP_DATA_I16_I32) != 0) | \
                            ((f_opcode_flags & OP_DATA_I16_I32_I64) != 0)

    def calc_imm_size_64_fn(): return jnp.array(8, dtype=jnp.uint8)
    def calc_imm_size_32_16_fn(): return jnp.array(4, dtype=jnp.uint8) - (pr_66 << 1)
    def calc_imm_size_else_fn(): return jnp.array(0, dtype=jnp.uint8)

    imm_size_base = lax.cond(
        has_imm64_sub_cond,
        calc_imm_size_64_fn,
        lambda: lax.cond(
            has_imm32_16_sub_cond,
            calc_imm_size_32_16_fn,
            calc_imm_size_else_fn
        )
    )
    imm_size = imm_size_base + (f_opcode_flags & 3)

    has_imm_cond = imm_size > 0

    def handle_imm_path_fn(pos_imm, flgs_imm, f_op_flgs_imm):
        imm_off_val = pos_imm
        new_flgs_imm = flgs_imm | F_IMM
        new_flgs_imm = jnp.where(
            (f_op_flgs_imm & OP_RELATIVE) != 0, new_flgs_imm | F_RELATIVE, new_flgs_imm)
        return imm_off_val, new_flgs_imm

    def no_imm_path_fn(pos_imm, flgs_imm, f_op_flgs_imm):
        return jnp.array(0, dtype=jnp.uint8), flgs_imm

    imm_offset, flags = lax.cond(
        has_imm_cond,
        lambda: handle_imm_path_fn(position, flags, f_opcode_flags),
        lambda: no_imm_path_fn(position, flags, f_opcode_flags)
    )

    s = s + imm_size  # Final update to instruction size

    # Check for too long instruction
    flags = flags | (F_INVALID * (s > 15).astype(jnp.uint8))

    # In C, if F_INVALID is set, `s` is returned early. JAX returns full result.
    # If flags has F_INVALID, `s` might be the position of invalidity, not full instr length.
    # C: `ld->instr_len = s;`
    # This seems okay, `s` is the total accumulated length.

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
    result = result.at[11].set(primary_opcode)  # primary opcode for flow kind

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
    padded_bytes = jnp.pad(
        byte_sequence, (0, max_instr_len - 1), constant_values=0x90)

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
def jax_disasm_sequence(byte_sequence, use_64_bit):
    """Disassemble a byte sequence into instruction bytes.

    Args:
        byte_sequence: A JAX array of bytes (uint8) with shape [L]
        use_64_bit: Boolean indicating if 64-bit mode is used

    Returns:
        A JAX array of shape [L, 12] containing disassembly info for each position
    """
    instr_bytes = byte_sequence_to_instr_bytes(byte_sequence)
    disasm_result = disasm_jax(instr_bytes, use_64_bit)
    return disasm_result


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
    instruction, control_flow, instr_len = parse_disasm(
        instr_bytes, disasm_result, flow_kind)

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
        flag_strings = [name for flag,
                        name in flag_names.items() if flag_value & flag]

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
    example_bytes = jnp.array(np.random.randint(0, 256, size=(
        32, 1024 * 1024), dtype=np.uint8), dtype=jnp.uint8)
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
