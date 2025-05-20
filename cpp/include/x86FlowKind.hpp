#pragma once
#include <cstdint>
#include <optional>

// The following code is borrowed from llvm-project's lldb.

/// Architecture-agnostic categorization of instructions for traversing the
/// control flow of a trace.
///
/// A single instruction can match one or more of these categories.
enum InstructionControlFlowKind {
  /// The instruction could not be classified.
  eInstructionControlFlowKindUnknown = 0,
  /// The instruction is something not listed below, i.e. it's a sequential
  /// instruction that doesn't affect the control flow of the program.
  eInstructionControlFlowKindOther,
  /// The instruction is a near (function) call.
  eInstructionControlFlowKindCall,
  /// The instruction is a near (function) return.
  eInstructionControlFlowKindReturn,
  /// The instruction is a near unconditional jump.
  eInstructionControlFlowKindJump,
  /// The instruction is a near conditional jump.
  eInstructionControlFlowKindCondJump,
  /// The instruction is a call-like far transfer.
  /// E.g. SYSCALL, SYSENTER, or FAR CALL.
  eInstructionControlFlowKindFarCall,
  /// The instruction is a return-like far transfer.
  /// E.g. SYSRET, SYSEXIT, IRET, or FAR RET.
  eInstructionControlFlowKindFarReturn,
  /// The instruction is a jump-like far transfer.
  /// E.g. FAR JMP.
  eInstructionControlFlowKindFarJump,
  /// The instruction halts or irreversibly disrupts execution without transferring control.
  /// Examples: HLT.
  eInstructionControlFlowKindHalt
};

/// These are the three values deciding instruction control flow kind.
/// InstructionLengthDecode function decodes an instruction and get this struct.
///
/// primary_opcode
///    Primary opcode of the instruction.
///    For one-byte opcode instruction, it's the first byte after prefix.
///    For two- and three-byte opcodes, it's the second byte.
///
/// opcode_len
///    The length of opcode in bytes. Valid opcode lengths are 1, 2, or 3.
///
/// modrm
///    ModR/M byte of the instruction.
///    Bits[7:6] indicate MOD. Bits[5:3] specify a register and R/M bits[2:0]
///    may contain a register or specify an addressing mode, depending on MOD.
struct InstructionOpcodeAndModrm {
  uint8_t primary_opcode;
  uint8_t opcode_len;
  uint8_t modrm;
};

/// Determine the InstructionControlFlowKind based on opcode and modrm bytes.
/// Refer to http://ref.x86asm.net/coder.html for the full list of opcode and
/// instruction set.
///
/// \param[in] opcode_and_modrm
///    Contains primary_opcode byte, its length, and ModR/M byte.
///    Refer to the struct InstructionOpcodeAndModrm for details.
///
/// \return
///   The control flow kind of the instruction or
///   eInstructionControlFlowKindOther if the instruction doesn't affect
///   the control flow of the program.
inline InstructionControlFlowKind
MapOpcodeIntoControlFlowKind(InstructionOpcodeAndModrm opcode_and_modrm) {
  uint8_t opcode = opcode_and_modrm.primary_opcode;
  uint8_t opcode_len = opcode_and_modrm.opcode_len;
  uint8_t modrm = opcode_and_modrm.modrm;

  if (opcode_len > 2)
    return eInstructionControlFlowKindOther;

  if (opcode >= 0x70 && opcode <= 0x7F) {
    if (opcode_len == 1)
      return eInstructionControlFlowKindCondJump;
    else
      return eInstructionControlFlowKindOther;
  }

  if (opcode >= 0x80 && opcode <= 0x8F) {
    if (opcode_len == 2)
      return eInstructionControlFlowKindCondJump;
    else
      return eInstructionControlFlowKindOther;
  }

  switch (opcode) {
  case 0xf4:
    if (opcode_len == 1)
      return eInstructionControlFlowKindHalt;
    break;
  case 0x9A:
    if (opcode_len == 1)
      return eInstructionControlFlowKindFarCall;
    break;
  case 0xFF:
    if (opcode_len == 1) {
      uint8_t modrm_reg = (modrm >> 3) & 7;
      if (modrm_reg == 2)
        return eInstructionControlFlowKindCall;
      else if (modrm_reg == 3)
        return eInstructionControlFlowKindFarCall;
      else if (modrm_reg == 4)
        return eInstructionControlFlowKindJump;
      else if (modrm_reg == 5)
        return eInstructionControlFlowKindFarJump;
    }
    break;
  case 0xE8:
    if (opcode_len == 1)
      return eInstructionControlFlowKindCall;
    break;
  case 0xCD:
  case 0xCC:
  case 0xCE:
  case 0xF1:
    if (opcode_len == 1)
      return eInstructionControlFlowKindFarCall;
    break;
  case 0xCF:
    if (opcode_len == 1)
      return eInstructionControlFlowKindFarReturn;
    break;
  case 0xE9:
  case 0xEB:
    if (opcode_len == 1)
      return eInstructionControlFlowKindJump;
    break;
  case 0xEA:
    if (opcode_len == 1)
      return eInstructionControlFlowKindFarJump;
    break;
  case 0xE3:
  case 0xE0:
  case 0xE1:
  case 0xE2:
    if (opcode_len == 1)
      return eInstructionControlFlowKindCondJump;
    break;
  case 0xC3:
  case 0xC2:
    if (opcode_len == 1)
      return eInstructionControlFlowKindReturn;
    break;
  case 0xCB:
  case 0xCA:
    if (opcode_len == 1)
      return eInstructionControlFlowKindFarReturn;
    break;
  case 0x05:
  case 0x34:
    if (opcode_len == 2)
      return eInstructionControlFlowKindFarCall;
    break;
  case 0x35:
  case 0x07:
    if (opcode_len == 2)
      return eInstructionControlFlowKindFarReturn;
    break;
  case 0x01:
    if (opcode_len == 2) {
      switch (modrm) {
      case 0xc1:
        return eInstructionControlFlowKindFarCall;
      case 0xc2:
      case 0xc3:
        return eInstructionControlFlowKindFarReturn;
      default:
        break;
      }
    }
    break;
  case 0x0b:
    if (opcode_len == 2) {
      return eInstructionControlFlowKindHalt;
    }
    break;
  default:
    break;
  }

  return eInstructionControlFlowKindOther;
}

/// Decode an instruction into opcode, modrm and opcode_len.
/// Refer to http://ref.x86asm.net/coder.html for the instruction bytes layout.
/// Opcodes in x86 are generally the first byte of instruction, though two-byte
/// instructions and prefixes exist. ModR/M is the byte following the opcode
/// and adds additional information for how the instruction is executed.
///
/// \param[in] inst_bytes
///    Raw bytes of the instruction
///
///
/// \param[in] bytes_len
///    The length of the inst_bytes array.
///
/// \param[in] is_exec_mode_64b
///    If true, the execution mode is 64 bit.
///
/// \return
///    Returns decoded instruction as struct InstructionOpcodeAndModrm, holding
///    primary_opcode, opcode_len and modrm byte. Refer to the struct definition
///    for more details.
///    Otherwise if the given instruction is invalid, returns std::nullopt.
inline std::optional<InstructionOpcodeAndModrm>
InstructionLengthDecode(const uint8_t *inst_bytes, int bytes_len,
                        bool is_exec_mode_64b) {
  int op_idx = 0;
  bool prefix_done = false;
  InstructionOpcodeAndModrm ret = {0, 0, 0};

  // In most cases, the primary_opcode is the first byte of the instruction
  // but some instructions have a prefix to be skipped for these calculations.
  // The following mapping is inspired from libipt's instruction decoding logic
  // in `src/pt_ild.c`
  while (!prefix_done) {
    if (op_idx >= bytes_len)
      return std::nullopt;

    ret.primary_opcode = inst_bytes[op_idx];
    switch (ret.primary_opcode) {
    // prefix_ignore
    case 0x26:
    case 0x2e:
    case 0x36:
    case 0x3e:
    case 0x64:
    case 0x65:
    // prefix_osz, prefix_asz
    case 0x66:
    case 0x67:
    // prefix_lock, prefix_f2, prefix_f3
    case 0xf0:
    case 0xf2:
    case 0xf3:
      op_idx++;
      break;

    // prefix_rex
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4a:
    case 0x4b:
    case 0x4c:
    case 0x4d:
    case 0x4e:
    case 0x4f:
      if (is_exec_mode_64b)
        op_idx++;
      else
        prefix_done = true;
      break;

    // prefix_vex_c4, c5
    case 0xc5:
      if (!is_exec_mode_64b && (inst_bytes[op_idx + 1] & 0xc0) != 0xc0) {
        prefix_done = true;
        break;
      }

      ret.opcode_len = 2;
      ret.primary_opcode = inst_bytes[op_idx + 2];
      ret.modrm = inst_bytes[op_idx + 3];
      return ret;

    case 0xc4:
      if (!is_exec_mode_64b && (inst_bytes[op_idx + 1] & 0xc0) != 0xc0) {
        prefix_done = true;
        break;
      }
      ret.opcode_len = inst_bytes[op_idx + 1] & 0x1f;
      ret.primary_opcode = inst_bytes[op_idx + 3];
      ret.modrm = inst_bytes[op_idx + 4];
      return ret;

    // prefix_evex
    case 0x62:
      if (!is_exec_mode_64b && (inst_bytes[op_idx + 1] & 0xc0) != 0xc0) {
        prefix_done = true;
        break;
      }
      ret.opcode_len = inst_bytes[op_idx + 1] & 0x03;
      ret.primary_opcode = inst_bytes[op_idx + 4];
      ret.modrm = inst_bytes[op_idx + 5];
      return ret;

    default:
      prefix_done = true;
      break;
    }
  } // prefix done

  ret.primary_opcode = inst_bytes[op_idx];
  ret.modrm = inst_bytes[op_idx + 1];
  ret.opcode_len = 1;

  // If the first opcode is 0F, it's two- or three- byte opcodes.
  if (ret.primary_opcode == 0x0F) {
    ret.primary_opcode = inst_bytes[++op_idx]; // get the next byte

    if (ret.primary_opcode == 0x38) {
      ret.opcode_len = 3;
      ret.primary_opcode = inst_bytes[++op_idx]; // get the next byte
      ret.modrm = inst_bytes[op_idx + 1];
    } else if (ret.primary_opcode == 0x3A) {
      ret.opcode_len = 3;
      ret.primary_opcode = inst_bytes[++op_idx];
      ret.modrm = inst_bytes[op_idx + 1];
    } else if ((ret.primary_opcode & 0xf8) == 0x38) {
      ret.opcode_len = 0;
      ret.primary_opcode = inst_bytes[++op_idx];
      ret.modrm = inst_bytes[op_idx + 1];
    } else if (ret.primary_opcode == 0x0F) {
      ret.opcode_len = 3;
      // opcode is 0x0F, no needs to update
      ret.modrm = inst_bytes[op_idx + 1];
    } else {
      ret.opcode_len = 2;
      ret.modrm = inst_bytes[op_idx + 1];
    }
  }

  return ret;
}

inline InstructionControlFlowKind GetControlFlowKind(bool is_exec_mode_64b,
                                                     const uint8_t *inst_bytes,
                                                     int bytes_len) {
  if (inst_bytes == nullptr || bytes_len <= 0)
    return eInstructionControlFlowKindUnknown;
  // Decode instruction bytes into opcode, modrm and opcode length.
  std::optional<InstructionOpcodeAndModrm> ret;
  // Opcode bytes will be decoded into primary_opcode, modrm and opcode length.
  // These are the three values deciding instruction control flow kind.
  ret = InstructionLengthDecode(inst_bytes, bytes_len, is_exec_mode_64b);
  if (!ret)
    return eInstructionControlFlowKindUnknown;
  else
    return MapOpcodeIntoControlFlowKind(*ret);
}

inline const char *GetNameForInstructionControlFlowKind(
    InstructionControlFlowKind instruction_control_flow_kind) {
  switch (instruction_control_flow_kind) {
  case eInstructionControlFlowKindUnknown:
    return "unknown";
  case eInstructionControlFlowKindOther:
    return "other";
  case eInstructionControlFlowKindCall:
    return "call";
  case eInstructionControlFlowKindReturn:
    return "return";
  case eInstructionControlFlowKindJump:
    return "jump";
  case eInstructionControlFlowKindCondJump:
    return "cond jump";
  case eInstructionControlFlowKindFarCall:
    return "far call";
  case eInstructionControlFlowKindFarReturn:
    return "far return";
  case eInstructionControlFlowKindFarJump:
    return "far jump";
  case eInstructionControlFlowKindHalt:
    return "halt";
  }
}
