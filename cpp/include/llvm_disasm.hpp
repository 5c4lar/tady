#pragma once
#include "mc_disasm.hpp"
#include "x86DisassemblerDecoder.hpp"
#include "x86FlowKind.hpp"
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace llvm;
using namespace llvm::X86Disassembler;

class Disassembler {
  std::unique_ptr<DisassemblerBase> mc_disasm_x86;
  std::unique_ptr<DisassemblerBase> mc_disasm_x64;

public:
  Disassembler() {
    mc_disasm_x86.reset(create_disassembler("i686", "", "", "superset"));
    mc_disasm_x64.reset(create_disassembler("x86_64", "", "", "superset"));
  }

  std::optional<InternalInstruction>
  disasm(ArrayRef<uint8_t> Bytes, uint64_t Address, DisassemblerMode fMode) {
    InternalInstruction Insn;
    memset(&Insn, 0, sizeof(InternalInstruction));
    Insn.bytes = Bytes;
    Insn.startLocation = Address;
    Insn.readerCursor = Address;
    Insn.mode = fMode;
    auto &mc_disasm =
        fMode == DisassemblerMode::MODE_32BIT ? mc_disasm_x86 : mc_disasm_x64;

    if (Bytes.empty() || readPrefixes(&Insn) || readOpcode(&Insn) ||
        getInstructionID(&Insn, mc_disasm->get_instruction_info()) ||
        Insn.instructionID == 0 || readOperands(&Insn)) {
      return std::nullopt;
    }
    Insn.operands = x86OperandSets[Insn.spec->operands];
    Insn.length = Insn.readerCursor - Insn.startLocation;
    return Insn;
  }

  std::optional<uint64_t> get_branch_target(const InternalInstruction &Insn,
                                            DisassemblerMode fMode) {
    auto &mc_disasm =
        fMode == DisassemblerMode::MODE_32BIT ? mc_disasm_x86 : mc_disasm_x64;
    if (Insn.operands.size() == 0 ||
        mc_disasm->get_instruction_info()
                ->get(Insn.instructionID)
                .operands()[0]
                .OperandType != MCOI::OPERAND_PCREL)
      return std::nullopt;
    auto operand = Insn.operands[0];
    uint64_t pcrel = Insn.startLocation + Insn.length;
    uint64_t immediate = Insn.immediates[0];
    switch (operand.encoding) {
    default:
      break;
    case ENCODING_Iv:
      switch (Insn.displacementSize) {
      default:
        break;
      case 1:
        if (immediate & 0x80)
          immediate |= ~(0xffull);
        break;
      case 2:
        if (immediate & 0x8000)
          immediate |= ~(0xffffull);
        break;
      case 4:
        if (immediate & 0x80000000)
          immediate |= ~(0xffffffffull);
        break;
      case 8:
        break;
      }
      break;
    case ENCODING_IB:
      if (immediate & 0x80)
        immediate |= ~(0xffull);
      break;
    case ENCODING_IW:
      if (immediate & 0x8000)
        immediate |= ~(0xffffull);
      break;
    case ENCODING_ID:
      if (immediate & 0x80000000)
        immediate |= ~(0xffffffffull);
      break;
    }
    return pcrel + immediate;
  }

  std::tuple<py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<int32_t>,
             py::array_t<int32_t>>
  superset_disasm(py::array_t<uint8_t> Bytes, bool is64) {
    ArrayRef<uint8_t> Buffer =
        ArrayRef<uint8_t>(Bytes.data(), Bytes.data() + Bytes.size());

    size_t num_bytes = Buffer.size();
    py::ssize_t n_rows = static_cast<py::ssize_t>(num_bytes);

    // instrlen: [instr_len], Shape: (n)
    py::array_t<uint8_t> instrlen(n_rows);

    // flow_kind: [flow_kind], Shape: (n)
    py::array_t<uint8_t> flow_kind(n_rows);

    // control_flow: must_transfer(1), may_transfer(2), next_addr(1), Shape: (n,
    // 4)
    py::array_t<int32_t> control_flow({n_rows, 4L});

    // successors: 2 possible successors, Shape: (n, 2)
    py::array_t<int32_t> successors({n_rows, 2L});

    // Early exit if buffer is empty, loop won't run, empty arrays are returned.
    if (num_bytes == 0) {
      return std::make_tuple(instrlen, flow_kind, control_flow, successors);
    }

    auto instrlen_ptr = instrlen.mutable_data();
    auto flow_kind_ptr = flow_kind.mutable_data();
    auto control_flow_ptr = control_flow.mutable_data();
    auto successors_ptr = successors.mutable_data();
    memset(instrlen_ptr, 0, instrlen.size() * sizeof(uint8_t));
    memset(flow_kind_ptr, 0, flow_kind.size() * sizeof(uint8_t));
    memset(control_flow_ptr, -1, control_flow.size() * sizeof(int32_t));
    memset(successors_ptr, -1, successors.size() * sizeof(int32_t));
    auto mode = is64 ? MODE_64BIT : MODE_32BIT;

    for (size_t i = 0; i < num_bytes; ++i) {
      py::ssize_t current_row_idx = static_cast<py::ssize_t>(i);

      auto insn_opt = disasm(Buffer.slice(i), static_cast<uint64_t>(i), mode);

      InstructionControlFlowKind flow_kind_enum =
          GetControlFlowKind(is64, Buffer.data() + i, num_bytes - i);
      uint8_t flow_kind_for_array = static_cast<uint8_t>(flow_kind_enum);

      if (insn_opt) {
        const auto &insn = *insn_opt;

        // Populate instrlen_flowkind: [length, flow_kind]
        instrlen_ptr[current_row_idx] = static_cast<uint8_t>(insn.length);
        flow_kind_ptr[current_row_idx] = flow_kind_for_array;
        auto target_opt = get_branch_target(insn, mode);
        int32_t branch_target =
            target_opt ? static_cast<int32_t>(*target_opt) : -1;
        if (!target_opt &&
            (flow_kind_enum == eInstructionControlFlowKindJump)) {
          flow_kind_ptr[current_row_idx] = 10;
        }
        branch_target = (branch_target >= num_bytes || branch_target < 0)
                            ? -1
                            : branch_target;
        // Populate control_flow: [must_transfer(1), may_transfer(2),
        // next_addr(1), target_addr(1)] insn.readerCursor is the offset of the
        // next instruction from the start of the buffer.
        int32_t next_addr = static_cast<int32_t>(insn.readerCursor);
        next_addr = next_addr >= num_bytes ? -1 : next_addr;
        control_flow_ptr[current_row_idx * 4 + 3] = next_addr;
        switch (flow_kind_enum) {
        case eInstructionControlFlowKindUnknown:
        case eInstructionControlFlowKindOther:
          control_flow_ptr[current_row_idx * 4 + 0] = next_addr;
          successors_ptr[current_row_idx * 2 + 0] = next_addr;
          break;
        case eInstructionControlFlowKindCall:
          control_flow_ptr[current_row_idx * 4 + 0] = next_addr;
          break;
        case eInstructionControlFlowKindJump:
          control_flow_ptr[current_row_idx * 4 + 0] = branch_target;
          successors_ptr[current_row_idx * 2 + 0] = branch_target;
          break;
        case eInstructionControlFlowKindCondJump:
          control_flow_ptr[current_row_idx * 4 + 1] = next_addr;
          control_flow_ptr[current_row_idx * 4 + 2] = branch_target;
          successors_ptr[current_row_idx * 2 + 0] = next_addr;
          successors_ptr[current_row_idx * 2 + 1] = branch_target;
          break;
        case eInstructionControlFlowKindReturn:
        case eInstructionControlFlowKindFarCall:
        case eInstructionControlFlowKindFarReturn:
        case eInstructionControlFlowKindFarJump:
        case eInstructionControlFlowKindHalt:
          break;
        }
      } else {
        // Failed to disassemble at offset i
        instrlen_ptr[current_row_idx] = 0;  // Length 0
        flow_kind_ptr[current_row_idx] = 0; // Store flow kind from raw bytes
      }
    }
    return std::make_tuple(instrlen, flow_kind, control_flow, successors);
  }

  py::array_t<uint8_t> flow_kind(py::array_t<uint8_t> Bytes, bool is64) {
    ArrayRef<uint8_t> Buffer =
        ArrayRef<uint8_t>(Bytes.data(), Bytes.data() + Bytes.shape(0));
    py::array_t<uint8_t> result(Buffer.size());
    auto result_ptr = result.mutable_data();
    for (size_t i = 0; i < Buffer.size(); i++) {
      InstructionControlFlowKind flow_kind =
          GetControlFlowKind(is64, Buffer.data() + i, Buffer.size() - i);
      result_ptr[i] = flow_kind;
    }
    return result;
  }

  // return an instruction string at the given address
  std::string inst_str(py::array_t<uint8_t> Bytes, bool is64, uint64_t address) {
    auto &mc_disasm = is64 ? mc_disasm_x64 : mc_disasm_x86;
    ArrayRef<uint8_t> Buffer =
        ArrayRef<uint8_t>(Bytes.data(), Bytes.data() + Bytes.size());
    uint64_t insn_size = 0;
    auto instr = mc_disasm->disassemble(Buffer, 0, insn_size);
    if (instr) {
      return mc_disasm->print_inst(*instr, address, insn_size);
    }
    return "invalid";
  }

  // return a list of strings of instructions to python
  py::list disasm_to_str(py::array_t<uint8_t> Bytes, bool is64,
                         uint64_t address) {
    auto &mc_disasm = is64 ? mc_disasm_x64 : mc_disasm_x86;
    ArrayRef<uint8_t> Buffer =
        ArrayRef<uint8_t>(Bytes.data(), Bytes.data() + Bytes.size());
    py::list result;
    for (int i = 0; i < Buffer.size(); i++) {
      uint64_t insn_size = 0;
      auto instr =
          mc_disasm->disassemble(Buffer.slice(i), address + i, insn_size);
      if (instr) {
        std::string instr_str = mc_disasm->print_inst(*instr, 0, insn_size);
        // remove the leading "\t"
        instr_str.erase(0, 1);
        result.append(instr_str);
      } else {
        result.append("invalid");
      }
    }
    return result;
  }

  py::array_t<int32_t> linear_disasm(py::array_t<uint8_t> Bytes, bool is64) {
    ArrayRef<uint8_t> Buffer =
        ArrayRef<uint8_t>(Bytes.data(), Bytes.data() + Bytes.size());
    std::vector<int32_t> result;
    auto &mc_disasm = is64 ? mc_disasm_x64 : mc_disasm_x86;
    int offset = 0;
    while (offset < Buffer.size()) {
      uint64_t insn_size = 0;
      auto instr = mc_disasm->disassemble(Buffer.slice(offset), offset, insn_size);
      if (instr) {
        result.push_back(offset);
        offset += insn_size;
        continue;
      } else {
        offset += mc_disasm->get_disassembler()->suggestBytesToSkip(Buffer.slice(offset), insn_size);
      }
    }
    py::array_t<int32_t> result_array(result.size());
    auto result_array_ptr = result_array.mutable_data();
    for (size_t i = 0; i < result.size(); i++) {
      result_array_ptr[i] = result[i];
    }
    return result_array;
  }
};
