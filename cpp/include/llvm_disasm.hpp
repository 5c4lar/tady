#pragma once
#include "x86DisassemblerDecoder.hpp"
#include "x86FlowKind.hpp"
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace llvm;
using namespace llvm::X86Disassembler;

static std::once_flag disassembler_initialized;

static void initialize_disassemblers() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();
}

class Disassembler {
  std::unique_ptr<llvm::MCInstrInfo> MII;

public:
  Disassembler(std::string triple_str) {
    std::call_once(disassembler_initialized, initialize_disassemblers);
    std::string lookup_target_error;
    auto triple = llvm::Triple(llvm::Triple::normalize(triple_str));
    auto target = llvm::TargetRegistry::lookupTarget(triple.getTriple(),
                                                     lookup_target_error);
    if (!target) {
      throw std::runtime_error(lookup_target_error);
    }
    MII.reset(target->createMCInstrInfo());
  }

  std::optional<InternalInstruction>
  disasm(ArrayRef<uint8_t> Bytes, uint64_t Address, DisassemblerMode fMode) {
    InternalInstruction Insn;
    memset(&Insn, 0, sizeof(InternalInstruction));
    Insn.bytes = Bytes;
    Insn.startLocation = Address;
    Insn.readerCursor = Address;
    Insn.mode = fMode;

    if (Bytes.empty() || readPrefixes(&Insn) || readOpcode(&Insn) ||
        getInstructionID(&Insn, MII.get()) || Insn.instructionID == 0 ||
        readOperands(&Insn)) {
      return std::nullopt;
    }
    Insn.operands = x86OperandSets[Insn.spec->operands];
    Insn.length = Insn.readerCursor - Insn.startLocation;
    return Insn;
  }

  std::optional<uint64_t> get_branch_target(const InternalInstruction &Insn) {
    if (Insn.operands.size() == 0 ||
        MII->get(Insn.instructionID).operands()[0].OperandType !=
            MCOI::OPERAND_PCREL)
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

  std::tuple<py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<int32_t>, py::array_t<int32_t>>
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
        auto target_opt = get_branch_target(insn);
        int32_t branch_target = target_opt ? static_cast<int32_t>(*target_opt) : -1;
        branch_target = (branch_target >= num_bytes || branch_target < 0) ? -1 : branch_target;
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
        instrlen_ptr[current_row_idx] = 0; // Length 0
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
};
