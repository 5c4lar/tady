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

  std::tuple<py::array_t<uint8_t>, py::array_t<int32_t>>
  superset_disasm(py::array_t<uint8_t> Bytes, bool is64) {
    ArrayRef<uint8_t> Buffer =
        ArrayRef<uint8_t>(Bytes.data(), Bytes.data() + Bytes.size());

    size_t num_bytes = Buffer.size();
    py::ssize_t n_rows = static_cast<py::ssize_t>(num_bytes);

    // instrlen_flowkind: [instr_len, flow_kind], Shape: (n, 2)
    py::array_t<uint8_t> instrlen_flowkind({n_rows, 2L});

    // control_flow: next_addr, target_addr, Shape: (n, 2)
    py::array_t<int32_t> control_flow({n_rows, 2L});

    // Early exit if buffer is empty, loop won't run, empty arrays are returned.
    if (num_bytes == 0) {
      return std::make_tuple(instrlen_flowkind, control_flow);
    }

    auto instrlen_flowkind_ptr = instrlen_flowkind.mutable_data();
    auto control_flow_ptr = control_flow.mutable_data();
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
        instrlen_flowkind_ptr[current_row_idx * 2 + 0] =
            static_cast<uint8_t>(insn.length);
        instrlen_flowkind_ptr[current_row_idx * 2 + 1] = flow_kind_for_array;

        // Populate control_flow: [next_addr, target_addr]
        // insn.readerCursor is the offset of the next instruction from the
        // start of the buffer.
        control_flow_ptr[current_row_idx * 2 + 0] =
            static_cast<int32_t>(insn.readerCursor);

        auto target_opt = get_branch_target(insn);
        if (target_opt) {
          control_flow_ptr[current_row_idx * 2 + 1] =
              static_cast<int32_t>(*target_opt);
        } else {
          control_flow_ptr[current_row_idx * 2 + 1] =
              -1; // No target or not a branch
        }
      } else {
        // Failed to disassemble at offset i
        instrlen_flowkind_ptr[current_row_idx * 2 + 0] = 0; // Length 0
        instrlen_flowkind_ptr[current_row_idx * 2 + 1] =
            flow_kind_for_array; // Store flow kind from raw bytes

        // Set control_flow to {0, 0} to indicate failure/unknown
        control_flow_ptr[current_row_idx * 2 + 0] = -1;
        control_flow_ptr[current_row_idx * 2 + 1] = -1;
      }
    }
    return std::make_tuple(instrlen_flowkind, control_flow);
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
