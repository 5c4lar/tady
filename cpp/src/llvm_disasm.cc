#include "x86DisassemblerDecoder.hpp"
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <mutex>

using namespace llvm;
using namespace llvm::X86Disassembler;

static std::once_flag disassembler_initialized;

void initialize_disassemblers() {
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

  InternalInstruction disasm(uint64_t &Size, ArrayRef<uint8_t> Bytes, uint64_t Address,
              DisassemblerMode fMode) {
    InternalInstruction Insn;
    memset(&Insn, 0, sizeof(InternalInstruction));
    Insn.bytes = Bytes;
    Insn.startLocation = Address;
    Insn.readerCursor = Address;
    Insn.mode = fMode;

    if (Bytes.empty() || readPrefixes(&Insn) || readOpcode(&Insn) ||
        getInstructionID(&Insn, MII.get()) || Insn.instructionID == 0 ||
        readOperands(&Insn)) {
      Size = Insn.readerCursor - Address;
    }
    return Insn;
  }
};


