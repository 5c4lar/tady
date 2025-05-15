#pragma once

#include <cstdint>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/TargetParser/Triple.h>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

void initialize_disassemblers();

struct InstructionInfo {
  std::optional<llvm::MCInst> instruction;
  uint64_t address;
  uint64_t size;
};

inline std::map<uint64_t, const InstructionInfo *>
insts_to_map(const std::vector<InstructionInfo> &insts) {
  std::map<uint64_t, const InstructionInfo *> result;
  for (const auto &inst : insts) {
    if (!inst.instruction) {
      continue;
    }
    result[inst.address] = &inst;
  }
  return result;
}

class DisassemblerBase {
public:
  DisassemblerBase(const std::string &triple_str, const std::string &cpu = "",
                   const std::string &feature = "");
  DisassemblerBase(const llvm::object::ObjectFile &obj);

  ~DisassemblerBase() = default;

  // Prevent copying
  DisassemblerBase(const DisassemblerBase &) = delete;
  DisassemblerBase &operator=(const DisassemblerBase &) = delete;
  // Get info objects
  const llvm::Triple &get_triple() const;
  const llvm::Target *get_target() const;
  const llvm::MCDisassembler *get_disassembler() const;
  const llvm::MCInstPrinter *get_instruction_printer() const;
  const llvm::MCInstrAnalysis *get_instruction_analysis() const;
  const llvm::MCRegisterInfo *get_register_info() const;
  const llvm::MCAsmInfo *get_assembler_info() const;
  const llvm::MCInstrInfo *get_instruction_info() const;
  const llvm::MCSubtargetInfo *get_subtarget_info() const;
  // DisassemblerBase interface
  std::optional<llvm::MCInst> disassemble(const llvm::ArrayRef<uint8_t> &data,
                                          uint64_t address,
                                          uint64_t &insn_size);
  virtual std::vector<InstructionInfo> disassemble(llvm::ArrayRef<uint8_t> data,
                                                   uint64_t address);

  // Get Instruction Info
  const llvm::MCInstrDesc &get_instruction_desc(const llvm::MCInst &inst) const;
  std::string print_inst(const llvm::MCInst &inst, uint64_t address,
                         uint64_t size) const;

  // Control flow analysis
  bool is_control_flow(const llvm::MCInst &inst) const {
    return instruction_analysis_->mayAffectControlFlow(inst, *register_info_);
  };
  uint64_t next_address(uint64_t address, uint64_t size) const {
    return address + size;
  }
  uint64_t must_transfer(const llvm::MCInst &inst, uint64_t address,
                         uint64_t size) const {
    if (is_control_flow(inst)) {
      if (instruction_analysis_->isUnconditionalBranch(inst)) {
        uint64_t target;
        instruction_analysis_->evaluateBranch(inst, address, size, target);
        return target;
      } else {
        return -1;
      }
    } else {
      return next_address(address, size);
    }
  }
  std::pair<uint64_t, uint64_t> may_transfer(const llvm::MCInst &inst,
                                             uint64_t address,
                                             uint64_t size) const {
    if (instruction_analysis_->isConditionalBranch(inst)) {
      uint64_t target;
      instruction_analysis_->evaluateBranch(inst, address, size, target);
      return {target, next_address(address, size)};
    } else {
      return {-1, -1};
    }
  }
  std::vector<uint64_t> successors(const llvm::MCInst &inst, uint64_t address,
                                   uint64_t size) const {
    std::vector<uint64_t> result;
    if (instruction_analysis_->isUnconditionalBranch(inst)) {
      result.push_back(must_transfer(inst, address, size));
    } else if (instruction_analysis_->isConditionalBranch(inst)) {
      auto [target, next] = may_transfer(inst, address, size);
      result.push_back(target);
      result.push_back(next);
    } else if (instruction_analysis_->isCall(inst)) {
      uint64_t target;
      if (instruction_analysis_->evaluateBranch(inst, address, size, target)) {
        result.push_back(target);
      }
    } else if (!is_control_flow(inst)) {
      result.push_back(next_address(address, size));
    }
    return result;
  }

protected:
  llvm::Triple triple_;
  const llvm::Target *target_;
  std::unique_ptr<llvm::MCRegisterInfo> register_info_;
  std::unique_ptr<llvm::MCAsmInfo> assembler_info_;
  std::unique_ptr<llvm::MCInstrInfo> instruction_info_;
  std::unique_ptr<llvm::MCSubtargetInfo> subtarget_info_;
  std::unique_ptr<llvm::MCContext> context_;
  std::unique_ptr<llvm::MCInstPrinter> instruction_printer_;
  std::unique_ptr<llvm::MCDisassembler> disassembler_;
  std::unique_ptr<llvm::MCInstrAnalysis> instruction_analysis_;
};

class LinearDisassembler : public DisassemblerBase {
public:
  LinearDisassembler(const std::string &triple_str, const std::string &cpu = "",
                     const std::string &feature = "")
      : DisassemblerBase(triple_str, cpu, feature) {}

  std::vector<InstructionInfo> disassemble(llvm::ArrayRef<uint8_t> data,
                                           uint64_t address) override;
};

class RecursiveDIsassembler : public DisassemblerBase {
  std::set<uint64_t> entries_;

public:
  RecursiveDIsassembler(const std::string &triple_str,
                        const std::string &cpu = "",
                        const std::string &feature = "")
      : DisassemblerBase(triple_str, cpu, feature) {}
  void add_entry(uint64_t address) { entries_.insert(address); }
  std::vector<InstructionInfo> disassemble(llvm::ArrayRef<uint8_t> data,
                                           uint64_t address) override;
};

class SupersetDisassembler : public DisassemblerBase {
  uint64_t step_;
  uint64_t workers_;

public:
  SupersetDisassembler(const std::string &triple_str,
                       const std::string &cpu = "",
                       const std::string &feature = "", uint64_t step = 1,
                       uint64_t workers = 1)
      : DisassemblerBase(triple_str, cpu, feature), step_(step),
        workers_(workers) {}
  std::vector<InstructionInfo> disassemble(llvm::ArrayRef<uint8_t> data,
                                           uint64_t address) override;
  std::vector<uint64_t> overlappings(uint64_t address, uint64_t size) const {
    std::vector<uint64_t> result;
    for (uint64_t i = 1; i < size; ++i) {
      result.push_back(address + i);
    }
    return result;
  }
};

static inline DisassemblerBase *
create_disassembler(const std::string &triple_str, const std::string &cpu = "",
                    const std::string &feature = "",
                    const std::string variant = "", uint64_t step = 1,
                    uint64_t workers = 1) {
  if (variant == "linear") {
    return new LinearDisassembler(triple_str, cpu, feature);
  } else if (variant == "recursive") {
    return new RecursiveDIsassembler(triple_str, cpu, feature);
  } else if (variant == "superset") {
    return new SupersetDisassembler(triple_str, cpu, feature, step, workers);
  } else {
    throw std::invalid_argument("Unknown disassembler variant: " + variant);
  }
}
