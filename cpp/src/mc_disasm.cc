#include "mc_disasm.hpp"
#include <mutex>
#include <sstream>

#include <llvm/MC/MCContext.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <stdexcept>

static std::once_flag disassembler_initialized;

void initialize_disassemblers() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();
}

DisassemblerBase::DisassemblerBase(const std::string &triple_str,
                                   const std::string &cpu,
                                   const std::string &feature) {
  std::call_once(disassembler_initialized, initialize_disassemblers);
  triple_ = llvm::Triple(llvm::Triple::normalize(triple_str));

  std::string lookup_target_error;
  target_ = llvm::TargetRegistry::lookupTarget(triple_.getTriple(),
                                               lookup_target_error);
  if (target_ == nullptr) {
    throw std::runtime_error("Failed to lookup target: " + lookup_target_error);
  }

  // Init reusable llvm info objects
  register_info_.reset(target_->createMCRegInfo(triple_.getTriple()));
  if (!register_info_) {
    throw std::runtime_error("Failed to create register info");
  }

  llvm::MCTargetOptions target_options;
  assembler_info_.reset(target_->createMCAsmInfo(
      *register_info_, triple_.getTriple(), target_options));
  if (!assembler_info_) {
    throw std::runtime_error("Failed to create assembler info");
  }

  instruction_info_.reset(target_->createMCInstrInfo());
  if (!instruction_info_) {
    throw std::runtime_error("Failed to create instruction info");
  }

  subtarget_info_.reset(
      target_->createMCSubtargetInfo(triple_.getTriple(), cpu, feature));
  if (!subtarget_info_) {
    throw std::runtime_error("Failed to create subtarget info");
  }

  context_.reset(
      new llvm::MCContext(triple_, assembler_info_.get(), register_info_.get(),
                          subtarget_info_.get(), nullptr, &target_options));

  // Create instruction printer
  auto syntax_variant = assembler_info_->getAssemblerDialect();
  if (triple_.getArch() == llvm::Triple::x86 ||
      triple_.getArch() == llvm::Triple::x86_64) {
    syntax_variant = 1;
  }

  instruction_analysis_.reset(
      target_->createMCInstrAnalysis(instruction_info_.get()));
  if (!instruction_analysis_) {
    throw std::runtime_error("Failed to create instruction analysis");
  }

  instruction_printer_.reset(
      target_->createMCInstPrinter(triple_, syntax_variant, *assembler_info_,
                                   *instruction_info_, *register_info_));
  if (!instruction_printer_) {
    throw std::runtime_error("Failed to create instruction printer");
  }
  instruction_printer_->setPrintImmHex(true);
  instruction_printer_->setPrintBranchImmAsAddress(true);
  instruction_printer_->setMCInstrAnalysis(instruction_analysis_.get());

  // Create disassembler
  disassembler_.reset(
      target_->createMCDisassembler(*subtarget_info_, *context_));
  if (!disassembler_) {
    throw std::runtime_error("Failed to create disassembler");
  }
}

DisassemblerBase::DisassemblerBase(const llvm::object::ObjectFile &obj)
    : DisassemblerBase(obj.makeTriple().getTriple(),
                       obj.tryGetCPUName() ? obj.tryGetCPUName()->str() : "",
                       obj.getFeatures() ? obj.getFeatures()->getString()
                                         : "") {}

const llvm::Triple &DisassemblerBase::get_triple() const { return triple_; }

const llvm::Target *DisassemblerBase::get_target() const { return target_; }

const llvm::MCDisassembler *DisassemblerBase::get_disassembler() const {
  return disassembler_.get();
}

const llvm::MCInstPrinter *DisassemblerBase::get_instruction_printer() const {
  return instruction_printer_.get();
}

const llvm::MCInstrAnalysis *
DisassemblerBase::get_instruction_analysis() const {
  return instruction_analysis_.get();
}

const llvm::MCRegisterInfo *DisassemblerBase::get_register_info() const {
  return register_info_.get();
}

const llvm::MCAsmInfo *DisassemblerBase::get_assembler_info() const {
  return assembler_info_.get();
}

const llvm::MCInstrInfo *DisassemblerBase::get_instruction_info() const {
  return instruction_info_.get();
}

const llvm::MCSubtargetInfo *DisassemblerBase::get_subtarget_info() const {
  return subtarget_info_.get();
}

std::optional<llvm::MCInst>
DisassemblerBase::disassemble(const llvm::ArrayRef<uint8_t> &data,
                              uint64_t address, uint64_t &insn_size) {
  llvm::MCInst insn;
  auto res = disassembler_->getInstruction(insn, insn_size, data, address,
                                           llvm::nulls());
  if (res == llvm::MCDisassembler::Fail ||
      res == llvm::MCDisassembler::SoftFail) {
    std::stringstream error_stream;
    error_stream << "Could not disassemble at address " << std::hex << address;
    return std::nullopt;
  }
  return insn;
}

std::string DisassemblerBase::print_inst(const llvm::MCInst &inst,
                                         uint64_t address,
                                         uint64_t size) const {
  std::string insn_str;
  llvm::raw_string_ostream str_stream(insn_str);
  address += (subtarget_info_->getTargetTriple().isX86() ? size : 0);
  instruction_printer_->printInst(&inst, address, "", *subtarget_info_,
                                  str_stream);
  return str_stream.str();
}

const llvm::MCInstrDesc &
DisassemblerBase::get_instruction_desc(const llvm::MCInst &inst) const {
  return instruction_info_->get(inst.getOpcode());
}

std::vector<InstructionInfo>
DisassemblerBase::disassemble(llvm::ArrayRef<uint8_t> data, uint64_t address) {
  std::vector<InstructionInfo> insts;
  uint64_t insn_size = 0;
  uint64_t offset = 0;
  while (offset < data.size()) {
    auto inst = disassemble(data.slice(offset), address + offset, insn_size);
    if (!inst) {
      offset += disassembler_->suggestBytesToSkip(data.slice(offset),
                                                  address + offset);
      continue;
    }
    insts.push_back({inst.value(), address + offset, insn_size});
    offset += insn_size;
  }
  return insts;
}

std::vector<InstructionInfo>
LinearDisassembler::disassemble(llvm::ArrayRef<uint8_t> data,
                                uint64_t address) {
  std::vector<InstructionInfo> insts;
  uint64_t insn_size = 0;
  uint64_t offset = 0;
  while (offset < data.size()) {
    auto inst = DisassemblerBase::disassemble(data.slice(offset),
                                              address + offset, insn_size);
    if (!inst) {
      offset += disassembler_->suggestBytesToSkip(data.slice(offset),
                                                  address + offset);
      continue;
    }
    insts.push_back({inst.value(), address + offset, insn_size});
    offset += insn_size;
  }
  return insts;
}

std::vector<InstructionInfo>
RecursiveDIsassembler::disassemble(llvm::ArrayRef<uint8_t> data,
                                   uint64_t address) {
  std::vector<InstructionInfo> insts;
  std::set<uint64_t> visited;
  std::vector<uint64_t> stack;
  uint64_t end = address + data.size();
  uint64_t insn_size = 0;
  for (auto entry : entries_) {
    if (entry >= address && entry < end) {
      stack.push_back(entry);
    }
  }
  while (!stack.empty()) {
    auto entry = stack.back();
    stack.pop_back();
    if (visited.count(entry)) {
      continue;
    }
    visited.insert(entry);
    auto offset = entry - address;
    auto inst =
        DisassemblerBase::disassemble(data.slice(offset), entry, insn_size);
    if (!inst) {
      continue;
    }
    insts.push_back({inst.value(), address + offset, insn_size});
    auto successors =
        DisassemblerBase::successors(inst.value(), entry, insn_size);
    for (auto succ : successors) {
      if (succ >= address && succ < end && !visited.count(succ)) {
        stack.push_back(succ);
      }
    }
  }
  return insts;
}

std::vector<InstructionInfo>
SupersetDisassembler::disassemble(llvm::ArrayRef<uint8_t> data,
                                  uint64_t address) {
  std::vector<InstructionInfo> insts;
  for (size_t i = 0; i < data.size(); i += step_) {
    uint64_t insn_size = 0;
    auto inst =
        DisassemblerBase::disassemble(data.slice(i), address + i, insn_size);
    insts.push_back({inst, address + i, insn_size});
  }
  return insts;
}
