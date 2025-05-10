//===-- X86DisassemblerDecoderInternal.h - Disassembler decoder -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the X86 Disassembler.
// It contains the public interface of the instruction decoder.
// Documentation for the disassembler can be found in X86Disassembler.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_DISASSEMBLER_X86DISASSEMBLERDECODER_H
#define LLVM_LIB_TARGET_X86_DISASSEMBLER_X86DISASSEMBLERDECODER_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/X86DisassemblerDecoderCommon.h>

namespace llvm {
namespace X86Disassembler {
#define DEBUG_TYPE "x86-disassembler"
#define debug(s) LLVM_DEBUG(dbgs() << __LINE__ << ": " << s);

// Helper macros
#define bitFromOffset0(val) ((val) & 0x1)
#define bitFromOffset1(val) (((val) >> 1) & 0x1)
#define bitFromOffset2(val) (((val) >> 2) & 0x1)
#define bitFromOffset3(val) (((val) >> 3) & 0x1)
#define bitFromOffset4(val) (((val) >> 4) & 0x1)
#define bitFromOffset5(val) (((val) >> 5) & 0x1)
#define bitFromOffset6(val) (((val) >> 6) & 0x1)
#define bitFromOffset7(val) (((val) >> 7) & 0x1)
#define twoBitsFromOffset0(val) ((val) & 0x3)
#define twoBitsFromOffset6(val) (((val) >> 6) & 0x3)
#define threeBitsFromOffset0(val) ((val) & 0x7)
#define threeBitsFromOffset3(val) (((val) >> 3) & 0x7)
#define fourBitsFromOffset0(val) ((val) & 0xf)
#define fourBitsFromOffset3(val) (((val) >> 3) & 0xf)
#define fiveBitsFromOffset0(val) ((val) & 0x1f)
#define invertedBitFromOffset2(val) (((~(val)) >> 2) & 0x1)
#define invertedBitFromOffset3(val) (((~(val)) >> 3) & 0x1)
#define invertedBitFromOffset4(val) (((~(val)) >> 4) & 0x1)
#define invertedBitFromOffset5(val) (((~(val)) >> 5) & 0x1)
#define invertedBitFromOffset6(val) (((~(val)) >> 6) & 0x1)
#define invertedBitFromOffset7(val) (((~(val)) >> 7) & 0x1)
#define invertedFourBitsFromOffset3(val) (((~(val)) >> 3) & 0xf)
// MOD/RM
#define modFromModRM(modRM) twoBitsFromOffset6(modRM)
#define regFromModRM(modRM) threeBitsFromOffset3(modRM)
#define rmFromModRM(modRM) threeBitsFromOffset0(modRM)
// SIB
#define scaleFromSIB(sib) twoBitsFromOffset6(sib)
#define indexFromSIB(sib) threeBitsFromOffset3(sib)
#define baseFromSIB(sib) threeBitsFromOffset0(sib)
// REX
#define wFromREX(rex) bitFromOffset3(rex)
#define rFromREX(rex) bitFromOffset2(rex)
#define xFromREX(rex) bitFromOffset1(rex)
#define bFromREX(rex) bitFromOffset0(rex)
// REX2
#define mFromREX2(rex2) bitFromOffset7(rex2)
#define r2FromREX2(rex2) bitFromOffset6(rex2)
#define x2FromREX2(rex2) bitFromOffset5(rex2)
#define b2FromREX2(rex2) bitFromOffset4(rex2)
#define wFromREX2(rex2) bitFromOffset3(rex2)
#define rFromREX2(rex2) bitFromOffset2(rex2)
#define xFromREX2(rex2) bitFromOffset1(rex2)
#define bFromREX2(rex2) bitFromOffset0(rex2)
// XOP
#define rFromXOP2of3(xop) invertedBitFromOffset7(xop)
#define xFromXOP2of3(xop) invertedBitFromOffset6(xop)
#define bFromXOP2of3(xop) invertedBitFromOffset5(xop)
#define mmmmmFromXOP2of3(xop) fiveBitsFromOffset0(xop)
#define wFromXOP3of3(xop) bitFromOffset7(xop)
#define vvvvFromXOP3of3(xop) invertedFourBitsFromOffset3(xop)
#define lFromXOP3of3(xop) bitFromOffset2(xop)
#define ppFromXOP3of3(xop) twoBitsFromOffset0(xop)
// VEX2
#define rFromVEX2of2(vex) invertedBitFromOffset7(vex)
#define vvvvFromVEX2of2(vex) invertedFourBitsFromOffset3(vex)
#define lFromVEX2of2(vex) bitFromOffset2(vex)
#define ppFromVEX2of2(vex) twoBitsFromOffset0(vex)
// VEX3
#define rFromVEX2of3(vex) invertedBitFromOffset7(vex)
#define xFromVEX2of3(vex) invertedBitFromOffset6(vex)
#define bFromVEX2of3(vex) invertedBitFromOffset5(vex)
#define mmmmmFromVEX2of3(vex) fiveBitsFromOffset0(vex)
#define wFromVEX3of3(vex) bitFromOffset7(vex)
#define vvvvFromVEX3of3(vex) invertedFourBitsFromOffset3(vex)
#define lFromVEX3of3(vex) bitFromOffset2(vex)
#define ppFromVEX3of3(vex) twoBitsFromOffset0(vex)
// EVEX
#define rFromEVEX2of4(evex) invertedBitFromOffset7(evex)
#define xFromEVEX2of4(evex) invertedBitFromOffset6(evex)
#define bFromEVEX2of4(evex) invertedBitFromOffset5(evex)
#define r2FromEVEX2of4(evex) invertedBitFromOffset4(evex)
#define b2FromEVEX2of4(evex) bitFromOffset3(evex)
#define mmmFromEVEX2of4(evex) threeBitsFromOffset0(evex)
#define wFromEVEX3of4(evex) bitFromOffset7(evex)
#define vvvvFromEVEX3of4(evex) invertedFourBitsFromOffset3(evex)
#define uFromEVEX3of4(evex) invertedBitFromOffset2(evex)
#define ppFromEVEX3of4(evex) twoBitsFromOffset0(evex)
#define oszcFromEVEX3of4(evex) fourBitsFromOffset3(evex)
#define zFromEVEX4of4(evex) bitFromOffset7(evex)
#define l2FromEVEX4of4(evex) bitFromOffset6(evex)
#define lFromEVEX4of4(evex) bitFromOffset5(evex)
#define bFromEVEX4of4(evex) bitFromOffset4(evex)
#define v2FromEVEX4of4(evex) invertedBitFromOffset3(evex)
#define aaaFromEVEX4of4(evex) threeBitsFromOffset0(evex)
#define nfFromEVEX4of4(evex) bitFromOffset2(evex)
#define scFromEVEX4of4(evex) fourBitsFromOffset0(evex)

// These enums represent Intel registers for use by the decoder.
#define REGS_8BIT                                                              \
  ENTRY(AL)                                                                    \
  ENTRY(CL)                                                                    \
  ENTRY(DL)                                                                    \
  ENTRY(BL)                                                                    \
  ENTRY(AH)                                                                    \
  ENTRY(CH)                                                                    \
  ENTRY(DH)                                                                    \
  ENTRY(BH)                                                                    \
  ENTRY(R8B)                                                                   \
  ENTRY(R9B)                                                                   \
  ENTRY(R10B)                                                                  \
  ENTRY(R11B)                                                                  \
  ENTRY(R12B)                                                                  \
  ENTRY(R13B)                                                                  \
  ENTRY(R14B)                                                                  \
  ENTRY(R15B)                                                                  \
  ENTRY(R16B)                                                                  \
  ENTRY(R17B)                                                                  \
  ENTRY(R18B)                                                                  \
  ENTRY(R19B)                                                                  \
  ENTRY(R20B)                                                                  \
  ENTRY(R21B)                                                                  \
  ENTRY(R22B)                                                                  \
  ENTRY(R23B)                                                                  \
  ENTRY(R24B)                                                                  \
  ENTRY(R25B)                                                                  \
  ENTRY(R26B)                                                                  \
  ENTRY(R27B)                                                                  \
  ENTRY(R28B)                                                                  \
  ENTRY(R29B)                                                                  \
  ENTRY(R30B)                                                                  \
  ENTRY(R31B)                                                                  \
  ENTRY(SPL)                                                                   \
  ENTRY(BPL)                                                                   \
  ENTRY(SIL)                                                                   \
  ENTRY(DIL)

#define EA_BASES_16BIT                                                         \
  ENTRY(BX_SI)                                                                 \
  ENTRY(BX_DI)                                                                 \
  ENTRY(BP_SI)                                                                 \
  ENTRY(BP_DI)                                                                 \
  ENTRY(SI)                                                                    \
  ENTRY(DI)                                                                    \
  ENTRY(BP)                                                                    \
  ENTRY(BX)                                                                    \
  ENTRY(R8W)                                                                   \
  ENTRY(R9W)                                                                   \
  ENTRY(R10W)                                                                  \
  ENTRY(R11W)                                                                  \
  ENTRY(R12W)                                                                  \
  ENTRY(R13W)                                                                  \
  ENTRY(R14W)                                                                  \
  ENTRY(R15W)                                                                  \
  ENTRY(R16W)                                                                  \
  ENTRY(R17W)                                                                  \
  ENTRY(R18W)                                                                  \
  ENTRY(R19W)                                                                  \
  ENTRY(R20W)                                                                  \
  ENTRY(R21W)                                                                  \
  ENTRY(R22W)                                                                  \
  ENTRY(R23W)                                                                  \
  ENTRY(R24W)                                                                  \
  ENTRY(R25W)                                                                  \
  ENTRY(R26W)                                                                  \
  ENTRY(R27W)                                                                  \
  ENTRY(R28W)                                                                  \
  ENTRY(R29W)                                                                  \
  ENTRY(R30W)                                                                  \
  ENTRY(R31W)

#define REGS_16BIT                                                             \
  ENTRY(AX)                                                                    \
  ENTRY(CX)                                                                    \
  ENTRY(DX)                                                                    \
  ENTRY(BX)                                                                    \
  ENTRY(SP)                                                                    \
  ENTRY(BP)                                                                    \
  ENTRY(SI)                                                                    \
  ENTRY(DI)                                                                    \
  ENTRY(R8W)                                                                   \
  ENTRY(R9W)                                                                   \
  ENTRY(R10W)                                                                  \
  ENTRY(R11W)                                                                  \
  ENTRY(R12W)                                                                  \
  ENTRY(R13W)                                                                  \
  ENTRY(R14W)                                                                  \
  ENTRY(R15W)                                                                  \
  ENTRY(R16W)                                                                  \
  ENTRY(R17W)                                                                  \
  ENTRY(R18W)                                                                  \
  ENTRY(R19W)                                                                  \
  ENTRY(R20W)                                                                  \
  ENTRY(R21W)                                                                  \
  ENTRY(R22W)                                                                  \
  ENTRY(R23W)                                                                  \
  ENTRY(R24W)                                                                  \
  ENTRY(R25W)                                                                  \
  ENTRY(R26W)                                                                  \
  ENTRY(R27W)                                                                  \
  ENTRY(R28W)                                                                  \
  ENTRY(R29W)                                                                  \
  ENTRY(R30W)                                                                  \
  ENTRY(R31W)

#define EA_BASES_32BIT                                                         \
  ENTRY(EAX)                                                                   \
  ENTRY(ECX)                                                                   \
  ENTRY(EDX)                                                                   \
  ENTRY(EBX)                                                                   \
  ENTRY(sib)                                                                   \
  ENTRY(EBP)                                                                   \
  ENTRY(ESI)                                                                   \
  ENTRY(EDI)                                                                   \
  ENTRY(R8D)                                                                   \
  ENTRY(R9D)                                                                   \
  ENTRY(R10D)                                                                  \
  ENTRY(R11D)                                                                  \
  ENTRY(R12D)                                                                  \
  ENTRY(R13D)                                                                  \
  ENTRY(R14D)                                                                  \
  ENTRY(R15D)                                                                  \
  ENTRY(R16D)                                                                  \
  ENTRY(R17D)                                                                  \
  ENTRY(R18D)                                                                  \
  ENTRY(R19D)                                                                  \
  ENTRY(R20D)                                                                  \
  ENTRY(R21D)                                                                  \
  ENTRY(R22D)                                                                  \
  ENTRY(R23D)                                                                  \
  ENTRY(R24D)                                                                  \
  ENTRY(R25D)                                                                  \
  ENTRY(R26D)                                                                  \
  ENTRY(R27D)                                                                  \
  ENTRY(R28D)                                                                  \
  ENTRY(R29D)                                                                  \
  ENTRY(R30D)                                                                  \
  ENTRY(R31D)

#define REGS_32BIT                                                             \
  ENTRY(EAX)                                                                   \
  ENTRY(ECX)                                                                   \
  ENTRY(EDX)                                                                   \
  ENTRY(EBX)                                                                   \
  ENTRY(ESP)                                                                   \
  ENTRY(EBP)                                                                   \
  ENTRY(ESI)                                                                   \
  ENTRY(EDI)                                                                   \
  ENTRY(R8D)                                                                   \
  ENTRY(R9D)                                                                   \
  ENTRY(R10D)                                                                  \
  ENTRY(R11D)                                                                  \
  ENTRY(R12D)                                                                  \
  ENTRY(R13D)                                                                  \
  ENTRY(R14D)                                                                  \
  ENTRY(R15D)                                                                  \
  ENTRY(R16D)                                                                  \
  ENTRY(R17D)                                                                  \
  ENTRY(R18D)                                                                  \
  ENTRY(R19D)                                                                  \
  ENTRY(R20D)                                                                  \
  ENTRY(R21D)                                                                  \
  ENTRY(R22D)                                                                  \
  ENTRY(R23D)                                                                  \
  ENTRY(R24D)                                                                  \
  ENTRY(R25D)                                                                  \
  ENTRY(R26D)                                                                  \
  ENTRY(R27D)                                                                  \
  ENTRY(R28D)                                                                  \
  ENTRY(R29D)                                                                  \
  ENTRY(R30D)                                                                  \
  ENTRY(R31D)

#define EA_BASES_64BIT                                                         \
  ENTRY(RAX)                                                                   \
  ENTRY(RCX)                                                                   \
  ENTRY(RDX)                                                                   \
  ENTRY(RBX)                                                                   \
  ENTRY(sib64)                                                                 \
  ENTRY(RBP)                                                                   \
  ENTRY(RSI)                                                                   \
  ENTRY(RDI)                                                                   \
  ENTRY(R8)                                                                    \
  ENTRY(R9)                                                                    \
  ENTRY(R10)                                                                   \
  ENTRY(R11)                                                                   \
  ENTRY(R12)                                                                   \
  ENTRY(R13)                                                                   \
  ENTRY(R14)                                                                   \
  ENTRY(R15)                                                                   \
  ENTRY(R16)                                                                   \
  ENTRY(R17)                                                                   \
  ENTRY(R18)                                                                   \
  ENTRY(R19)                                                                   \
  ENTRY(R20)                                                                   \
  ENTRY(R21)                                                                   \
  ENTRY(R22)                                                                   \
  ENTRY(R23)                                                                   \
  ENTRY(R24)                                                                   \
  ENTRY(R25)                                                                   \
  ENTRY(R26)                                                                   \
  ENTRY(R27)                                                                   \
  ENTRY(R28)                                                                   \
  ENTRY(R29)                                                                   \
  ENTRY(R30)                                                                   \
  ENTRY(R31)

#define REGS_64BIT                                                             \
  ENTRY(RAX)                                                                   \
  ENTRY(RCX)                                                                   \
  ENTRY(RDX)                                                                   \
  ENTRY(RBX)                                                                   \
  ENTRY(RSP)                                                                   \
  ENTRY(RBP)                                                                   \
  ENTRY(RSI)                                                                   \
  ENTRY(RDI)                                                                   \
  ENTRY(R8)                                                                    \
  ENTRY(R9)                                                                    \
  ENTRY(R10)                                                                   \
  ENTRY(R11)                                                                   \
  ENTRY(R12)                                                                   \
  ENTRY(R13)                                                                   \
  ENTRY(R14)                                                                   \
  ENTRY(R15)                                                                   \
  ENTRY(R16)                                                                   \
  ENTRY(R17)                                                                   \
  ENTRY(R18)                                                                   \
  ENTRY(R19)                                                                   \
  ENTRY(R20)                                                                   \
  ENTRY(R21)                                                                   \
  ENTRY(R22)                                                                   \
  ENTRY(R23)                                                                   \
  ENTRY(R24)                                                                   \
  ENTRY(R25)                                                                   \
  ENTRY(R26)                                                                   \
  ENTRY(R27)                                                                   \
  ENTRY(R28)                                                                   \
  ENTRY(R29)                                                                   \
  ENTRY(R30)                                                                   \
  ENTRY(R31)

#define REGS_MMX                                                               \
  ENTRY(MM0)                                                                   \
  ENTRY(MM1)                                                                   \
  ENTRY(MM2)                                                                   \
  ENTRY(MM3)                                                                   \
  ENTRY(MM4)                                                                   \
  ENTRY(MM5)                                                                   \
  ENTRY(MM6)                                                                   \
  ENTRY(MM7)

#define REGS_XMM                                                               \
  ENTRY(XMM0)                                                                  \
  ENTRY(XMM1)                                                                  \
  ENTRY(XMM2)                                                                  \
  ENTRY(XMM3)                                                                  \
  ENTRY(XMM4)                                                                  \
  ENTRY(XMM5)                                                                  \
  ENTRY(XMM6)                                                                  \
  ENTRY(XMM7)                                                                  \
  ENTRY(XMM8)                                                                  \
  ENTRY(XMM9)                                                                  \
  ENTRY(XMM10)                                                                 \
  ENTRY(XMM11)                                                                 \
  ENTRY(XMM12)                                                                 \
  ENTRY(XMM13)                                                                 \
  ENTRY(XMM14)                                                                 \
  ENTRY(XMM15)                                                                 \
  ENTRY(XMM16)                                                                 \
  ENTRY(XMM17)                                                                 \
  ENTRY(XMM18)                                                                 \
  ENTRY(XMM19)                                                                 \
  ENTRY(XMM20)                                                                 \
  ENTRY(XMM21)                                                                 \
  ENTRY(XMM22)                                                                 \
  ENTRY(XMM23)                                                                 \
  ENTRY(XMM24)                                                                 \
  ENTRY(XMM25)                                                                 \
  ENTRY(XMM26)                                                                 \
  ENTRY(XMM27)                                                                 \
  ENTRY(XMM28)                                                                 \
  ENTRY(XMM29)                                                                 \
  ENTRY(XMM30)                                                                 \
  ENTRY(XMM31)

#define REGS_YMM                                                               \
  ENTRY(YMM0)                                                                  \
  ENTRY(YMM1)                                                                  \
  ENTRY(YMM2)                                                                  \
  ENTRY(YMM3)                                                                  \
  ENTRY(YMM4)                                                                  \
  ENTRY(YMM5)                                                                  \
  ENTRY(YMM6)                                                                  \
  ENTRY(YMM7)                                                                  \
  ENTRY(YMM8)                                                                  \
  ENTRY(YMM9)                                                                  \
  ENTRY(YMM10)                                                                 \
  ENTRY(YMM11)                                                                 \
  ENTRY(YMM12)                                                                 \
  ENTRY(YMM13)                                                                 \
  ENTRY(YMM14)                                                                 \
  ENTRY(YMM15)                                                                 \
  ENTRY(YMM16)                                                                 \
  ENTRY(YMM17)                                                                 \
  ENTRY(YMM18)                                                                 \
  ENTRY(YMM19)                                                                 \
  ENTRY(YMM20)                                                                 \
  ENTRY(YMM21)                                                                 \
  ENTRY(YMM22)                                                                 \
  ENTRY(YMM23)                                                                 \
  ENTRY(YMM24)                                                                 \
  ENTRY(YMM25)                                                                 \
  ENTRY(YMM26)                                                                 \
  ENTRY(YMM27)                                                                 \
  ENTRY(YMM28)                                                                 \
  ENTRY(YMM29)                                                                 \
  ENTRY(YMM30)                                                                 \
  ENTRY(YMM31)

#define REGS_ZMM                                                               \
  ENTRY(ZMM0)                                                                  \
  ENTRY(ZMM1)                                                                  \
  ENTRY(ZMM2)                                                                  \
  ENTRY(ZMM3)                                                                  \
  ENTRY(ZMM4)                                                                  \
  ENTRY(ZMM5)                                                                  \
  ENTRY(ZMM6)                                                                  \
  ENTRY(ZMM7)                                                                  \
  ENTRY(ZMM8)                                                                  \
  ENTRY(ZMM9)                                                                  \
  ENTRY(ZMM10)                                                                 \
  ENTRY(ZMM11)                                                                 \
  ENTRY(ZMM12)                                                                 \
  ENTRY(ZMM13)                                                                 \
  ENTRY(ZMM14)                                                                 \
  ENTRY(ZMM15)                                                                 \
  ENTRY(ZMM16)                                                                 \
  ENTRY(ZMM17)                                                                 \
  ENTRY(ZMM18)                                                                 \
  ENTRY(ZMM19)                                                                 \
  ENTRY(ZMM20)                                                                 \
  ENTRY(ZMM21)                                                                 \
  ENTRY(ZMM22)                                                                 \
  ENTRY(ZMM23)                                                                 \
  ENTRY(ZMM24)                                                                 \
  ENTRY(ZMM25)                                                                 \
  ENTRY(ZMM26)                                                                 \
  ENTRY(ZMM27)                                                                 \
  ENTRY(ZMM28)                                                                 \
  ENTRY(ZMM29)                                                                 \
  ENTRY(ZMM30)                                                                 \
  ENTRY(ZMM31)

#define REGS_MASKS                                                             \
  ENTRY(K0)                                                                    \
  ENTRY(K1)                                                                    \
  ENTRY(K2)                                                                    \
  ENTRY(K3)                                                                    \
  ENTRY(K4)                                                                    \
  ENTRY(K5)                                                                    \
  ENTRY(K6)                                                                    \
  ENTRY(K7)

#define REGS_MASK_PAIRS                                                        \
  ENTRY(K0_K1)                                                                 \
  ENTRY(K2_K3)                                                                 \
  ENTRY(K4_K5)                                                                 \
  ENTRY(K6_K7)

#define REGS_SEGMENT                                                           \
  ENTRY(ES)                                                                    \
  ENTRY(CS)                                                                    \
  ENTRY(SS)                                                                    \
  ENTRY(DS)                                                                    \
  ENTRY(FS)                                                                    \
  ENTRY(GS)

#define REGS_DEBUG                                                             \
  ENTRY(DR0)                                                                   \
  ENTRY(DR1)                                                                   \
  ENTRY(DR2)                                                                   \
  ENTRY(DR3)                                                                   \
  ENTRY(DR4)                                                                   \
  ENTRY(DR5)                                                                   \
  ENTRY(DR6)                                                                   \
  ENTRY(DR7)                                                                   \
  ENTRY(DR8)                                                                   \
  ENTRY(DR9)                                                                   \
  ENTRY(DR10)                                                                  \
  ENTRY(DR11)                                                                  \
  ENTRY(DR12)                                                                  \
  ENTRY(DR13)                                                                  \
  ENTRY(DR14)                                                                  \
  ENTRY(DR15)

#define REGS_CONTROL                                                           \
  ENTRY(CR0)                                                                   \
  ENTRY(CR1)                                                                   \
  ENTRY(CR2)                                                                   \
  ENTRY(CR3)                                                                   \
  ENTRY(CR4)                                                                   \
  ENTRY(CR5)                                                                   \
  ENTRY(CR6)                                                                   \
  ENTRY(CR7)                                                                   \
  ENTRY(CR8)                                                                   \
  ENTRY(CR9)                                                                   \
  ENTRY(CR10)                                                                  \
  ENTRY(CR11)                                                                  \
  ENTRY(CR12)                                                                  \
  ENTRY(CR13)                                                                  \
  ENTRY(CR14)                                                                  \
  ENTRY(CR15)

#undef REGS_TMM
#define REGS_TMM                                                               \
  ENTRY(TMM0)                                                                  \
  ENTRY(TMM1)                                                                  \
  ENTRY(TMM2)                                                                  \
  ENTRY(TMM3)                                                                  \
  ENTRY(TMM4)                                                                  \
  ENTRY(TMM5)                                                                  \
  ENTRY(TMM6)                                                                  \
  ENTRY(TMM7)

#define REGS_TMM_PAIRS                                                         \
  ENTRY(TMM0_TMM1)                                                             \
  ENTRY(TMM2_TMM3)                                                             \
  ENTRY(TMM4_TMM5)                                                             \
  ENTRY(TMM6_TMM7)

#define ALL_EA_BASES                                                           \
  EA_BASES_16BIT                                                               \
  EA_BASES_32BIT                                                               \
  EA_BASES_64BIT

#define ALL_SIB_BASES                                                          \
  REGS_32BIT                                                                   \
  REGS_64BIT

#define ALL_REGS                                                               \
  REGS_8BIT                                                                    \
  REGS_16BIT                                                                   \
  REGS_32BIT                                                                   \
  REGS_64BIT                                                                   \
  REGS_MMX                                                                     \
  REGS_XMM                                                                     \
  REGS_YMM                                                                     \
  REGS_ZMM                                                                     \
  REGS_MASKS                                                                   \
  REGS_MASK_PAIRS                                                              \
  REGS_SEGMENT                                                                 \
  REGS_DEBUG                                                                   \
  REGS_CONTROL                                                                 \
  REGS_TMM                                                                     \
  REGS_TMM_PAIRS                                                               \
  ENTRY(RIP)

/// All possible values of the base field for effective-address
/// computations, a.k.a. the Mod and R/M fields of the ModR/M byte.
/// We distinguish between bases (EA_BASE_*) and registers that just happen
/// to be referred to when Mod == 0b11 (EA_REG_*).
enum EABase {
  // clang-format off
  EA_BASE_NONE,
#define ENTRY(x) EA_BASE_##x,
  ALL_EA_BASES
#undef ENTRY
#define ENTRY(x) EA_REG_##x,
  ALL_REGS
#undef ENTRY
  EA_max
  // clang-format on
};

/// All possible values of the SIB index field.
/// borrows entries from ALL_EA_BASES with the special case that
/// sib is synonymous with NONE.
/// Vector SIB: index can be XMM or YMM.
enum SIBIndex {
  // clang-format off
  SIB_INDEX_NONE,
#define ENTRY(x) SIB_INDEX_##x,
  ALL_EA_BASES
  REGS_XMM
  REGS_YMM
  REGS_ZMM
#undef ENTRY
  SIB_INDEX_max
  // clang-format on
};

/// All possible values of the SIB base field.
enum SIBBase {
  // clang-format off
  SIB_BASE_NONE,
#define ENTRY(x) SIB_BASE_##x,
  ALL_SIB_BASES
#undef ENTRY
  SIB_BASE_max
  // clang-format on
};

/// Possible displacement types for effective-address computations.
enum EADisplacement { EA_DISP_NONE, EA_DISP_8, EA_DISP_16, EA_DISP_32 };

/// All possible values of the reg field in the ModR/M byte.
// clang-format off
enum Reg {
#define ENTRY(x) MODRM_REG_##x,
  ALL_REGS
#undef ENTRY
  MODRM_REG_max
};
// clang-format on

/// All possible segment overrides.
enum SegmentOverride {
  SEG_OVERRIDE_NONE,
  SEG_OVERRIDE_CS,
  SEG_OVERRIDE_SS,
  SEG_OVERRIDE_DS,
  SEG_OVERRIDE_ES,
  SEG_OVERRIDE_FS,
  SEG_OVERRIDE_GS,
  SEG_OVERRIDE_max
};

/// Possible values for the VEX.m-mmmm field
enum VEXLeadingOpcodeByte {
  VEX_LOB_0F = 0x1,
  VEX_LOB_0F38 = 0x2,
  VEX_LOB_0F3A = 0x3,
  VEX_LOB_MAP4 = 0x4,
  VEX_LOB_MAP5 = 0x5,
  VEX_LOB_MAP6 = 0x6,
  VEX_LOB_MAP7 = 0x7
};

enum XOPMapSelect {
  XOP_MAP_SELECT_8 = 0x8,
  XOP_MAP_SELECT_9 = 0x9,
  XOP_MAP_SELECT_A = 0xA
};

/// Possible values for the VEX.pp/EVEX.pp field
enum VEXPrefixCode {
  VEX_PREFIX_NONE = 0x0,
  VEX_PREFIX_66 = 0x1,
  VEX_PREFIX_F3 = 0x2,
  VEX_PREFIX_F2 = 0x3
};

enum VectorExtensionType {
  TYPE_NO_VEX_XOP = 0x0,
  TYPE_VEX_2B = 0x1,
  TYPE_VEX_3B = 0x2,
  TYPE_EVEX = 0x3,
  TYPE_XOP = 0x4
};

/// The specification for how to extract and interpret a full instruction and
/// its operands.
struct InstructionSpecifier {
  uint16_t operands;
};

/// The x86 internal instruction, which is produced by the decoder.
struct InternalInstruction {
  // Opaque value passed to the reader
  llvm::ArrayRef<uint8_t> bytes;
  // The address of the next byte to read via the reader
  uint64_t readerCursor;

  // General instruction information

  // The mode to disassemble for (64-bit, protected, real)
  DisassemblerMode mode;
  // The start of the instruction, usable with the reader
  uint64_t startLocation;
  // The length of the instruction, in bytes
  size_t length;

  // Prefix state

  // The possible mandatory prefix
  uint8_t mandatoryPrefix;
  // The value of the vector extension prefix(EVEX/VEX/XOP), if present
  uint8_t vectorExtensionPrefix[4];
  // The type of the vector extension prefix
  VectorExtensionType vectorExtensionType;
  // The value of the REX2 prefix, if present
  uint8_t rex2ExtensionPrefix[2];
  // The value of the REX prefix, if present
  uint8_t rexPrefix;
  // The segment override type
  SegmentOverride segmentOverride;
  // 1 if the prefix byte, 0xf2 or 0xf3 is xacquire or xrelease
  bool xAcquireRelease;

  // Address-size override
  bool hasAdSize;
  // Operand-size override
  bool hasOpSize;
  // Lock prefix
  bool hasLockPrefix;
  // The repeat prefix if any
  uint8_t repeatPrefix;

  // Sizes of various critical pieces of data, in bytes
  uint8_t registerSize;
  uint8_t addressSize;
  uint8_t displacementSize;
  uint8_t immediateSize;

  // Offsets from the start of the instruction to the pieces of data, which is
  // needed to find relocation entries for adding symbolic operands.
  uint8_t displacementOffset;
  uint8_t immediateOffset;

  // opcode state

  // The last byte of the opcode, not counting any ModR/M extension
  uint8_t opcode;

  // decode state

  // The type of opcode, used for indexing into the array of decode tables
  OpcodeType opcodeType;
  // The instruction ID, extracted from the decode table
  uint16_t instructionID;
  // The specifier for the instruction, from the instruction info table
  const InstructionSpecifier *spec;

  // state for additional bytes, consumed during operand decode.  Pattern:
  // consumed___ indicates that the byte was already consumed and does not
  // need to be consumed again.

  // The VEX.vvvv field, which contains a third register operand for some AVX
  // instructions.
  Reg vvvv;

  // The writemask for AVX-512 instructions which is contained in EVEX.aaa
  Reg writemask;

  // The ModR/M byte, which contains most register operands and some portion of
  // all memory operands.
  bool consumedModRM;
  uint8_t modRM;

  // The SIB byte, used for more complex 32- or 64-bit memory operands
  uint8_t sib;

  // The displacement, used for memory operands
  int32_t displacement;

  // Immediates.  There can be three in some cases
  uint8_t numImmediatesConsumed;
  uint8_t numImmediatesTranslated;
  uint64_t immediates[3];

  // A register or immediate operand encoded into the opcode
  Reg opcodeRegister;

  // Portions of the ModR/M byte

  // These fields determine the allowable values for the ModR/M fields, which
  // depend on operand and address widths.
  EABase eaRegBase;
  Reg regBase;

  // The Mod and R/M fields can encode a base for an effective address, or a
  // register.  These are separated into two fields here.
  EABase eaBase;
  EADisplacement eaDisplacement;
  // The reg field always encodes a register
  Reg reg;

  // SIB state
  SIBIndex sibIndexBase;
  SIBIndex sibIndex;
  uint8_t sibScale;
  SIBBase sibBase;

  // Embedded rounding control.
  uint8_t RC;

  ArrayRef<OperandSpecifier> operands;
};

// Specifies whether a ModR/M byte is needed and (if so) which
// instruction each possible value of the ModR/M byte corresponds to.  Once
// this information is known, we have narrowed down to a single instruction.
struct ModRMDecision {
  uint8_t modrm_type;
  uint16_t instructionIDs;
};

// Specifies which set of ModR/M->instruction tables to look at
// given a particular opcode.
struct OpcodeDecision {
  ModRMDecision modRMDecisions[256];
};

// Specifies which opcode->instruction tables to look at given
// a particular context (set of attributes).  Since there are many possible
// contexts, the decoder first uses CONTEXTS_SYM to determine which context
// applies given a specific set of attributes.  Hence there are only IC_max
// entries in this table, rather than 2^(ATTR_max).
struct ContextDecision {
  OpcodeDecision opcodeDecisions[IC_max];
};

#include "X86GenDisassemblerTables.inc"

static InstrUID decode(OpcodeType type, InstructionContext insnContext,
                       uint8_t opcode, uint8_t modRM) {
  const struct ModRMDecision *dec;

  switch (type) {
  case ONEBYTE:
    dec = &ONEBYTE_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case TWOBYTE:
    dec = &TWOBYTE_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEBYTE_38:
    dec = &THREEBYTE38_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEBYTE_3A:
    dec = &THREEBYTE3A_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case XOP8_MAP:
    dec = &XOP8_MAP_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case XOP9_MAP:
    dec = &XOP9_MAP_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case XOPA_MAP:
    dec = &XOPA_MAP_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEDNOW_MAP:
    dec =
        &THREEDNOW_MAP_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case MAP4:
    dec = &MAP4_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case MAP5:
    dec = &MAP5_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case MAP6:
    dec = &MAP6_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case MAP7:
    dec = &MAP7_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  }

  switch (dec->modrm_type) {
  default:
    llvm_unreachable("Corrupt table!  Unknown modrm_type");
    return 0;
  case MODRM_ONEENTRY:
    return modRMTable[dec->instructionIDs];
  case MODRM_SPLITRM:
    if (modFromModRM(modRM) == 0x3)
      return modRMTable[dec->instructionIDs + 1];
    return modRMTable[dec->instructionIDs];
  case MODRM_SPLITREG:
    if (modFromModRM(modRM) == 0x3)
      return modRMTable[dec->instructionIDs + ((modRM & 0x38) >> 3) + 8];
    return modRMTable[dec->instructionIDs + ((modRM & 0x38) >> 3)];
  case MODRM_SPLITMISC:
    if (modFromModRM(modRM) == 0x3)
      return modRMTable[dec->instructionIDs + (modRM & 0x3f) + 8];
    return modRMTable[dec->instructionIDs + ((modRM & 0x38) >> 3)];
  case MODRM_FULL:
    return modRMTable[dec->instructionIDs + modRM];
  }
}

static bool peek(struct InternalInstruction *insn, uint8_t &byte) {
  uint64_t offset = insn->readerCursor - insn->startLocation;
  if (offset >= insn->bytes.size())
    return true;
  byte = insn->bytes[offset];
  return false;
}

template <typename T> static bool consume(InternalInstruction *insn, T &ptr) {
  auto r = insn->bytes;
  uint64_t offset = insn->readerCursor - insn->startLocation;
  if (offset + sizeof(T) > r.size())
    return true;
  ptr = support::endian::read<T>(&r[offset], llvm::endianness::little);
  insn->readerCursor += sizeof(T);
  return false;
}

static bool isREX(struct InternalInstruction *insn, uint8_t prefix) {
  return insn->mode == MODE_64BIT && prefix >= 0x40 && prefix <= 0x4f;
}

static bool isREX2(struct InternalInstruction *insn, uint8_t prefix) {
  return insn->mode == MODE_64BIT && prefix == 0xd5;
}

// Consumes all of an instruction's prefix bytes, and marks the
// instruction as having them.  Also sets the instruction's default operand,
// address, and other relevant data sizes to report operands correctly.
//
// insn must not be empty.
static int readPrefixes(struct InternalInstruction *insn) {
  bool isPrefix = true;
  uint8_t byte = 0;
  uint8_t nextByte;

  LLVM_DEBUG(dbgs() << "readPrefixes()");

  while (isPrefix) {
    // If we fail reading prefixes, just stop here and let the opcode reader
    // deal with it.
    if (consume(insn, byte))
      break;

    // If the byte is a LOCK/REP/REPNE prefix and not a part of the opcode, then
    // break and let it be disassembled as a normal "instruction".
    if (insn->readerCursor - 1 == insn->startLocation && byte == 0xf0) // LOCK
      break;

    if ((byte == 0xf2 || byte == 0xf3) && !peek(insn, nextByte)) {
      // If the byte is 0xf2 or 0xf3, and any of the following conditions are
      // met:
      // - it is followed by a LOCK (0xf0) prefix
      // - it is followed by an xchg instruction
      // then it should be disassembled as a xacquire/xrelease not repne/rep.
      if (((nextByte == 0xf0) ||
           ((nextByte & 0xfe) == 0x86 || (nextByte & 0xf8) == 0x90))) {
        insn->xAcquireRelease = true;
        if (!(byte == 0xf3 && nextByte == 0x90)) // PAUSE instruction support
          break;
      }
      // Also if the byte is 0xf3, and the following condition is met:
      // - it is followed by a "mov mem, reg" (opcode 0x88/0x89) or
      //                       "mov mem, imm" (opcode 0xc6/0xc7) instructions.
      // then it should be disassembled as an xrelease not rep.
      if (byte == 0xf3 && (nextByte == 0x88 || nextByte == 0x89 ||
                           nextByte == 0xc6 || nextByte == 0xc7)) {
        insn->xAcquireRelease = true;
        break;
      }
      if (isREX(insn, nextByte)) {
        uint8_t nnextByte;
        // Go to REX prefix after the current one
        if (consume(insn, nnextByte))
          return -1;
        // We should be able to read next byte after REX prefix
        if (peek(insn, nnextByte))
          return -1;
        --insn->readerCursor;
      }
    }

    switch (byte) {
    case 0xf0: // LOCK
      insn->hasLockPrefix = true;
      break;
    case 0xf2:   // REPNE/REPNZ
    case 0xf3: { // REP or REPE/REPZ
      uint8_t nextByte;
      if (peek(insn, nextByte))
        break;
      // TODO:
      //  1. There could be several 0x66
      //  2. if (nextByte == 0x66) and nextNextByte != 0x0f then
      //      it's not mandatory prefix
      //  3. if (nextByte >= 0x40 && nextByte <= 0x4f) it's REX and we need
      //     0x0f exactly after it to be mandatory prefix
      //  4. if (nextByte == 0xd5) it's REX2 and we need
      //     0x0f exactly after it to be mandatory prefix
      if (isREX(insn, nextByte) || isREX2(insn, nextByte) || nextByte == 0x0f ||
          nextByte == 0x66)
        // The last of 0xf2 /0xf3 is mandatory prefix
        insn->mandatoryPrefix = byte;
      insn->repeatPrefix = byte;
      break;
    }
    case 0x2e: // CS segment override -OR- Branch not taken
      insn->segmentOverride = SEG_OVERRIDE_CS;
      break;
    case 0x36: // SS segment override -OR- Branch taken
      insn->segmentOverride = SEG_OVERRIDE_SS;
      break;
    case 0x3e: // DS segment override
      insn->segmentOverride = SEG_OVERRIDE_DS;
      break;
    case 0x26: // ES segment override
      insn->segmentOverride = SEG_OVERRIDE_ES;
      break;
    case 0x64: // FS segment override
      insn->segmentOverride = SEG_OVERRIDE_FS;
      break;
    case 0x65: // GS segment override
      insn->segmentOverride = SEG_OVERRIDE_GS;
      break;
    case 0x66: { // Operand-size override {
      uint8_t nextByte;
      insn->hasOpSize = true;
      if (peek(insn, nextByte))
        break;
      // 0x66 can't overwrite existing mandatory prefix and should be ignored
      if (!insn->mandatoryPrefix && (nextByte == 0x0f || isREX(insn, nextByte)))
        insn->mandatoryPrefix = byte;
      break;
    }
    case 0x67: // Address-size override
      insn->hasAdSize = true;
      break;
    default: // Not a prefix byte
      isPrefix = false;
      break;
    }

    if (isREX(insn, byte)) {
      insn->rexPrefix = byte;
      isPrefix = true;
      LLVM_DEBUG(dbgs() << format("Found REX prefix 0x%hhx", byte));
    } else if (isPrefix) {
      insn->rexPrefix = 0;
    }

    if (isPrefix)
      LLVM_DEBUG(dbgs() << format("Found prefix 0x%hhx", byte));
  }

  insn->vectorExtensionType = TYPE_NO_VEX_XOP;

  if (byte == 0x62) {
    uint8_t byte1, byte2;
    if (consume(insn, byte1)) {
      LLVM_DEBUG(dbgs() << "Couldn't read second byte of EVEX prefix");
      return -1;
    }

    if (peek(insn, byte2)) {
      LLVM_DEBUG(dbgs() << "Couldn't read third byte of EVEX prefix");
      return -1;
    }

    if ((insn->mode == MODE_64BIT || (byte1 & 0xc0) == 0xc0)) {
      insn->vectorExtensionType = TYPE_EVEX;
    } else {
      --insn->readerCursor; // unconsume byte1
      --insn->readerCursor; // unconsume byte
    }

    if (insn->vectorExtensionType == TYPE_EVEX) {
      insn->vectorExtensionPrefix[0] = byte;
      insn->vectorExtensionPrefix[1] = byte1;
      if (consume(insn, insn->vectorExtensionPrefix[2])) {
        LLVM_DEBUG(dbgs() << "Couldn't read third byte of EVEX prefix");
        return -1;
      }
      if (consume(insn, insn->vectorExtensionPrefix[3])) {
        LLVM_DEBUG(dbgs() << "Couldn't read fourth byte of EVEX prefix");
        return -1;
      }

      if (insn->mode == MODE_64BIT) {
        // We simulate the REX prefix for simplicity's sake
        insn->rexPrefix = 0x40 |
                          (wFromEVEX3of4(insn->vectorExtensionPrefix[2]) << 3) |
                          (rFromEVEX2of4(insn->vectorExtensionPrefix[1]) << 2) |
                          (xFromEVEX2of4(insn->vectorExtensionPrefix[1]) << 1) |
                          (bFromEVEX2of4(insn->vectorExtensionPrefix[1]) << 0);

        // We simulate the REX2 prefix for simplicity's sake
        insn->rex2ExtensionPrefix[1] =
            (r2FromEVEX2of4(insn->vectorExtensionPrefix[1]) << 6) |
            (uFromEVEX3of4(insn->vectorExtensionPrefix[2]) << 5) |
            (b2FromEVEX2of4(insn->vectorExtensionPrefix[1]) << 4);
      }

      LLVM_DEBUG(
          dbgs() << format(
              "Found EVEX prefix 0x%hhx 0x%hhx 0x%hhx 0x%hhx",
              insn->vectorExtensionPrefix[0], insn->vectorExtensionPrefix[1],
              insn->vectorExtensionPrefix[2], insn->vectorExtensionPrefix[3]));
    }
  } else if (byte == 0xc4) {
    uint8_t byte1;
    if (peek(insn, byte1)) {
      LLVM_DEBUG(dbgs() << "Couldn't read second byte of VEX");
      return -1;
    }

    if (insn->mode == MODE_64BIT || (byte1 & 0xc0) == 0xc0)
      insn->vectorExtensionType = TYPE_VEX_3B;
    else
      --insn->readerCursor;

    if (insn->vectorExtensionType == TYPE_VEX_3B) {
      insn->vectorExtensionPrefix[0] = byte;
      consume(insn, insn->vectorExtensionPrefix[1]);
      consume(insn, insn->vectorExtensionPrefix[2]);

      // We simulate the REX prefix for simplicity's sake

      if (insn->mode == MODE_64BIT)
        insn->rexPrefix = 0x40 |
                          (wFromVEX3of3(insn->vectorExtensionPrefix[2]) << 3) |
                          (rFromVEX2of3(insn->vectorExtensionPrefix[1]) << 2) |
                          (xFromVEX2of3(insn->vectorExtensionPrefix[1]) << 1) |
                          (bFromVEX2of3(insn->vectorExtensionPrefix[1]) << 0);

      LLVM_DEBUG(dbgs() << format("Found VEX prefix 0x%hhx 0x%hhx 0x%hhx",
                                  insn->vectorExtensionPrefix[0],
                                  insn->vectorExtensionPrefix[1],
                                  insn->vectorExtensionPrefix[2]));
    }
  } else if (byte == 0xc5) {
    uint8_t byte1;
    if (peek(insn, byte1)) {
      LLVM_DEBUG(dbgs() << "Couldn't read second byte of VEX");
      return -1;
    }

    if (insn->mode == MODE_64BIT || (byte1 & 0xc0) == 0xc0)
      insn->vectorExtensionType = TYPE_VEX_2B;
    else
      --insn->readerCursor;

    if (insn->vectorExtensionType == TYPE_VEX_2B) {
      insn->vectorExtensionPrefix[0] = byte;
      consume(insn, insn->vectorExtensionPrefix[1]);

      if (insn->mode == MODE_64BIT)
        insn->rexPrefix =
            0x40 | (rFromVEX2of2(insn->vectorExtensionPrefix[1]) << 2);

      switch (ppFromVEX2of2(insn->vectorExtensionPrefix[1])) {
      default:
        break;
      case VEX_PREFIX_66:
        insn->hasOpSize = true;
        break;
      }

      LLVM_DEBUG(dbgs() << format("Found VEX prefix 0x%hhx 0x%hhx",
                                  insn->vectorExtensionPrefix[0],
                                  insn->vectorExtensionPrefix[1]));
    }
  } else if (byte == 0x8f) {
    uint8_t byte1;
    if (peek(insn, byte1)) {
      LLVM_DEBUG(dbgs() << "Couldn't read second byte of XOP");
      return -1;
    }

    if ((byte1 & 0x38) != 0x0) // 0 in these 3 bits is a POP instruction.
      insn->vectorExtensionType = TYPE_XOP;
    else
      --insn->readerCursor;

    if (insn->vectorExtensionType == TYPE_XOP) {
      insn->vectorExtensionPrefix[0] = byte;
      consume(insn, insn->vectorExtensionPrefix[1]);
      consume(insn, insn->vectorExtensionPrefix[2]);

      // We simulate the REX prefix for simplicity's sake

      if (insn->mode == MODE_64BIT)
        insn->rexPrefix = 0x40 |
                          (wFromXOP3of3(insn->vectorExtensionPrefix[2]) << 3) |
                          (rFromXOP2of3(insn->vectorExtensionPrefix[1]) << 2) |
                          (xFromXOP2of3(insn->vectorExtensionPrefix[1]) << 1) |
                          (bFromXOP2of3(insn->vectorExtensionPrefix[1]) << 0);

      switch (ppFromXOP3of3(insn->vectorExtensionPrefix[2])) {
      default:
        break;
      case VEX_PREFIX_66:
        insn->hasOpSize = true;
        break;
      }

      LLVM_DEBUG(dbgs() << format("Found XOP prefix 0x%hhx 0x%hhx 0x%hhx",
                                  insn->vectorExtensionPrefix[0],
                                  insn->vectorExtensionPrefix[1],
                                  insn->vectorExtensionPrefix[2]));
    }
  } else if (isREX2(insn, byte)) {
    uint8_t byte1;
    if (peek(insn, byte1)) {
      LLVM_DEBUG(dbgs() << "Couldn't read second byte of REX2");
      return -1;
    }
    insn->rex2ExtensionPrefix[0] = byte;
    consume(insn, insn->rex2ExtensionPrefix[1]);

    // We simulate the REX prefix for simplicity's sake
    insn->rexPrefix = 0x40 | (wFromREX2(insn->rex2ExtensionPrefix[1]) << 3) |
                      (rFromREX2(insn->rex2ExtensionPrefix[1]) << 2) |
                      (xFromREX2(insn->rex2ExtensionPrefix[1]) << 1) |
                      (bFromREX2(insn->rex2ExtensionPrefix[1]) << 0);
    LLVM_DEBUG(dbgs() << format("Found REX2 prefix 0x%hhx 0x%hhx",
                                insn->rex2ExtensionPrefix[0],
                                insn->rex2ExtensionPrefix[1]));
  } else
    --insn->readerCursor;

  if (insn->mode == MODE_16BIT) {
    insn->registerSize = (insn->hasOpSize ? 4 : 2);
    insn->addressSize = (insn->hasAdSize ? 4 : 2);
    insn->displacementSize = (insn->hasAdSize ? 4 : 2);
    insn->immediateSize = (insn->hasOpSize ? 4 : 2);
  } else if (insn->mode == MODE_32BIT) {
    insn->registerSize = (insn->hasOpSize ? 2 : 4);
    insn->addressSize = (insn->hasAdSize ? 2 : 4);
    insn->displacementSize = (insn->hasAdSize ? 2 : 4);
    insn->immediateSize = (insn->hasOpSize ? 2 : 4);
  } else if (insn->mode == MODE_64BIT) {
    insn->displacementSize = 4;
    if (insn->rexPrefix && wFromREX(insn->rexPrefix)) {
      insn->registerSize = 8;
      insn->addressSize = (insn->hasAdSize ? 4 : 8);
      insn->immediateSize = 4;
      insn->hasOpSize = false;
    } else {
      insn->registerSize = (insn->hasOpSize ? 2 : 4);
      insn->addressSize = (insn->hasAdSize ? 4 : 8);
      insn->immediateSize = (insn->hasOpSize ? 2 : 4);
    }
  }

  return 0;
}

// Consumes the SIB byte to determine addressing information.
static int readSIB(struct InternalInstruction *insn) {
  SIBBase sibBaseBase = SIB_BASE_NONE;
  uint8_t index, base;

  LLVM_DEBUG(dbgs() << "readSIB()");
  switch (insn->addressSize) {
  case 2:
  default:
    llvm_unreachable("SIB-based addressing doesn't work in 16-bit mode");
  case 4:
    insn->sibIndexBase = SIB_INDEX_EAX;
    sibBaseBase = SIB_BASE_EAX;
    break;
  case 8:
    insn->sibIndexBase = SIB_INDEX_RAX;
    sibBaseBase = SIB_BASE_RAX;
    break;
  }

  if (consume(insn, insn->sib))
    return -1;

  index = indexFromSIB(insn->sib) | (xFromREX(insn->rexPrefix) << 3) |
          (x2FromREX2(insn->rex2ExtensionPrefix[1]) << 4);

  if (index == 0x4) {
    insn->sibIndex = SIB_INDEX_NONE;
  } else {
    insn->sibIndex = (SIBIndex)(insn->sibIndexBase + index);
  }

  insn->sibScale = 1 << scaleFromSIB(insn->sib);

  base = baseFromSIB(insn->sib) | (bFromREX(insn->rexPrefix) << 3) |
         (b2FromREX2(insn->rex2ExtensionPrefix[1]) << 4);

  switch (base) {
  case 0x5:
  case 0xd:
    switch (modFromModRM(insn->modRM)) {
    case 0x0:
      insn->eaDisplacement = EA_DISP_32;
      insn->sibBase = SIB_BASE_NONE;
      break;
    case 0x1:
      insn->eaDisplacement = EA_DISP_8;
      insn->sibBase = (SIBBase)(sibBaseBase + base);
      break;
    case 0x2:
      insn->eaDisplacement = EA_DISP_32;
      insn->sibBase = (SIBBase)(sibBaseBase + base);
      break;
    default:
      llvm_unreachable("Cannot have Mod = 0b11 and a SIB byte");
    }
    break;
  default:
    insn->sibBase = (SIBBase)(sibBaseBase + base);
    break;
  }

  return 0;
}

static int readDisplacement(struct InternalInstruction *insn) {
  int8_t d8;
  int16_t d16;
  int32_t d32;
  LLVM_DEBUG(dbgs() << "readDisplacement()");

  insn->displacementOffset = insn->readerCursor - insn->startLocation;
  switch (insn->eaDisplacement) {
  case EA_DISP_NONE:
    break;
  case EA_DISP_8:
    if (consume(insn, d8))
      return -1;
    insn->displacement = d8;
    break;
  case EA_DISP_16:
    if (consume(insn, d16))
      return -1;
    insn->displacement = d16;
    break;
  case EA_DISP_32:
    if (consume(insn, d32))
      return -1;
    insn->displacement = d32;
    break;
  }

  return 0;
}

// Consumes all addressing information (ModR/M byte, SIB byte, and displacement.
static int readModRM(struct InternalInstruction *insn) {
  uint8_t mod, rm, reg;
  LLVM_DEBUG(dbgs() << "readModRM()");

  if (insn->consumedModRM)
    return 0;

  if (consume(insn, insn->modRM))
    return -1;
  insn->consumedModRM = true;

  mod = modFromModRM(insn->modRM);
  rm = rmFromModRM(insn->modRM);
  reg = regFromModRM(insn->modRM);

  // This goes by insn->registerSize to pick the correct register, which messes
  // up if we're using (say) XMM or 8-bit register operands. That gets fixed in
  // fixupReg().
  switch (insn->registerSize) {
  case 2:
    insn->regBase = MODRM_REG_AX;
    insn->eaRegBase = EA_REG_AX;
    break;
  case 4:
    insn->regBase = MODRM_REG_EAX;
    insn->eaRegBase = EA_REG_EAX;
    break;
  case 8:
    insn->regBase = MODRM_REG_RAX;
    insn->eaRegBase = EA_REG_RAX;
    break;
  }

  reg |= (rFromREX(insn->rexPrefix) << 3) |
         (r2FromREX2(insn->rex2ExtensionPrefix[1]) << 4);
  rm |= (bFromREX(insn->rexPrefix) << 3) |
        (b2FromREX2(insn->rex2ExtensionPrefix[1]) << 4);

  if (insn->vectorExtensionType == TYPE_EVEX && insn->mode == MODE_64BIT)
    reg |= r2FromEVEX2of4(insn->vectorExtensionPrefix[1]) << 4;

  insn->reg = (Reg)(insn->regBase + reg);

  switch (insn->addressSize) {
  case 2: {
    EABase eaBaseBase = EA_BASE_BX_SI;

    switch (mod) {
    case 0x0:
      if (rm == 0x6) {
        insn->eaBase = EA_BASE_NONE;
        insn->eaDisplacement = EA_DISP_16;
        if (readDisplacement(insn))
          return -1;
      } else {
        insn->eaBase = (EABase)(eaBaseBase + rm);
        insn->eaDisplacement = EA_DISP_NONE;
      }
      break;
    case 0x1:
      insn->eaBase = (EABase)(eaBaseBase + rm);
      insn->eaDisplacement = EA_DISP_8;
      insn->displacementSize = 1;
      if (readDisplacement(insn))
        return -1;
      break;
    case 0x2:
      insn->eaBase = (EABase)(eaBaseBase + rm);
      insn->eaDisplacement = EA_DISP_16;
      if (readDisplacement(insn))
        return -1;
      break;
    case 0x3:
      insn->eaBase = (EABase)(insn->eaRegBase + rm);
      if (readDisplacement(insn))
        return -1;
      break;
    }
    break;
  }
  case 4:
  case 8: {
    EABase eaBaseBase = (insn->addressSize == 4 ? EA_BASE_EAX : EA_BASE_RAX);

    switch (mod) {
    case 0x0:
      insn->eaDisplacement = EA_DISP_NONE; // readSIB may override this
      // In determining whether RIP-relative mode is used (rm=5),
      // or whether a SIB byte is present (rm=4),
      // the extension bits (REX.b and EVEX.x) are ignored.
      switch (rm & 7) {
      case 0x4: // SIB byte is present
        insn->eaBase = (insn->addressSize == 4 ? EA_BASE_sib : EA_BASE_sib64);
        if (readSIB(insn) || readDisplacement(insn))
          return -1;
        break;
      case 0x5: // RIP-relative
        insn->eaBase = EA_BASE_NONE;
        insn->eaDisplacement = EA_DISP_32;
        if (readDisplacement(insn))
          return -1;
        break;
      default:
        insn->eaBase = (EABase)(eaBaseBase + rm);
        break;
      }
      break;
    case 0x1:
      insn->displacementSize = 1;
      [[fallthrough]];
    case 0x2:
      insn->eaDisplacement = (mod == 0x1 ? EA_DISP_8 : EA_DISP_32);
      switch (rm & 7) {
      case 0x4: // SIB byte is present
        insn->eaBase = EA_BASE_sib;
        if (readSIB(insn) || readDisplacement(insn))
          return -1;
        break;
      default:
        insn->eaBase = (EABase)(eaBaseBase + rm);
        if (readDisplacement(insn))
          return -1;
        break;
      }
      break;
    case 0x3:
      insn->eaDisplacement = EA_DISP_NONE;
      insn->eaBase = (EABase)(insn->eaRegBase + rm);
      break;
    }
    break;
  }
  } // switch (insn->addressSize)

  return 0;
}

#define GENERIC_FIXUP_FUNC(name, base, prefix)                                 \
  static uint16_t name(struct InternalInstruction *insn, OperandType type,     \
                       uint8_t index, uint8_t *valid) {                        \
    *valid = 1;                                                                \
    switch (type) {                                                            \
    default:                                                                   \
      debug("Unhandled register type");                                        \
      *valid = 0;                                                              \
      return 0;                                                                \
    case TYPE_Rv:                                                              \
      return base + index;                                                     \
    case TYPE_R8:                                                              \
      if (insn->rexPrefix && index >= 4 && index <= 7)                         \
        return prefix##_SPL + (index - 4);                                     \
      else                                                                     \
        return prefix##_AL + index;                                            \
    case TYPE_R16:                                                             \
      return prefix##_AX + index;                                              \
    case TYPE_R32:                                                             \
      return prefix##_EAX + index;                                             \
    case TYPE_R64:                                                             \
      return prefix##_RAX + index;                                             \
    case TYPE_ZMM:                                                             \
      return prefix##_ZMM0 + index;                                            \
    case TYPE_YMM:                                                             \
      return prefix##_YMM0 + index;                                            \
    case TYPE_XMM:                                                             \
      return prefix##_XMM0 + index;                                            \
    case TYPE_TMM:                                                             \
      if (index > 7)                                                           \
        *valid = 0;                                                            \
      return prefix##_TMM0 + index;                                            \
    case TYPE_VK:                                                              \
      index &= 0xf;                                                            \
      if (index > 7)                                                           \
        *valid = 0;                                                            \
      return prefix##_K0 + index;                                              \
    case TYPE_VK_PAIR:                                                         \
      if (index > 7)                                                           \
        *valid = 0;                                                            \
      return prefix##_K0_K1 + (index / 2);                                     \
    case TYPE_MM64:                                                            \
      return prefix##_MM0 + (index & 0x7);                                     \
    case TYPE_SEGMENTREG:                                                      \
      if ((index & 7) > 5)                                                     \
        *valid = 0;                                                            \
      return prefix##_ES + (index & 7);                                        \
    case TYPE_DEBUGREG:                                                        \
      return prefix##_DR0 + index;                                             \
    case TYPE_CONTROLREG:                                                      \
      return prefix##_CR0 + index;                                             \
    case TYPE_MVSIBX:                                                          \
      return prefix##_XMM0 + index;                                            \
    case TYPE_MVSIBY:                                                          \
      return prefix##_YMM0 + index;                                            \
    case TYPE_MVSIBZ:                                                          \
      return prefix##_ZMM0 + index;                                            \
    }                                                                          \
  }

// Consult an operand type to determine the meaning of the reg or R/M field. If
// the operand is an XMM operand, for example, an operand would be XMM0 instead
// of AX, which readModRM() would otherwise misinterpret it as.
//
// @param insn  - The instruction containing the operand.
// @param type  - The operand type.
// @param index - The existing value of the field as reported by readModRM().
// @param valid - The address of a uint8_t.  The target is set to 1 if the
//                field is valid for the register class; 0 if not.
// @return      - The proper value.
GENERIC_FIXUP_FUNC(fixupRegValue, insn->regBase, MODRM_REG)
GENERIC_FIXUP_FUNC(fixupRMValue, insn->eaRegBase, EA_REG)

// Consult an operand specifier to determine which of the fixup*Value functions
// to use in correcting readModRM()'ss interpretation.
//
// @param insn  - See fixup*Value().
// @param op    - The operand specifier.
// @return      - 0 if fixup was successful; -1 if the register returned was
//                invalid for its class.
static int fixupReg(struct InternalInstruction *insn,
                    const struct OperandSpecifier *op) {
  uint8_t valid;
  LLVM_DEBUG(dbgs() << "fixupReg()");

  switch ((OperandEncoding)op->encoding) {
  default:
    debug("Expected a REG or R/M encoding in fixupReg");
    return -1;
  case ENCODING_VVVV:
    insn->vvvv =
        (Reg)fixupRegValue(insn, (OperandType)op->type, insn->vvvv, &valid);
    if (!valid)
      return -1;
    break;
  case ENCODING_REG:
    insn->reg = (Reg)fixupRegValue(insn, (OperandType)op->type,
                                   insn->reg - insn->regBase, &valid);
    if (!valid)
      return -1;
    break;
  CASE_ENCODING_RM:
    if (insn->vectorExtensionType == TYPE_EVEX && insn->mode == MODE_64BIT &&
        modFromModRM(insn->modRM) == 3) {
      // EVEX_X can extend the register id to 32 for a non-GPR register that is
      // encoded in RM.
      // mode : MODE_64_BIT
      //  Only 8 vector registers are available in 32 bit mode
      // mod : 3
      //  RM encodes a register
      switch (op->type) {
      case TYPE_Rv:
      case TYPE_R8:
      case TYPE_R16:
      case TYPE_R32:
      case TYPE_R64:
        break;
      default:
        insn->eaBase =
            (EABase)(insn->eaBase +
                     (xFromEVEX2of4(insn->vectorExtensionPrefix[1]) << 4));
        break;
      }
    }
    [[fallthrough]];
  case ENCODING_SIB:
    if (insn->eaBase >= insn->eaRegBase) {
      insn->eaBase = (EABase)fixupRMValue(
          insn, (OperandType)op->type, insn->eaBase - insn->eaRegBase, &valid);
      if (!valid)
        return -1;
    }
    break;
  }

  return 0;
}

// Read the opcode (except the ModR/M byte in the case of extended or escape
// opcodes).
static bool readOpcode(struct InternalInstruction *insn) {
  uint8_t current;
  LLVM_DEBUG(dbgs() << "readOpcode()");

  insn->opcodeType = ONEBYTE;
  if (insn->vectorExtensionType == TYPE_EVEX) {
    switch (mmmFromEVEX2of4(insn->vectorExtensionPrefix[1])) {
    default:
      LLVM_DEBUG(
          dbgs() << format("Unhandled mmm field for instruction (0x%hhx)",
                           mmmFromEVEX2of4(insn->vectorExtensionPrefix[1])));
      return true;
    case VEX_LOB_0F:
      insn->opcodeType = TWOBYTE;
      return consume(insn, insn->opcode);
    case VEX_LOB_0F38:
      insn->opcodeType = THREEBYTE_38;
      return consume(insn, insn->opcode);
    case VEX_LOB_0F3A:
      insn->opcodeType = THREEBYTE_3A;
      return consume(insn, insn->opcode);
    case VEX_LOB_MAP4:
      insn->opcodeType = MAP4;
      return consume(insn, insn->opcode);
    case VEX_LOB_MAP5:
      insn->opcodeType = MAP5;
      return consume(insn, insn->opcode);
    case VEX_LOB_MAP6:
      insn->opcodeType = MAP6;
      return consume(insn, insn->opcode);
    }
  } else if (insn->vectorExtensionType == TYPE_VEX_3B) {
    switch (mmmmmFromVEX2of3(insn->vectorExtensionPrefix[1])) {
    default:
      LLVM_DEBUG(
          dbgs() << format("Unhandled m-mmmm field for instruction (0x%hhx)",
                           mmmmmFromVEX2of3(insn->vectorExtensionPrefix[1])));
      return true;
    case VEX_LOB_0F:
      insn->opcodeType = TWOBYTE;
      return consume(insn, insn->opcode);
    case VEX_LOB_0F38:
      insn->opcodeType = THREEBYTE_38;
      return consume(insn, insn->opcode);
    case VEX_LOB_0F3A:
      insn->opcodeType = THREEBYTE_3A;
      return consume(insn, insn->opcode);
    case VEX_LOB_MAP5:
      insn->opcodeType = MAP5;
      return consume(insn, insn->opcode);
    case VEX_LOB_MAP6:
      insn->opcodeType = MAP6;
      return consume(insn, insn->opcode);
    case VEX_LOB_MAP7:
      insn->opcodeType = MAP7;
      return consume(insn, insn->opcode);
    }
  } else if (insn->vectorExtensionType == TYPE_VEX_2B) {
    insn->opcodeType = TWOBYTE;
    return consume(insn, insn->opcode);
  } else if (insn->vectorExtensionType == TYPE_XOP) {
    switch (mmmmmFromXOP2of3(insn->vectorExtensionPrefix[1])) {
    default:
      LLVM_DEBUG(
          dbgs() << format("Unhandled m-mmmm field for instruction (0x%hhx)",
                           mmmmmFromVEX2of3(insn->vectorExtensionPrefix[1])));
      return true;
    case XOP_MAP_SELECT_8:
      insn->opcodeType = XOP8_MAP;
      return consume(insn, insn->opcode);
    case XOP_MAP_SELECT_9:
      insn->opcodeType = XOP9_MAP;
      return consume(insn, insn->opcode);
    case XOP_MAP_SELECT_A:
      insn->opcodeType = XOPA_MAP;
      return consume(insn, insn->opcode);
    }
  } else if (mFromREX2(insn->rex2ExtensionPrefix[1])) {
    // m bit indicates opcode map 1
    insn->opcodeType = TWOBYTE;
    return consume(insn, insn->opcode);
  }

  if (consume(insn, current))
    return true;

  if (current == 0x0f) {
    LLVM_DEBUG(
        dbgs() << format("Found a two-byte escape prefix (0x%hhx)", current));
    if (consume(insn, current))
      return true;

    if (current == 0x38) {
      LLVM_DEBUG(dbgs() << format("Found a three-byte escape prefix (0x%hhx)",
                                  current));
      if (consume(insn, current))
        return true;

      insn->opcodeType = THREEBYTE_38;
    } else if (current == 0x3a) {
      LLVM_DEBUG(dbgs() << format("Found a three-byte escape prefix (0x%hhx)",
                                  current));
      if (consume(insn, current))
        return true;

      insn->opcodeType = THREEBYTE_3A;
    } else if (current == 0x0f) {
      LLVM_DEBUG(
          dbgs() << format("Found a 3dnow escape prefix (0x%hhx)", current));

      // Consume operands before the opcode to comply with the 3DNow encoding
      if (readModRM(insn))
        return true;

      if (consume(insn, current))
        return true;

      insn->opcodeType = THREEDNOW_MAP;
    } else {
      LLVM_DEBUG(dbgs() << "Didn't find a three-byte escape prefix");
      insn->opcodeType = TWOBYTE;
    }
  } else if (insn->mandatoryPrefix)
    // The opcode with mandatory prefix must start with opcode escape.
    // If not it's legacy repeat prefix
    insn->mandatoryPrefix = 0;

  // At this point we have consumed the full opcode.
  // Anything we consume from here on must be unconsumed.
  insn->opcode = current;

  return false;
}

// Determine whether equiv is the 16-bit equivalent of orig (32-bit or 64-bit).
static bool is16BitEquivalent(const char *orig, const char *equiv) {
  for (int i = 0;; i++) {
    if (orig[i] == '\0' && equiv[i] == '\0')
      return true;
    if (orig[i] == '\0' || equiv[i] == '\0')
      return false;
    if (orig[i] != equiv[i]) {
      if ((orig[i] == 'Q' || orig[i] == 'L') && equiv[i] == 'W')
        continue;
      if ((orig[i] == '6' || orig[i] == '3') && equiv[i] == '1')
        continue;
      if ((orig[i] == '4' || orig[i] == '2') && equiv[i] == '6')
        continue;
      return false;
    }
  }
}

// Determine whether this instruction is a 64-bit instruction.
static bool is64Bit(const char *name) {
  for (int i = 0;; ++i) {
    if (name[i] == '\0')
      return false;
    if (name[i] == '6' && name[i + 1] == '4')
      return true;
  }
}

// Determine the ID of an instruction, consuming the ModR/M byte as appropriate
// for extended and escape opcodes, and using a supplied attribute mask.
static int getInstructionIDWithAttrMask(uint16_t *instructionID,
                                        struct InternalInstruction *insn,
                                        uint16_t attrMask) {
  auto insnCtx = InstructionContext(x86DisassemblerContexts[attrMask]);
  const ContextDecision *decision;
  switch (insn->opcodeType) {
  case ONEBYTE:
    decision = &ONEBYTE_SYM;
    break;
  case TWOBYTE:
    decision = &TWOBYTE_SYM;
    break;
  case THREEBYTE_38:
    decision = &THREEBYTE38_SYM;
    break;
  case THREEBYTE_3A:
    decision = &THREEBYTE3A_SYM;
    break;
  case XOP8_MAP:
    decision = &XOP8_MAP_SYM;
    break;
  case XOP9_MAP:
    decision = &XOP9_MAP_SYM;
    break;
  case XOPA_MAP:
    decision = &XOPA_MAP_SYM;
    break;
  case THREEDNOW_MAP:
    decision = &THREEDNOW_MAP_SYM;
    break;
  case MAP4:
    decision = &MAP4_SYM;
    break;
  case MAP5:
    decision = &MAP5_SYM;
    break;
  case MAP6:
    decision = &MAP6_SYM;
    break;
  case MAP7:
    decision = &MAP7_SYM;
    break;
  }

  if (decision->opcodeDecisions[insnCtx]
          .modRMDecisions[insn->opcode]
          .modrm_type != MODRM_ONEENTRY) {
    if (readModRM(insn))
      return -1;
    *instructionID =
        decode(insn->opcodeType, insnCtx, insn->opcode, insn->modRM);
  } else {
    *instructionID = decode(insn->opcodeType, insnCtx, insn->opcode, 0);
  }

  return 0;
}

// Determine the ID of an instruction, consuming the ModR/M byte as appropriate
// for extended and escape opcodes. Determines the attributes and context for
// the instruction before doing so.
static int getInstructionID(struct InternalInstruction *insn,
                            const MCInstrInfo *mii) {
  uint16_t attrMask;
  uint16_t instructionID;

  LLVM_DEBUG(dbgs() << "getID()");

  attrMask = ATTR_NONE;

  if (insn->mode == MODE_64BIT)
    attrMask |= ATTR_64BIT;

  if (insn->vectorExtensionType != TYPE_NO_VEX_XOP) {
    attrMask |= (insn->vectorExtensionType == TYPE_EVEX) ? ATTR_EVEX : ATTR_VEX;

    if (insn->vectorExtensionType == TYPE_EVEX) {
      switch (ppFromEVEX3of4(insn->vectorExtensionPrefix[2])) {
      case VEX_PREFIX_66:
        attrMask |= ATTR_OPSIZE;
        break;
      case VEX_PREFIX_F3:
        attrMask |= ATTR_XS;
        break;
      case VEX_PREFIX_F2:
        attrMask |= ATTR_XD;
        break;
      }

      if (zFromEVEX4of4(insn->vectorExtensionPrefix[3]))
        attrMask |= ATTR_EVEXKZ;
      if (bFromEVEX4of4(insn->vectorExtensionPrefix[3]))
        attrMask |= ATTR_EVEXB;
      // nf bit is the MSB of aaa
      if (nfFromEVEX4of4(insn->vectorExtensionPrefix[3]) &&
          insn->opcodeType == MAP4)
        attrMask |= ATTR_EVEXNF;
      else if (aaaFromEVEX4of4(insn->vectorExtensionPrefix[3]))
        attrMask |= ATTR_EVEXK;
      if (lFromEVEX4of4(insn->vectorExtensionPrefix[3]))
        attrMask |= ATTR_VEXL;
      if (l2FromEVEX4of4(insn->vectorExtensionPrefix[3]))
        attrMask |= ATTR_EVEXL2;
    } else if (insn->vectorExtensionType == TYPE_VEX_3B) {
      switch (ppFromVEX3of3(insn->vectorExtensionPrefix[2])) {
      case VEX_PREFIX_66:
        attrMask |= ATTR_OPSIZE;
        break;
      case VEX_PREFIX_F3:
        attrMask |= ATTR_XS;
        break;
      case VEX_PREFIX_F2:
        attrMask |= ATTR_XD;
        break;
      }

      if (lFromVEX3of3(insn->vectorExtensionPrefix[2]))
        attrMask |= ATTR_VEXL;
    } else if (insn->vectorExtensionType == TYPE_VEX_2B) {
      switch (ppFromVEX2of2(insn->vectorExtensionPrefix[1])) {
      case VEX_PREFIX_66:
        attrMask |= ATTR_OPSIZE;
        if (insn->hasAdSize)
          attrMask |= ATTR_ADSIZE;
        break;
      case VEX_PREFIX_F3:
        attrMask |= ATTR_XS;
        break;
      case VEX_PREFIX_F2:
        attrMask |= ATTR_XD;
        break;
      }

      if (lFromVEX2of2(insn->vectorExtensionPrefix[1]))
        attrMask |= ATTR_VEXL;
    } else if (insn->vectorExtensionType == TYPE_XOP) {
      switch (ppFromXOP3of3(insn->vectorExtensionPrefix[2])) {
      case VEX_PREFIX_66:
        attrMask |= ATTR_OPSIZE;
        break;
      case VEX_PREFIX_F3:
        attrMask |= ATTR_XS;
        break;
      case VEX_PREFIX_F2:
        attrMask |= ATTR_XD;
        break;
      }

      if (lFromXOP3of3(insn->vectorExtensionPrefix[2]))
        attrMask |= ATTR_VEXL;
    } else {
      return -1;
    }
  } else if (!insn->mandatoryPrefix) {
    // If we don't have mandatory prefix we should use legacy prefixes here
    if (insn->hasOpSize && (insn->mode != MODE_16BIT))
      attrMask |= ATTR_OPSIZE;
    if (insn->hasAdSize)
      attrMask |= ATTR_ADSIZE;
    if (insn->opcodeType == ONEBYTE) {
      if (insn->repeatPrefix == 0xf3 && (insn->opcode == 0x90))
        // Special support for PAUSE
        attrMask |= ATTR_XS;
    } else {
      if (insn->repeatPrefix == 0xf2)
        attrMask |= ATTR_XD;
      else if (insn->repeatPrefix == 0xf3)
        attrMask |= ATTR_XS;
    }
  } else {
    switch (insn->mandatoryPrefix) {
    case 0xf2:
      attrMask |= ATTR_XD;
      break;
    case 0xf3:
      attrMask |= ATTR_XS;
      break;
    case 0x66:
      if (insn->mode != MODE_16BIT)
        attrMask |= ATTR_OPSIZE;
      if (insn->hasAdSize)
        attrMask |= ATTR_ADSIZE;
      break;
    case 0x67:
      attrMask |= ATTR_ADSIZE;
      break;
    }
  }

  if (insn->rexPrefix & 0x08) {
    attrMask |= ATTR_REXW;
    attrMask &= ~ATTR_ADSIZE;
  }

  // Absolute jump and pushp/popp need special handling
  if (insn->rex2ExtensionPrefix[0] == 0xd5 && insn->opcodeType == ONEBYTE &&
      (insn->opcode == 0xA1 || (insn->opcode & 0xf0) == 0x50))
    attrMask |= ATTR_REX2;

  if (insn->mode == MODE_16BIT) {
    // JCXZ/JECXZ need special handling for 16-bit mode because the meaning
    // of the AdSize prefix is inverted w.r.t. 32-bit mode.
    if (insn->opcodeType == ONEBYTE && insn->opcode == 0xE3)
      attrMask ^= ATTR_ADSIZE;
    // If we're in 16-bit mode and this is one of the relative jumps and opsize
    // prefix isn't present, we need to force the opsize attribute since the
    // prefix is inverted relative to 32-bit mode.
    if (!insn->hasOpSize && insn->opcodeType == ONEBYTE &&
        (insn->opcode == 0xE8 || insn->opcode == 0xE9))
      attrMask |= ATTR_OPSIZE;

    if (!insn->hasOpSize && insn->opcodeType == TWOBYTE &&
        insn->opcode >= 0x80 && insn->opcode <= 0x8F)
      attrMask |= ATTR_OPSIZE;
  }

  if (getInstructionIDWithAttrMask(&instructionID, insn, attrMask))
    return -1;

  // The following clauses compensate for limitations of the tables.

  if (insn->mode != MODE_64BIT &&
      insn->vectorExtensionType != TYPE_NO_VEX_XOP) {
    // The tables can't distinquish between cases where the W-bit is used to
    // select register size and cases where its a required part of the opcode.
    if ((insn->vectorExtensionType == TYPE_EVEX &&
         wFromEVEX3of4(insn->vectorExtensionPrefix[2])) ||
        (insn->vectorExtensionType == TYPE_VEX_3B &&
         wFromVEX3of3(insn->vectorExtensionPrefix[2])) ||
        (insn->vectorExtensionType == TYPE_XOP &&
         wFromXOP3of3(insn->vectorExtensionPrefix[2]))) {

      uint16_t instructionIDWithREXW;
      if (getInstructionIDWithAttrMask(&instructionIDWithREXW, insn,
                                       attrMask | ATTR_REXW)) {
        insn->instructionID = instructionID;
        insn->spec = &INSTRUCTIONS_SYM[instructionID];
        return 0;
      }

      auto SpecName = mii->getName(instructionIDWithREXW);
      // If not a 64-bit instruction. Switch the opcode.
      if (!is64Bit(SpecName.data())) {
        insn->instructionID = instructionIDWithREXW;
        insn->spec = &INSTRUCTIONS_SYM[instructionIDWithREXW];
        return 0;
      }
    }
  }

  // Absolute moves, umonitor, and movdir64b need special handling.
  // -For 16-bit mode because the meaning of the AdSize and OpSize prefixes are
  //  inverted w.r.t.
  // -For 32-bit mode we need to ensure the ADSIZE prefix is observed in
  //  any position.
  if ((insn->opcodeType == ONEBYTE && ((insn->opcode & 0xFC) == 0xA0)) ||
      (insn->opcodeType == TWOBYTE && (insn->opcode == 0xAE)) ||
      (insn->opcodeType == THREEBYTE_38 && insn->opcode == 0xF8) ||
      (insn->opcodeType == MAP4 && insn->opcode == 0xF8)) {
    // Make sure we observed the prefixes in any position.
    if (insn->hasAdSize)
      attrMask |= ATTR_ADSIZE;
    if (insn->hasOpSize)
      attrMask |= ATTR_OPSIZE;

    // In 16-bit, invert the attributes.
    if (insn->mode == MODE_16BIT) {
      attrMask ^= ATTR_ADSIZE;

      // The OpSize attribute is only valid with the absolute moves.
      if (insn->opcodeType == ONEBYTE && ((insn->opcode & 0xFC) == 0xA0))
        attrMask ^= ATTR_OPSIZE;
    }

    if (getInstructionIDWithAttrMask(&instructionID, insn, attrMask))
      return -1;

    insn->instructionID = instructionID;
    insn->spec = &INSTRUCTIONS_SYM[instructionID];
    return 0;
  }

  if ((insn->mode == MODE_16BIT || insn->hasOpSize) &&
      !(attrMask & ATTR_OPSIZE)) {
    // The instruction tables make no distinction between instructions that
    // allow OpSize anywhere (i.e., 16-bit operations) and that need it in a
    // particular spot (i.e., many MMX operations). In general we're
    // conservative, but in the specific case where OpSize is present but not in
    // the right place we check if there's a 16-bit operation.
    const struct InstructionSpecifier *spec;
    uint16_t instructionIDWithOpsize;
    llvm::StringRef specName, specWithOpSizeName;

    spec = &INSTRUCTIONS_SYM[instructionID];

    if (getInstructionIDWithAttrMask(&instructionIDWithOpsize, insn,
                                     attrMask | ATTR_OPSIZE)) {
      // ModRM required with OpSize but not present. Give up and return the
      // version without OpSize set.
      insn->instructionID = instructionID;
      insn->spec = spec;
      return 0;
    }

    specName = mii->getName(instructionID);
    specWithOpSizeName = mii->getName(instructionIDWithOpsize);

    if (is16BitEquivalent(specName.data(), specWithOpSizeName.data()) &&
        (insn->mode == MODE_16BIT) ^ insn->hasOpSize) {
      insn->instructionID = instructionIDWithOpsize;
      insn->spec = &INSTRUCTIONS_SYM[instructionIDWithOpsize];
    } else {
      insn->instructionID = instructionID;
      insn->spec = spec;
    }
    return 0;
  }

  if (insn->opcodeType == ONEBYTE && insn->opcode == 0x90 &&
      insn->rexPrefix & 0x01) {
    // NOOP shouldn't decode as NOOP if REX.b is set. Instead it should decode
    // as XCHG %r8, %eax.
    const struct InstructionSpecifier *spec;
    uint16_t instructionIDWithNewOpcode;
    const struct InstructionSpecifier *specWithNewOpcode;

    spec = &INSTRUCTIONS_SYM[instructionID];

    // Borrow opcode from one of the other XCHGar opcodes
    insn->opcode = 0x91;

    if (getInstructionIDWithAttrMask(&instructionIDWithNewOpcode, insn,
                                     attrMask)) {
      insn->opcode = 0x90;

      insn->instructionID = instructionID;
      insn->spec = spec;
      return 0;
    }

    specWithNewOpcode = &INSTRUCTIONS_SYM[instructionIDWithNewOpcode];

    // Change back
    insn->opcode = 0x90;

    insn->instructionID = instructionIDWithNewOpcode;
    insn->spec = specWithNewOpcode;

    return 0;
  }

  insn->instructionID = instructionID;
  insn->spec = &INSTRUCTIONS_SYM[insn->instructionID];

  return 0;
}

// Read an operand from the opcode field of an instruction and interprets it
// appropriately given the operand width. Handles AddRegFrm instructions.
//
// @param insn  - the instruction whose opcode field is to be read.
// @param size  - The width (in bytes) of the register being specified.
//                1 means AL and friends, 2 means AX, 4 means EAX, and 8 means
//                RAX.
// @return      - 0 on success; nonzero otherwise.
static int readOpcodeRegister(struct InternalInstruction *insn, uint8_t size) {
  LLVM_DEBUG(dbgs() << "readOpcodeRegister()");

  if (size == 0)
    size = insn->registerSize;

  auto setOpcodeRegister = [&](unsigned base) {
    insn->opcodeRegister =
        (Reg)(base + ((bFromREX(insn->rexPrefix) << 3) |
                      (b2FromREX2(insn->rex2ExtensionPrefix[1]) << 4) |
                      (insn->opcode & 7)));
  };

  switch (size) {
  case 1:
    setOpcodeRegister(MODRM_REG_AL);
    if (insn->rexPrefix && insn->opcodeRegister >= MODRM_REG_AL + 0x4 &&
        insn->opcodeRegister < MODRM_REG_AL + 0x8) {
      insn->opcodeRegister =
          (Reg)(MODRM_REG_SPL + (insn->opcodeRegister - MODRM_REG_AL - 4));
    }

    break;
  case 2:
    setOpcodeRegister(MODRM_REG_AX);
    break;
  case 4:
    setOpcodeRegister(MODRM_REG_EAX);
    break;
  case 8:
    setOpcodeRegister(MODRM_REG_RAX);
    break;
  }

  return 0;
}

// Consume an immediate operand from an instruction, given the desired operand
// size.
//
// @param insn  - The instruction whose operand is to be read.
// @param size  - The width (in bytes) of the operand.
// @return      - 0 if the immediate was successfully consumed; nonzero
//                otherwise.
static int readImmediate(struct InternalInstruction *insn, uint8_t size) {
  uint8_t imm8;
  uint16_t imm16;
  uint32_t imm32;
  uint64_t imm64;

  LLVM_DEBUG(dbgs() << "readImmediate()");

  assert(insn->numImmediatesConsumed < 2 && "Already consumed two immediates");

  insn->immediateSize = size;
  insn->immediateOffset = insn->readerCursor - insn->startLocation;

  switch (size) {
  case 1:
    if (consume(insn, imm8))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm8;
    break;
  case 2:
    if (consume(insn, imm16))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm16;
    break;
  case 4:
    if (consume(insn, imm32))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm32;
    break;
  case 8:
    if (consume(insn, imm64))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm64;
    break;
  default:
    llvm_unreachable("invalid size");
  }

  insn->numImmediatesConsumed++;

  return 0;
}

// Consume vvvv from an instruction if it has a VEX prefix.
static int readVVVV(struct InternalInstruction *insn) {
  LLVM_DEBUG(dbgs() << "readVVVV()");

  int vvvv;
  if (insn->vectorExtensionType == TYPE_EVEX)
    vvvv = (v2FromEVEX4of4(insn->vectorExtensionPrefix[3]) << 4 |
            vvvvFromEVEX3of4(insn->vectorExtensionPrefix[2]));
  else if (insn->vectorExtensionType == TYPE_VEX_3B)
    vvvv = vvvvFromVEX3of3(insn->vectorExtensionPrefix[2]);
  else if (insn->vectorExtensionType == TYPE_VEX_2B)
    vvvv = vvvvFromVEX2of2(insn->vectorExtensionPrefix[1]);
  else if (insn->vectorExtensionType == TYPE_XOP)
    vvvv = vvvvFromXOP3of3(insn->vectorExtensionPrefix[2]);
  else
    return -1;

  if (insn->mode != MODE_64BIT)
    vvvv &= 0xf; // Can only clear bit 4. Bit 3 must be cleared later.

  insn->vvvv = static_cast<Reg>(vvvv);
  return 0;
}

// Read an mask register from the opcode field of an instruction.
//
// @param insn    - The instruction whose opcode field is to be read.
// @return        - 0 on success; nonzero otherwise.
static int readMaskRegister(struct InternalInstruction *insn) {
  LLVM_DEBUG(dbgs() << "readMaskRegister()");

  if (insn->vectorExtensionType != TYPE_EVEX)
    return -1;

  insn->writemask =
      static_cast<Reg>(aaaFromEVEX4of4(insn->vectorExtensionPrefix[3]));
  return 0;
}

// Consults the specifier for an instruction and consumes all
// operands for that instruction, interpreting them as it goes.
static int readOperands(struct InternalInstruction *insn) {
  int hasVVVV, needVVVV;
  int sawRegImm = 0;

  LLVM_DEBUG(dbgs() << "readOperands()");

  // If non-zero vvvv specified, make sure one of the operands uses it.
  hasVVVV = !readVVVV(insn);
  needVVVV = hasVVVV && (insn->vvvv != 0);

  for (const auto &Op : x86OperandSets[insn->spec->operands]) {
    switch (Op.encoding) {
    case ENCODING_NONE:
    case ENCODING_SI:
    case ENCODING_DI:
      break;
    CASE_ENCODING_VSIB:
      // VSIB can use the V2 bit so check only the other bits.
      if (needVVVV)
        needVVVV = hasVVVV & ((insn->vvvv & 0xf) != 0);
      if (readModRM(insn))
        return -1;

      // Reject if SIB wasn't used.
      if (insn->eaBase != EA_BASE_sib && insn->eaBase != EA_BASE_sib64)
        return -1;

      // If sibIndex was set to SIB_INDEX_NONE, index offset is 4.
      if (insn->sibIndex == SIB_INDEX_NONE)
        insn->sibIndex = (SIBIndex)(insn->sibIndexBase + 4);

      // If EVEX.v2 is set this is one of the 16-31 registers.
      if (insn->vectorExtensionType == TYPE_EVEX && insn->mode == MODE_64BIT &&
          v2FromEVEX4of4(insn->vectorExtensionPrefix[3]))
        insn->sibIndex = (SIBIndex)(insn->sibIndex + 16);

      // Adjust the index register to the correct size.
      switch ((OperandType)Op.type) {
      default:
        debug("Unhandled VSIB index type");
        return -1;
      case TYPE_MVSIBX:
        insn->sibIndex =
            (SIBIndex)(SIB_INDEX_XMM0 + (insn->sibIndex - insn->sibIndexBase));
        break;
      case TYPE_MVSIBY:
        insn->sibIndex =
            (SIBIndex)(SIB_INDEX_YMM0 + (insn->sibIndex - insn->sibIndexBase));
        break;
      case TYPE_MVSIBZ:
        insn->sibIndex =
            (SIBIndex)(SIB_INDEX_ZMM0 + (insn->sibIndex - insn->sibIndexBase));
        break;
      }

      // Apply the AVX512 compressed displacement scaling factor.
      if (Op.encoding != ENCODING_REG && insn->eaDisplacement == EA_DISP_8)
        insn->displacement *= 1 << (Op.encoding - ENCODING_VSIB);
      break;
    case ENCODING_SIB:
      // Reject if SIB wasn't used.
      if (insn->eaBase != EA_BASE_sib && insn->eaBase != EA_BASE_sib64)
        return -1;
      if (readModRM(insn))
        return -1;
      if (fixupReg(insn, &Op))
        return -1;
      break;
    case ENCODING_REG:
    CASE_ENCODING_RM:
      if (readModRM(insn))
        return -1;
      if (fixupReg(insn, &Op))
        return -1;
      // Apply the AVX512 compressed displacement scaling factor.
      if (Op.encoding != ENCODING_REG && insn->eaDisplacement == EA_DISP_8)
        insn->displacement *= 1 << (Op.encoding - ENCODING_RM);
      break;
    case ENCODING_IB:
      if (sawRegImm) {
        // Saw a register immediate so don't read again and instead split the
        // previous immediate. FIXME: This is a hack.
        insn->immediates[insn->numImmediatesConsumed] =
            insn->immediates[insn->numImmediatesConsumed - 1] & 0xf;
        ++insn->numImmediatesConsumed;
        break;
      }
      if (readImmediate(insn, 1))
        return -1;
      if (Op.type == TYPE_XMM || Op.type == TYPE_YMM)
        sawRegImm = 1;
      break;
    case ENCODING_IW:
      if (readImmediate(insn, 2))
        return -1;
      break;
    case ENCODING_ID:
      if (readImmediate(insn, 4))
        return -1;
      break;
    case ENCODING_IO:
      if (readImmediate(insn, 8))
        return -1;
      break;
    case ENCODING_Iv:
      if (readImmediate(insn, insn->immediateSize))
        return -1;
      break;
    case ENCODING_Ia:
      if (readImmediate(insn, insn->addressSize))
        return -1;
      break;
    case ENCODING_IRC:
      insn->RC = (l2FromEVEX4of4(insn->vectorExtensionPrefix[3]) << 1) |
                 lFromEVEX4of4(insn->vectorExtensionPrefix[3]);
      break;
    case ENCODING_RB:
      if (readOpcodeRegister(insn, 1))
        return -1;
      break;
    case ENCODING_RW:
      if (readOpcodeRegister(insn, 2))
        return -1;
      break;
    case ENCODING_RD:
      if (readOpcodeRegister(insn, 4))
        return -1;
      break;
    case ENCODING_RO:
      if (readOpcodeRegister(insn, 8))
        return -1;
      break;
    case ENCODING_Rv:
      if (readOpcodeRegister(insn, 0))
        return -1;
      break;
    case ENCODING_CC:
      insn->immediates[1] = insn->opcode & 0xf;
      break;
    case ENCODING_FP:
      break;
    case ENCODING_VVVV:
      needVVVV = 0; // Mark that we have found a VVVV operand.
      if (!hasVVVV)
        return -1;
      if (insn->mode != MODE_64BIT)
        insn->vvvv = static_cast<Reg>(insn->vvvv & 0x7);
      if (fixupReg(insn, &Op))
        return -1;
      break;
    case ENCODING_WRITEMASK:
      if (readMaskRegister(insn))
        return -1;
      break;
    case ENCODING_DUP:
      break;
    default:
      LLVM_DEBUG(dbgs() << "Encountered an operand with an unknown encoding.");
      return -1;
    }
  }

  // If we didn't find ENCODING_VVVV operand, but non-zero vvvv present, fail
  if (needVVVV)
    return -1;

  return 0;
}

} // namespace X86Disassembler
} // namespace llvm

#endif
