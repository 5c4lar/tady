/*
 *
 * Copyright (c) 2009-2011
 * vol4ok <admin@vol4ok.net> PGP KEY ID: 26EC143CCDC61C9D
 *

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
 *
 *  @modifier : rrrfff@foxmail.com
 *  https://github.com/rrrfff/LDasm
 *
 */
#include "table.h"
#include <cstddef> // for NULL definition
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <sys/cdefs.h>
#include <iostream>
namespace py = pybind11;

struct ldasm_data {
  uint8_t instr_len;
  uint8_t flags;
  uint8_t rex;
  uint8_t modrm;
  uint8_t sib;
  uint8_t opcd_offset;
  uint8_t opcd_size;
  uint8_t disp_offset;
  uint8_t disp_size;
  uint8_t imm_offset;
  uint8_t imm_size;
  uint8_t f;
  uint8_t op;
};

/*
 Description:
    Disassemble one instruction

 Arguments:
    code    - pointer to the code for disassemble
    ld      - pointer to structure ldasm_data

 Return:
    length of instruction
 */
inline unsigned int ldasm(const void *code, ldasm_data *ld, bool is64 = 1) {
  uint8_t *p = (uint8_t *)code;
  uint8_t s, op, f;
  uint8_t rexw, pr_66, pr_67;

  s = rexw = pr_66 = pr_67 = 0;

  /* init output data */
  memset(ld, 0, sizeof(ldasm_data));

  /* phase 1: parse prefixies */
  while (flags_table[*p] & OP_PREFIX) {
    if (*p == 0x66)
      pr_66 = 1u;
    if (*p == 0x67)
      pr_67 = 1u;
    ++p;
    ++s;
    ld->flags |= F_PREFIX;
    if (s == 15u) {
      ld->flags |= F_INVALID;
      ld->instr_len = s;
      return s;
    } // if
  }

  if (is64) {
    /* parse REX prefix */
    if (*p >> 4u == 4u) {
      ld->rex = *p;
      rexw = (ld->rex >> 3u) & 1u;
      ld->flags |= F_REX;
      ++p;
      ++s;
    } // if

    /* can be only one REX prefix */
    if (*p >> 4 == 4) {
      ld->flags |= F_INVALID;
      ++s;
      ld->instr_len = s;
      return s;
    } // if
  }

  /* phase 2: parse opcode */
  ld->opcd_offset = (uint8_t)(p - (uint8_t *)code);
  ld->opcd_size = 1;
  op = *p++;
  ++s;

  /* is 2 byte opcode? */
  if (op == 0x0F) {
    op = *p++;
    ++s;
    ++ld->opcd_size;
    ld->f = f = flags_table_ex[op];
    if (f & OP_INVALID) {
      ld->flags |= F_INVALID;
      ld->instr_len = s;
      ld->op = op;
      return s;
    } // if
    /* for SSE instructions */
    if (f & OP_EXTENDED) {
      op = *p++;
      ++s;
      ++ld->opcd_size;
    } // if
  } else {
    ld->f = f = flags_table[op];
    /* pr_66 = pr_67 for opcodes A0-A3 */
    if (op >= 0xA0 && op <= 0xA3)
      pr_66 = pr_67;
  } // if
  ld->op = op;
  /* phase 3: parse ModR/M, SIB and DISP */
  if (f & OP_MODRM) {
    uint8_t mod = (*p >> 6);
    uint8_t ro = (*p & 0x38) >> 3;
    uint8_t rm = (*p & 7);

    ld->modrm = *p++;
    ++s;
    ld->flags |= F_MODRM;

    /* in F6,F7 opcodes immediate data present if R/O == 0 */
    if (op == 0xF6 && (ro == 0 || ro == 1))
      f |= OP_DATA_I8;
    if (op == 0xF7 && (ro == 0 || ro == 1))
      f |= OP_DATA_I16_I32_I64;

    /* is SIB byte exist? */
    if (mod != 3 && rm == 4 && (is64 || !pr_67)) {
      ld->sib = *p++;
      ++s;
      ld->flags |= F_SIB;

      /* if base == 5 and mod == 0 */
      if ((ld->sib & 7) == 5 && mod == 0) {
        ld->disp_size = 4;
      } // if
    } // if

    switch (mod) {
    case 0u:
      if (is64) {
        if (rm == 5u) {
          ld->disp_size = 4u;
          ld->flags |= F_RELATIVE;
        } // if
      } else if (pr_67) {
        if (rm == 6u)
          ld->disp_size = 2u;
      } else {
        if (rm == 5u)
          ld->disp_size = 4u;
      } // if
      break;
    case 1u:
      ld->disp_size = 1u;
      break;
    case 2u:
      if (is64)
        ld->disp_size = 4u;
      else if (pr_67)
        ld->disp_size = 2u;
      else
        ld->disp_size = 4u;
      break;
    }

    if (ld->disp_size) {
      ld->disp_offset = (uint8_t)(p - (uint8_t *)code);
      p += ld->disp_size;
      s += ld->disp_size;
      ld->flags |= F_DISP;
    } // if
  }

  /* phase 4: parse immediate data */
  if ((rexw || (is64 && op >= 0xA0u && op <= 0xA3u)) && f & OP_DATA_I16_I32_I64)
    ld->imm_size = 8u;
  else if (f & OP_DATA_I16_I32 || f & OP_DATA_I16_I32_I64)
    ld->imm_size = 4u - (pr_66 << 1u);

  /* if exist, add OP_DATA_I16 and OP_DATA_I8 size */
  ld->imm_size += f & 3u;

  if (ld->imm_size) {
    s += ld->imm_size;
    ld->imm_offset = (uint8_t)(p - (uint8_t *)code);
    ld->flags |= F_IMM;
    if (f & OP_RELATIVE)
      ld->flags |= F_RELATIVE;
  } // if

  ld->f = f;
  /* instruction is too long */
  if (s > 15u)
    ld->flags |= F_INVALID;
  ld->instr_len = s;
  return s;
}

/*
 Description:
    Evaluates jmp instruction

 Return:
    target address
 */
inline void *evaluate_jmp(const void *code, bool is64 = 1) {
  unsigned char *addr = static_cast<unsigned char *>(const_cast<void *>(code));
  do {
    switch (addr[0]) {
    case 0xe9u:
      addr = addr + 5 + *reinterpret_cast<uint32_t *>(addr + 1);
      break;
    case 0xebu:
      addr = addr + 3 + *reinterpret_cast<uint16_t *>(addr + 1);
      break;
    case 0xffu:
      if (addr[1] == 0x25u) {
        if (is64) {
          addr =
              reinterpret_cast<unsigned char *>(*reinterpret_cast<uint64_t *>(
                  addr + 6 + *reinterpret_cast<uint32_t *>(addr + 2)));
        } else {
          addr =
              reinterpret_cast<unsigned char *>(*reinterpret_cast<uint32_t *>(
                  *reinterpret_cast<uint32_t *>(addr + 2)));
        }
        break;
      } // if
    default:
      code = NULL;
    }
  } while (code);

  return addr;
}

py::array_t<uint8_t> disasm(const py::array_t<uint8_t> &code, bool is64 = 1) {
    // Validate input array
    if (!code.ptr()) {
        throw std::runtime_error("Input array is null");
    }
    
    // Create result array with proper shape
    py::array_t<uint8_t> result({static_cast<py::ssize_t>(code.size()), static_cast<py::ssize_t>(sizeof(ldasm_data))});
    
    // Initialize result buffer
    auto result_buf = result.request();
    if (!result_buf.ptr) {
        throw std::runtime_error("Failed to allocate result buffer");
    }
    std::memset(result_buf.ptr, 0, result_buf.size);
    
    // Process each instruction
    ldasm_data* ld = static_cast<ldasm_data*>(result_buf.ptr);
    for (size_t i = 0; i < code.size(); ++i) {
        ldasm(static_cast<const void*>(code.data() + i), &ld[i], is64);
    }

    return result;
}
