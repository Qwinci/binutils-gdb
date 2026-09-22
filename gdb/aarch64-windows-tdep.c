/* Target-dependent code for AArch64 systems running Windows, for GDB.

   Copyright (C) 2025 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* This file implements SEH-based unwinding for AArch64 Windows PE/COFF
   binaries (target triple aarch64-*-windows-msvc / aarch64-*-windows-gnu).

   Unlike x86_64 and riscv64 (see amd64-windows-tdep.c and
   riscv-windows-tdep.c), the ARM64 exception-handling format is not built
   on the shared PEX64 UNWIND_INFO/RUNTIME_FUNCTION container.  It has its
   own layout, documented at:

     https://learn.microsoft.com/en-us/cpp/build/arm64-exception-handling

   The two salient differences this unwinder must cope with are:

     1. Each .pdata RUNTIME_FUNCTION entry is only *two* 32-bit words:
	the function-start RVA and a second word that is either an RVA to a
	variable-length .xdata block, or -- for canonical prologs/epilogs --
	a self-contained "packed" unwind description.  There is no
	end-address word, so the function length comes from the packed data
	or the .xdata header, and the .pdata binary search finds the entry
	with the greatest start RVA not exceeding the PC.

     2. The .xdata unwind codes explicitly describe epilogues (via epilog
	scopes), so -- unlike the riscv64 unwinder -- epilogue handling is
	driven by the unwind data rather than by pattern-matching instruction
	bytes.  The codes map 1:1 onto prolog/epilog instructions, which lets
	the unwinder cope with a PC in a partially-executed prolog.  */

#include "extract-store-integer.h"
#include "osabi.h"
#include "aarch64-tdep.h"
#include "arch/aarch64.h"
#include "gdbcore.h"
#include "windows-tdep.h"
#include "frame.h"
#include "frame-unwind.h"
#include "objfiles.h"
#include "coff/internal.h"
#include "coff/aarch64.h"
#include "coff/pe.h"
#include "libcoff.h"
#include <algorithm>

/* Cached information about an AArch64 Windows SEH frame.  */

struct aarch64_windows_frame_cache
{
  /* ImageBase for the module.  */
  CORE_ADDR image_base;

  /* Function start and (computed) end rva.  */
  CORE_ADDR start_rva;
  CORE_ADDR end_rva;

  /* Next instruction to be executed.  */
  CORE_ADDR pc;

  /* Current sp.  */
  CORE_ADDR sp;

  /* Address at which each x-register x0-x30 (indexed by GDB regnum minus
     AARCH64_X0_REGNUM, i.e. 0..30, where 29 is fp and 30 is lr) was saved
     by this frame, or 0 if it was not saved (its previous value is then
     the same as its current, live value).  */
  CORE_ADDR prev_gpr_addr[31];

  /* Likewise for the SIMD/FP registers v0-v31 (only d8-d15 are ever
     callee-saved, but the array is sized for all of them for simplicity).
     Indexed by GDB regnum minus AARCH64_V0_REGNUM.  */
  CORE_ADDR prev_fpr_addr[32];

  /* The CFA: the value SP had on entry to this function.  On AArch64 the
     return address is passed in lr rather than pushed by the call, so this
     is not adjusted by any fixed constant.  */
  CORE_ADDR prev_sp;
};

/* On-disk unwind-code first-byte opcode groups.  Only the ones this
   unwinder acts on need names; the rest are handled by length alone.  */
enum aarch64_unwind_op
{
  AARCH64_UOP_ALLOC_S,		/* 000xxxxx  */
  AARCH64_UOP_SAVE_R19R20_X,	/* 001zzzzz  */
  AARCH64_UOP_SAVE_FPLR,	/* 01zzzzzz  */
  AARCH64_UOP_SAVE_FPLR_X,	/* 10zzzzzz  */
  AARCH64_UOP_ALLOC_M,		/* 11000xxx xxxxxxxx  */
  AARCH64_UOP_SAVE_REGP,	/* 110010xx xxzzzzzz  */
  AARCH64_UOP_SAVE_REGP_X,	/* 110011xx xxzzzzzz  */
  AARCH64_UOP_SAVE_REG,		/* 110100xx xxzzzzzz  */
  AARCH64_UOP_SAVE_REG_X,	/* 1101010x xxxzzzzz  */
  AARCH64_UOP_SAVE_LRPAIR,	/* 1101011x xxzzzzzz  */
  AARCH64_UOP_SAVE_FREGP,	/* 1101100x xxzzzzzz  */
  AARCH64_UOP_SAVE_FREGP_X,	/* 1101101x xxzzzzzz  */
  AARCH64_UOP_SAVE_FREG,	/* 1101110x xxzzzzzz  */
  AARCH64_UOP_SAVE_FREG_X,	/* 11011110 xxxzzzzz  */
  AARCH64_UOP_ALLOC_L,		/* 11100000 + 3 bytes  */
  AARCH64_UOP_SET_FP,		/* 11100001  */
  AARCH64_UOP_ADD_FP,		/* 11100010 xxxxxxxx  */
  AARCH64_UOP_NOP,		/* 11100011  */
  AARCH64_UOP_END,		/* 11100100  */
  AARCH64_UOP_END_C,		/* 11100101  */
  AARCH64_UOP_SAVE_NEXT,	/* 11100110  */
  AARCH64_UOP_PAC,		/* 11111100  */
  AARCH64_UOP_UNSUPPORTED	/* SVE saves, save_any_reg, custom stacks,
				   reserved -- not modelled.  */
};

/* Classify an unwind code by its first byte B, and return its total length
   in bytes in *LEN.  */

static enum aarch64_unwind_op
aarch64_windows_unwind_op (gdb_byte b, int *len)
{
  if (b < 0x20) { *len = 1; return AARCH64_UOP_ALLOC_S; }
  if (b < 0x40) { *len = 1; return AARCH64_UOP_SAVE_R19R20_X; }
  if (b < 0x80) { *len = 1; return AARCH64_UOP_SAVE_FPLR; }
  if (b < 0xc0) { *len = 1; return AARCH64_UOP_SAVE_FPLR_X; }
  if (b < 0xc8) { *len = 2; return AARCH64_UOP_ALLOC_M; }
  if (b < 0xcc) { *len = 2; return AARCH64_UOP_SAVE_REGP; }
  if (b < 0xd0) { *len = 2; return AARCH64_UOP_SAVE_REGP_X; }
  if (b < 0xd4) { *len = 2; return AARCH64_UOP_SAVE_REG; }
  if (b < 0xd6) { *len = 2; return AARCH64_UOP_SAVE_REG_X; }
  if (b < 0xd8) { *len = 2; return AARCH64_UOP_SAVE_LRPAIR; }
  if (b < 0xda) { *len = 2; return AARCH64_UOP_SAVE_FREGP; }
  if (b < 0xdc) { *len = 2; return AARCH64_UOP_SAVE_FREGP_X; }
  if (b < 0xde) { *len = 2; return AARCH64_UOP_SAVE_FREG; }
  if (b == 0xde) { *len = 2; return AARCH64_UOP_SAVE_FREG_X; }
  if (b == 0xdf) { *len = 2; return AARCH64_UOP_UNSUPPORTED; }	/* alloc_z */
  if (b == 0xe0) { *len = 4; return AARCH64_UOP_ALLOC_L; }
  if (b == 0xe1) { *len = 1; return AARCH64_UOP_SET_FP; }
  if (b == 0xe2) { *len = 2; return AARCH64_UOP_ADD_FP; }
  if (b == 0xe3) { *len = 1; return AARCH64_UOP_NOP; }
  if (b == 0xe4) { *len = 1; return AARCH64_UOP_END; }
  if (b == 0xe5) { *len = 1; return AARCH64_UOP_END_C; }
  if (b == 0xe6) { *len = 1; return AARCH64_UOP_SAVE_NEXT; }
  if (b == 0xe7) { *len = 3; return AARCH64_UOP_UNSUPPORTED; } /* save_any_reg */
  if (b >= 0xe8 && b <= 0xec) { *len = 1; return AARCH64_UOP_UNSUPPORTED; }
  if (b >= 0xed && b <= 0xf7) { *len = 1; return AARCH64_UOP_UNSUPPORTED; }
  if (b == 0xf8) { *len = 2; return AARCH64_UOP_UNSUPPORTED; }
  if (b == 0xf9) { *len = 3; return AARCH64_UOP_UNSUPPORTED; }
  if (b == 0xfa) { *len = 4; return AARCH64_UOP_UNSUPPORTED; }
  if (b == 0xfb) { *len = 5; return AARCH64_UOP_UNSUPPORTED; }
  if (b == 0xfc) { *len = 1; return AARCH64_UOP_PAC; }
  *len = 1;
  return AARCH64_UOP_UNSUPPORTED;
}

/* Record that x-register REG (0..30) was saved at ADDR in CACHE.  */

static void
aarch64_windows_save_gpr (struct aarch64_windows_frame_cache *cache,
			  int reg, CORE_ADDR addr)
{
  if (reg >= 0 && reg < (int) ARRAY_SIZE (cache->prev_gpr_addr))
    cache->prev_gpr_addr[reg] = addr;
}

/* Record that d-register D (a v-register index 0..31) was saved at ADDR.  */

static void
aarch64_windows_save_fpr (struct aarch64_windows_frame_cache *cache,
			  int d, CORE_ADDR addr)
{
  if (d >= 0 && d < (int) ARRAY_SIZE (cache->prev_fpr_addr))
    cache->prev_fpr_addr[d] = addr;
}

/* Execute a single decoded unwind code onto the running unwind state.  P
   points at the (up to 4) code bytes, OP/LEN come from
   aarch64_windows_unwind_op.  *CUR_SP is the running stack pointer.  The
   save_next continuation state is threaded through *NEXT_KIND (0 none,
   1 int, 2 fp), *NEXT_REG (next register number) and *NEXT_ADDR.

   Returns false if an unsupported/terminating code was hit and decoding of
   this sequence must stop.  */

static bool
aarch64_windows_exec_one (const frame_info_ptr &this_frame,
			  struct aarch64_windows_frame_cache *cache,
			  const gdb_byte *p, enum aarch64_unwind_op op, int len,
			  CORE_ADDR *cur_sp, int *next_kind, int *next_reg,
			  CORE_ADDR *next_addr)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  gdb_byte b0 = p[0];
  gdb_byte b1 = len > 1 ? p[1] : 0;

  switch (op)
    {
    case AARCH64_UOP_ALLOC_S:
      *cur_sp += (b0 & 0x1f) * 16;
      break;

    case AARCH64_UOP_ALLOC_M:
      *cur_sp += (((b0 & 0x7) << 8) | b1) * 16;
      break;

    case AARCH64_UOP_ALLOC_L:
      *cur_sp += (((CORE_ADDR) p[1] << 16) | ((CORE_ADDR) p[2] << 8) | p[3])
		 * 16;
      break;

    case AARCH64_UOP_SAVE_R19R20_X:
      {
	int z = b0 & 0x1f;
	aarch64_windows_save_gpr (cache, 19, *cur_sp);
	aarch64_windows_save_gpr (cache, 20, *cur_sp + 8);
	*next_kind = 1; *next_reg = 21; *next_addr = *cur_sp + 16;
	*cur_sp += z * 8;
      }
      break;

    case AARCH64_UOP_SAVE_FPLR:
      {
	int z = b0 & 0x3f;
	aarch64_windows_save_gpr (cache, 29, *cur_sp + z * 8);
	aarch64_windows_save_gpr (cache, 30, *cur_sp + z * 8 + 8);
      }
      break;

    case AARCH64_UOP_SAVE_FPLR_X:
      {
	int z = b0 & 0x3f;
	aarch64_windows_save_gpr (cache, 29, *cur_sp);
	aarch64_windows_save_gpr (cache, 30, *cur_sp + 8);
	*cur_sp += (z + 1) * 8;
      }
      break;

    case AARCH64_UOP_SAVE_REGP:
      {
	int x = ((b0 & 0x3) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_gpr (cache, 19 + x, *cur_sp + z * 8);
	aarch64_windows_save_gpr (cache, 20 + x, *cur_sp + z * 8 + 8);
	*next_kind = 1; *next_reg = 21 + x; *next_addr = *cur_sp + z * 8 + 16;
      }
      break;

    case AARCH64_UOP_SAVE_REGP_X:
      {
	int x = ((b0 & 0x3) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_gpr (cache, 19 + x, *cur_sp);
	aarch64_windows_save_gpr (cache, 20 + x, *cur_sp + 8);
	*next_kind = 1; *next_reg = 21 + x; *next_addr = *cur_sp + 16;
	*cur_sp += (z + 1) * 8;
      }
      break;

    case AARCH64_UOP_SAVE_REG:
      {
	int x = ((b0 & 0x3) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_gpr (cache, 19 + x, *cur_sp + z * 8);
      }
      break;

    case AARCH64_UOP_SAVE_REG_X:
      {
	int x = ((b0 & 0x1) << 3) | (b1 >> 5);
	int z = b1 & 0x1f;
	aarch64_windows_save_gpr (cache, 19 + x, *cur_sp);
	*cur_sp += (z + 1) * 8;
      }
      break;

    case AARCH64_UOP_SAVE_LRPAIR:
      {
	int x = ((b0 & 0x1) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_gpr (cache, 19 + 2 * x, *cur_sp + z * 8);
	aarch64_windows_save_gpr (cache, 30, *cur_sp + z * 8 + 8);
      }
      break;

    case AARCH64_UOP_SAVE_FREGP:
      {
	int x = ((b0 & 0x1) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_fpr (cache, 8 + x, *cur_sp + z * 8);
	aarch64_windows_save_fpr (cache, 9 + x, *cur_sp + z * 8 + 8);
	*next_kind = 2; *next_reg = 10 + x; *next_addr = *cur_sp + z * 8 + 16;
      }
      break;

    case AARCH64_UOP_SAVE_FREGP_X:
      {
	int x = ((b0 & 0x1) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_fpr (cache, 8 + x, *cur_sp);
	aarch64_windows_save_fpr (cache, 9 + x, *cur_sp + 8);
	*next_kind = 2; *next_reg = 10 + x; *next_addr = *cur_sp + 16;
	*cur_sp += (z + 1) * 8;
      }
      break;

    case AARCH64_UOP_SAVE_FREG:
      {
	int x = ((b0 & 0x1) << 2) | (b1 >> 6);
	int z = b1 & 0x3f;
	aarch64_windows_save_fpr (cache, 8 + x, *cur_sp + z * 8);
      }
      break;

    case AARCH64_UOP_SAVE_FREG_X:
      {
	int x = b1 >> 5;
	int z = b1 & 0x1f;
	aarch64_windows_save_fpr (cache, 8 + x, *cur_sp);
	*cur_sp += (z + 1) * 8;
      }
      break;

    case AARCH64_UOP_SAVE_NEXT:
      /* Save the next register pair, contiguous with the previous save.  */
      if (*next_kind == 1)
	{
	  aarch64_windows_save_gpr (cache, *next_reg, *next_addr);
	  aarch64_windows_save_gpr (cache, *next_reg + 1, *next_addr + 8);
	}
      else if (*next_kind == 2)
	{
	  aarch64_windows_save_fpr (cache, *next_reg, *next_addr);
	  aarch64_windows_save_fpr (cache, *next_reg + 1, *next_addr + 8);
	}
      else
	return false;
      *next_reg += 2;
      *next_addr += 16;
      break;

    case AARCH64_UOP_SET_FP:
      {
	/* mov x29,sp in the prolog: recover sp from the frame pointer.  */
	std::array<gdb_byte, 8> fpbuf;
	get_frame_register (this_frame, AARCH64_FP_REGNUM, fpbuf);
	*cur_sp = extract_unsigned_integer (fpbuf, byte_order);
      }
      break;

    case AARCH64_UOP_ADD_FP:
      {
	/* add x29,sp,#x*8: sp = x29 - x*8.  */
	std::array<gdb_byte, 8> fpbuf;
	CORE_ADDR fp_val;
	get_frame_register (this_frame, AARCH64_FP_REGNUM, fpbuf);
	fp_val = extract_unsigned_integer (fpbuf, byte_order);
	*cur_sp = fp_val - b1 * 8;
      }
      break;

    case AARCH64_UOP_NOP:
    case AARCH64_UOP_PAC:
      /* pac_sign_lr only signs lr; it does not move it.  Nothing to do for
	 an unwinder that reads lr's saved value directly.  */
      break;

    case AARCH64_UOP_END:
    case AARCH64_UOP_END_C:
      return false;

    default:
      /* Unsupported code -- stop, keeping whatever we decoded so far.  */
      return false;
    }

  return true;
}

/* Walk the unwind-code array [CODES, CODES+CODES_LEN) starting at byte
   index START, executing codes onto CACHE, until an end code or the array
   end.  SKIP_INSNS unwind codes (each is one prolog/epilog instruction)
   are skipped before execution begins -- used to model a PC partway
   through a prolog or epilog.  *CUR_SP is the running stack pointer.  */

static void
aarch64_windows_exec_codes (const frame_info_ptr &this_frame,
			    struct aarch64_windows_frame_cache *cache,
			    const gdb_byte *codes, int codes_len,
			    int start, int skip_insns, CORE_ADDR *cur_sp)
{
  int next_kind = 0, next_reg = 0;
  CORE_ADDR next_addr = 0;
  int i = start;

  while (i < codes_len)
    {
      int len;
      enum aarch64_unwind_op op = aarch64_windows_unwind_op (codes[i], &len);

      if (i + len > codes_len)
	break;

      if (skip_insns > 0)
	{
	  /* Even when skipping, end terminates the sequence.  */
	  if (op == AARCH64_UOP_END || op == AARCH64_UOP_END_C)
	    break;
	  skip_insns--;
	  i += len;
	  continue;
	}

      if (!aarch64_windows_exec_one (this_frame, cache, &codes[i], op, len,
				     cur_sp, &next_kind, &next_reg, &next_addr))
	break;

      i += len;
    }

  cache->prev_sp = *cur_sp;
}

/* Count the number of unwind codes (== prolog/epilog instructions) from
   byte index START up to but not including the first end/end_c code, in
   the array [CODES, CODES+CODES_LEN).  */

static int
aarch64_windows_count_insns (const gdb_byte *codes, int codes_len, int start)
{
  int n = 0;

  for (int i = start; i < codes_len; )
    {
      int len;
      enum aarch64_unwind_op op = aarch64_windows_unwind_op (codes[i], &len);

      if (op == AARCH64_UOP_END || op == AARCH64_UOP_END_C)
	break;
      if (i + len > codes_len)
	break;
      n++;
      i += len;
    }

  return n;
}

/* Decode a full .xdata record at XDATA_RVA and fill CACHE.  */

static void
aarch64_windows_decode_xdata (const frame_info_ptr &this_frame,
			      struct aarch64_windows_frame_cache *cache,
			      CORE_ADDR xdata_rva)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  CORE_ADDR addr = cache->image_base + xdata_rva;
  gdb_byte hdr[8];
  uint32_t w0, w1 = 0;
  int header_words = 1;

  if (target_read_memory (addr, hdr, 4) != 0)
    return;
  w0 = extract_unsigned_integer (hdr, 4, byte_order);

  unsigned func_len = (w0 & 0x3ffff) * 4;
  unsigned vers = (w0 >> 18) & 0x3;
  unsigned e_bit = (w0 >> 21) & 0x1;
  unsigned epilog_count = (w0 >> 22) & 0x1f;
  unsigned code_words = (w0 >> 27) & 0x1f;

  if (vers != 0)
    return;

  cache->end_rva = cache->start_rva + func_len;

  if (epilog_count == 0 && code_words == 0)
    {
      /* Extension word present.  */
      if (target_read_memory (addr + 4, hdr + 4, 4) != 0)
	return;
      w1 = extract_unsigned_integer (hdr + 4, 4, byte_order);
      epilog_count = w1 & 0xffff;
      code_words = (w1 >> 16) & 0xff;
      header_words = 2;
    }

  /* Offset (in bytes from XDATA start) of the epilog-scope array and of the
     unwind-code array that follows it.  When E == 1 there is no scope
     array; the single epilog's start index is packed in the epilog_count
     field.  */
  int scope_bytes = e_bit ? 0 : (int) epilog_count * 4;
  CORE_ADDR codes_addr = addr + header_words * 4 + scope_bytes;
  int codes_len = code_words * 4;

  if (codes_len <= 0 || codes_len > 255 * 4)
    return;

  gdb_byte codes[255 * 4];
  if (target_read_memory (codes_addr, codes, codes_len) != 0)
    return;

  /* Where is the PC relative to the function start?  */
  CORE_ADDR func_start = cache->image_base + cache->start_rva;
  LONGEST pc_off = (LONGEST) cache->pc - (LONGEST) func_start;
  CORE_ADDR cur_sp = cache->sp;

  /* Determine whether the PC lies in an epilog, and if so start decoding
     at that epilog's code index, skipping the instructions already run.  */
  if (e_bit)
    {
      /* Single epilog whose codes start at index EPILOG_COUNT and which is
	 located at the tail of the function.  The epilog spans the load
	 instructions plus the terminating ret (the "end" code), hence the
	 +1.  */
      int start = epilog_count;
      int epilog_insns = aarch64_windows_count_insns (codes, codes_len, start);
      CORE_ADDR epilog_off = func_len - (epilog_insns + 1) * 4;

      if (pc_off >= (LONGEST) epilog_off && pc_off < (LONGEST) func_len)
	{
	  int done = (pc_off - epilog_off) / 4;
	  aarch64_windows_exec_codes (this_frame, cache, codes, codes_len,
				      start, done, &cur_sp);
	  return;
	}
    }
  else
    {
      for (unsigned s = 0; s < epilog_count; s++)
	{
	  gdb_byte sb[4];
	  uint32_t sw;

	  if (target_read_memory (addr + header_words * 4 + s * 4, sb, 4) != 0)
	    return;
	  sw = extract_unsigned_integer (sb, 4, byte_order);

	  CORE_ADDR epilog_off = (sw & 0x3ffff) * 4;
	  int epilog_index = (sw >> 22) & 0x3ff;
	  int epilog_insns
	    = aarch64_windows_count_insns (codes, codes_len, epilog_index);
	  /* +1 to include the terminating ret (the "end" code).  */
	  CORE_ADDR epilog_end = epilog_off + (epilog_insns + 1) * 4;

	  if (pc_off >= (LONGEST) epilog_off && pc_off < (LONGEST) epilog_end)
	    {
	      int done = (pc_off - epilog_off) / 4;
	      aarch64_windows_exec_codes (this_frame, cache, codes, codes_len,
					  epilog_index, done, &cur_sp);
	      return;
	    }
	}
    }

  /* Not in an epilog: unwind using the prolog codes (index 0).  If the PC
     is partway through the prolog, skip the codes for the not-yet-executed
     (trailing) prolog instructions.  The prolog codes are stored in reverse
     execution order, so the not-yet-executed instructions correspond to the
     leading unwind codes.  */
  int prolog_insns = aarch64_windows_count_insns (codes, codes_len, 0);
  CORE_ADDR prolog_len = prolog_insns * 4;
  int skip = 0;

  if (pc_off >= 0 && pc_off < (LONGEST) prolog_len)
    skip = prolog_insns - (pc_off / 4);

  aarch64_windows_exec_codes (this_frame, cache, codes, codes_len, 0, skip,
			      &cur_sp);
}

/* One prolog instruction, reconstructed from a canonical "packed" unwind
   description.  The prolog maps 1:1 onto instructions, so this ordered list
   lets the packed path model a PC in a partially-executed prolog or epilog
   exactly as the .xdata path does, not just an established frame.  */

enum aarch64_pop_kind
{
  AARCH64_POP_SAVE,	/* store register(s), possibly pre-indexed  */
  AARCH64_POP_ALLOC,	/* plain sub sp,sp,#imm  */
  AARCH64_POP_SETFP,	/* mov x29,sp / add x29,sp,#imm  */
  AARCH64_POP_PAC,	/* pacibsp  */
  AARCH64_POP_HOME	/* home an incoming argument pair (volatile)  */
};

struct aarch64_windows_pop
{
  enum aarch64_pop_kind kind;
  CORE_ADDR alloc;	/* sp decrement performed by this instruction  */
  int nsave;
  int sreg[2];		/* x-number (0..30), or v-index for FP saves  */
  bool sfp[2];		/* true if the save is an FP (d) register  */
  CORE_ADDR soff[2];	/* offset from sp *after* this insn's own predec  */
  CORE_ADDR setfp_k;	/* for SETFP: unwound sp = x29 - setfp_k  */
};

/* Execute the ordered ops referenced by ORDER[SKIP..N) -- which are in unwind
   (reverse-prolog) order -- onto CACHE, threading the running sp through
   *CUR_SP.  Used for both the prolog/body walk and the epilog walk.  */

static void
aarch64_windows_exec_pops (const frame_info_ptr &this_frame,
			   struct aarch64_windows_frame_cache *cache,
			   const struct aarch64_windows_pop *ops,
			   const int *order, int n, int skip, CORE_ADDR *cur_sp)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);

  for (int i = skip; i < n; i++)
    {
      const struct aarch64_windows_pop *o = &ops[order[i]];

      if (o->kind == AARCH64_POP_SETFP)
	{
	  std::array<gdb_byte, 8> fpbuf;
	  get_frame_register (this_frame, AARCH64_FP_REGNUM, fpbuf);
	  *cur_sp = extract_unsigned_integer (fpbuf, byte_order) - o->setfp_k;
	  continue;
	}

      for (int s = 0; s < o->nsave; s++)
	{
	  if (o->sfp[s])
	    aarch64_windows_save_fpr (cache, o->sreg[s], *cur_sp + o->soff[s]);
	  else
	    aarch64_windows_save_gpr (cache, o->sreg[s], *cur_sp + o->soff[s]);
	}
      *cur_sp += o->alloc;
    }

  cache->prev_sp = *cur_sp;
}

/* Build the prolog op list for a canonical packed function into OPS, in
   prolog execution order, returning the number of ops.  The layout follows
   the canonical prolog algorithm in the ARM64 exception-handling spec.  */

static int
aarch64_windows_build_packed_ops (struct aarch64_windows_pop *ops,
				  unsigned reg_i, unsigned reg_f, unsigned h_bit,
				  unsigned cr, unsigned intsz, unsigned savsz,
				  unsigned locsz)
{
  int n = 0;
  /* The very first store instruction pre-decrements sp by the whole
     callee-save area; every later store is a plain offset store.  */
  CORE_ADDR predec = savsz;
  bool lr_in_step2 = false;

  auto emit_save = [&] (CORE_ADDR alloc, int r0, bool fp0, CORE_ADDR o0,
			int nsave, int r1, bool fp1, CORE_ADDR o1)
    {
      struct aarch64_windows_pop *o = &ops[n++];
      o->kind = AARCH64_POP_SAVE;
      o->alloc = alloc;
      o->nsave = nsave;
      o->sreg[0] = r0; o->sfp[0] = fp0; o->soff[0] = o0;
      o->sreg[1] = r1; o->sfp[1] = fp1; o->soff[1] = o1;
      o->setfp_k = 0;
    };

  /* Step 1: pacibsp (chained with signed return address).  */
  if (cr == 2)
    {
      struct aarch64_windows_pop *o = &ops[n++];
      o->kind = AARCH64_POP_PAC;
      o->alloc = 0; o->nsave = 0; o->setfp_k = 0;
    }

  /* Step 2: save non-volatile integer registers x19..x(19+reg_i-1).  */
  for (unsigned i = 0; i < reg_i; )
    {
      CORE_ADDR alloc = predec; predec = 0;
      CORE_ADDR off = i * 8;
      unsigned rem = reg_i - i;

      if (rem == 1 && cr == 1 && (reg_i & 1))
	{
	  /* Odd RegI with saved lr: last int reg pairs with lr.  */
	  emit_save (alloc, 19 + i, false, off, 2, 30, false, off + 8);
	  lr_in_step2 = true;
	  i += 1;
	}
      else if (rem == 1)
	{
	  emit_save (alloc, 19 + i, false, off, 1, 0, false, 0);
	  i += 1;
	}
      else
	{
	  emit_save (alloc, 19 + i, false, off, 2, 20 + i, false, off + 8);
	  i += 2;
	}
    }

  /* Step 3: save lr (unchained function that keeps lr in the int area).  */
  if (cr == 1 && !lr_in_step2)
    {
      CORE_ADDR alloc = predec; predec = 0;
      emit_save (alloc, 30, false, intsz - 8, 1, 0, false, 0);
    }

  /* Step 4: save non-volatile FP registers d8..d(8+reg_f).  */
  if (reg_f)
    {
      unsigned count = reg_f + 1;
      for (unsigned j = 0; j < count; )
	{
	  CORE_ADDR alloc = predec; predec = 0;
	  CORE_ADDR off = intsz + j * 8;
	  unsigned rem = count - j;

	  if (rem == 1)
	    {
	      emit_save (alloc, 8 + j, true, off, 1, 0, false, 0);
	      j += 1;
	    }
	  else
	    {
	      emit_save (alloc, 8 + j, true, off, 2, 9 + j, true, off + 8);
	      j += 2;
	    }
	}
    }

  /* Step 5: home the incoming integer argument registers (volatile; only
     the instruction slots and any leading predec matter to unwinding).  */
  if (h_bit)
    for (int k = 0; k < 4; k++)
      {
	struct aarch64_windows_pop *o = &ops[n++];
	o->kind = AARCH64_POP_HOME;
	o->alloc = predec; predec = 0;
	o->nsave = 0; o->setfp_k = 0;
      }

  /* Step 6: allocate the local area and, for chained functions, save and
     establish the frame chain.  */
  if (cr == 2 || cr == 3)
    {
      if (locsz <= 512)
	{
	  /* stp x29,lr,[sp,#-locsz]! ; mov x29,sp  */
	  emit_save (locsz, 29, false, 0, 2, 30, false, 8);
	  struct aarch64_windows_pop *o = &ops[n++];
	  o->kind = AARCH64_POP_SETFP; o->alloc = 0; o->nsave = 0;
	  o->setfp_k = 0;
	}
      else
	{
	  /* sub sp,sp,#locsz [split if >4080] ; stp x29,lr,[sp] ; add x29,sp */
	  if (locsz > 4080)
	    {
	      struct aarch64_windows_pop *o = &ops[n++];
	      o->kind = AARCH64_POP_ALLOC; o->alloc = 4080; o->nsave = 0;
	      o->setfp_k = 0;
	      o = &ops[n++];
	      o->kind = AARCH64_POP_ALLOC; o->alloc = locsz - 4080;
	      o->nsave = 0; o->setfp_k = 0;
	    }
	  else
	    {
	      struct aarch64_windows_pop *o = &ops[n++];
	      o->kind = AARCH64_POP_ALLOC; o->alloc = locsz; o->nsave = 0;
	      o->setfp_k = 0;
	    }
	  emit_save (0, 29, false, 0, 2, 30, false, 8);
	  struct aarch64_windows_pop *o = &ops[n++];
	  o->kind = AARCH64_POP_SETFP; o->alloc = 0; o->nsave = 0;
	  o->setfp_k = 0;
	}
    }
  else if (locsz > 0)
    {
      /* Unchained: just allocate the local area (split if >4080).  */
      if (locsz > 4080)
	{
	  struct aarch64_windows_pop *o = &ops[n++];
	  o->kind = AARCH64_POP_ALLOC; o->alloc = 4080; o->nsave = 0;
	  o->setfp_k = 0;
	  o = &ops[n++];
	  o->kind = AARCH64_POP_ALLOC; o->alloc = locsz - 4080; o->nsave = 0;
	  o->setfp_k = 0;
	}
      else
	{
	  struct aarch64_windows_pop *o = &ops[n++];
	  o->kind = AARCH64_POP_ALLOC; o->alloc = locsz; o->nsave = 0;
	  o->setfp_k = 0;
	}
    }

  return n;
}

/* Fill CACHE from a canonical "packed" unwind description PACKED (the 32-bit
   second .pdata word, whose low two bits are the non-zero Flag).  This is the
   compact form for functions with a canonical prolog/epilog; it needs no
   .xdata.  The prolog is reconstructed as an ordered op list so that a PC in
   the body, in a partially-executed prolog, or in an epilog is all handled.  */

static void
aarch64_windows_decode_packed (const frame_info_ptr &this_frame,
			       struct aarch64_windows_frame_cache *cache,
			       uint32_t packed)
{
  unsigned func_len = ((packed >> 2) & 0x7ff) * 4;
  unsigned reg_f = (packed >> 13) & 0x7;
  unsigned reg_i = (packed >> 16) & 0xf;
  unsigned h_bit = (packed >> 20) & 0x1;
  unsigned cr = (packed >> 21) & 0x3;
  unsigned frame_size = ((packed >> 23) & 0x1ff) * 16;

  cache->end_rva = cache->start_rva + func_len;

  unsigned intsz = reg_i * 8;
  if (cr == 1)			/* unchained, lr saved in int area  */
    intsz += 8;
  unsigned fpsz = reg_f ? (reg_f + 1) * 8 : 0;
  unsigned savsz = ((intsz + fpsz + 8 * 8 * h_bit) + 0xf) & ~0xfU;
  unsigned locsz = frame_size - savsz;

  /* At most: pac + 5 int + 1 lr + 4 fp + 4 home + 2 alloc + fplr + setfp.  */
  struct aarch64_windows_pop ops[32];
  int nops = aarch64_windows_build_packed_ops (ops, reg_i, reg_f, h_bit, cr,
					       intsz, savsz, locsz);

  CORE_ADDR func_start = cache->image_base + cache->start_rva;
  LONGEST pc_off = (LONGEST) cache->pc - (LONGEST) func_start;
  CORE_ADDR cur_sp = cache->sp;

  /* Prolog length in instructions == number of ops.  */
  int prolog_insns = nops;

  /* The epilog mirrors the prolog in reverse, minus the pacibsp, the frame
     setup (mov/add x29), and the argument-homing stores.  Build its op order
     (unwind order, i.e. reverse prolog order) and locate it at the tail of
     the function (its final instruction is the ret).  */
  int eorder[32];
  int en = 0;
  for (int i = nops - 1; i >= 0; i--)
    if (ops[i].kind == AARCH64_POP_SAVE || ops[i].kind == AARCH64_POP_ALLOC)
      eorder[en++] = i;

  if (en > 0)
    {
      CORE_ADDR epilog_start = func_len - (CORE_ADDR) (en + 1) * 4;

      if (pc_off >= (LONGEST) epilog_start && pc_off < (LONGEST) func_len)
	{
	  int done = (pc_off - epilog_start) / 4;
	  aarch64_windows_exec_pops (this_frame, cache, ops, eorder, en, done,
				     &cur_sp);
	  return;
	}
    }

  /* Body or (partial) prolog: unwind order is the full op list reversed.  */
  int order[32];
  for (int i = 0; i < nops; i++)
    order[i] = nops - 1 - i;

  int skip = 0;
  if (pc_off >= 0 && pc_off < (LONGEST) prolog_insns * 4)
    skip = prolog_insns - (pc_off / 4);

  aarch64_windows_exec_pops (this_frame, cache, ops, order, nops, skip,
			     &cur_sp);
}

/* Find SEH unwind info for PC.  On success return 0 and set *PACKED_OR_RVA
   to the second .pdata word, *IS_PACKED to whether it is packed unwind data
   (Flag != 0) rather than an .xdata RVA, *IMAGE_BASE and *START_RVA.  On
   failure (no covering entry) return -1.  */

static int
aarch64_windows_find_unwind_info (struct gdbarch *gdbarch, CORE_ADDR pc,
				  uint32_t *packed_or_rva, int *is_packed,
				  CORE_ADDR *image_base, CORE_ADDR *start_rva)
{
  struct obj_section *sec;
  pe_data_type *pe;
  IMAGE_DATA_DIRECTORY *dir;
  struct objfile *objfile;
  CORE_ADDR base;
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);

  sec = find_pc_section (pc);
  if (sec == NULL)
    return -1;
  objfile = sec->objfile;
  pe = pe_data (sec->objfile->obfd);
  dir = &pe->pe_opthdr.DataDirectory[PE_EXCEPTION_TABLE];

  base = pe->pe_opthdr.ImageBase + objfile->text_section_offset ();
  *image_base = base;

  /* Each ARM64 .pdata entry is 8 bytes: a function-start RVA and a second
     word.  Entries are sorted by start RVA; there is no end RVA, so binary
     search for the last entry whose start does not exceed PC.  */
  unsigned long n = dir->Size / 8;
  if (n == 0)
    return -1;

  unsigned long lo = 0, hi = n - 1;
  long found = -1;
  CORE_ADDR found_start = 0;
  uint32_t found_word = 0;

  while (lo <= hi)
    {
      unsigned long mid = lo + (hi - lo) / 2;
      gdb_byte d[8];
      CORE_ADDR sa;

      if (target_read_memory (base + dir->VirtualAddress + mid * 8, d, 8) != 0)
	return -1;
      sa = extract_unsigned_integer (d, 4, byte_order);

      if (base + sa <= pc)
	{
	  found = mid;
	  found_start = sa;
	  found_word = extract_unsigned_integer (d + 4, 4, byte_order);
	  lo = mid + 1;
	}
      else
	{
	  if (mid == 0)
	    break;
	  hi = mid - 1;
	}
    }

  if (found < 0)
    return -1;

  *start_rva = found_start;
  if ((found_word & 0x3) != 0)
    {
      *is_packed = 1;
      *packed_or_rva = found_word;
    }
  else
    {
      *is_packed = 0;
      *packed_or_rva = found_word & ~0x3U;
    }

  frame_debug_printf ("image_base=%s, start_rva=%s, %s=%s",
		      paddress (gdbarch, base),
		      paddress (gdbarch, found_start),
		      *is_packed ? "packed" : "xdata_rva",
		      paddress (gdbarch, *packed_or_rva));
  return 0;
}

/* Fill THIS_CACHE using the native aarch64-windows unwinding data.  */

static struct aarch64_windows_frame_cache *
aarch64_windows_frame_cache (const frame_info_ptr &this_frame,
			     void **this_cache)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  struct aarch64_windows_frame_cache *cache;
  std::array<gdb_byte, 8> buf;
  CORE_ADDR pc;
  uint32_t packed_or_rva = 0;
  int is_packed = 0;

  if (*this_cache)
    return (struct aarch64_windows_frame_cache *) *this_cache;

  cache = FRAME_OBSTACK_ZALLOC (struct aarch64_windows_frame_cache);
  *this_cache = cache;

  pc = get_frame_pc (this_frame);
  get_frame_register (this_frame, AARCH64_SP_REGNUM, buf);
  cache->sp = extract_unsigned_integer (buf, byte_order);
  cache->pc = pc;
  cache->prev_sp = cache->sp;

  if (aarch64_windows_find_unwind_info (gdbarch, pc, &packed_or_rva,
					&is_packed, &cache->image_base,
					&cache->start_rva) != 0)
    {
      /* No .pdata entry: a leaf function that manipulates neither the stack
	 nor any non-volatile register.  Return address is live lr; caller's
	 sp equals this sp.  */
      return cache;
    }

  if (is_packed)
    aarch64_windows_decode_packed (this_frame, cache, packed_or_rva);
  else
    aarch64_windows_decode_xdata (this_frame, cache, packed_or_rva);

  return cache;
}

/* Implement the "prev_register" method.  */

static struct value *
aarch64_windows_frame_prev_register (const frame_info_ptr &this_frame,
				     void **this_cache, int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  struct aarch64_windows_frame_cache *cache
    = aarch64_windows_frame_cache (this_frame, this_cache);
  CORE_ADDR prev = 0;

  if (regnum == AARCH64_SP_REGNUM)
    return frame_unwind_got_constant (this_frame, regnum, cache->prev_sp);

  if (regnum == AARCH64_PC_REGNUM)
    {
      /* The return address lives in lr: either where this frame saved it,
	 or lr's current live value if it was never saved.  */
      CORE_ADDR lr_addr = cache->prev_gpr_addr[AARCH64_LR_REGNUM
					       - AARCH64_X0_REGNUM];
      CORE_ADDR pc;

      if (lr_addr != 0)
	{
	  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
	  gdb_byte b[8];

	  if (target_read_memory (lr_addr, b, 8) != 0)
	    return frame_unwind_got_register (this_frame, regnum,
					      AARCH64_LR_REGNUM);
	  pc = extract_unsigned_integer (b, 8, byte_order);
	}
      else
	{
	  struct value *lr
	    = frame_unwind_got_register (this_frame, AARCH64_LR_REGNUM,
					 AARCH64_LR_REGNUM);
	  pc = value_as_long (lr);
	}

      /* Strip any pointer-authentication bits.  */
      pc = gdbarch_addr_bits_remove (gdbarch, pc);
      return frame_unwind_got_constant (this_frame, regnum, pc);
    }

  if (regnum >= AARCH64_V0_REGNUM && regnum <= AARCH64_V31_REGNUM)
    prev = cache->prev_fpr_addr[regnum - AARCH64_V0_REGNUM];
  else if (regnum >= AARCH64_X0_REGNUM && regnum <= AARCH64_LR_REGNUM)
    prev = cache->prev_gpr_addr[regnum - AARCH64_X0_REGNUM];

  if (prev != 0)
    return frame_unwind_got_memory (this_frame, regnum, prev);

  /* Volatile, or not modified by this frame.  */
  return frame_unwind_got_register (this_frame, regnum, regnum);
}

/* Implement the "this_id" method.  */

static void
aarch64_windows_frame_this_id (const frame_info_ptr &this_frame,
			       void **this_cache, struct frame_id *this_id)
{
  struct aarch64_windows_frame_cache *cache
    = aarch64_windows_frame_cache (this_frame, this_cache);

  *this_id = frame_id_build (cache->prev_sp,
			     cache->image_base + cache->start_rva);
}

/* Windows aarch64 SEH unwinder.  */

static const struct frame_unwind_legacy aarch64_windows_frame_unwind (
  "aarch64 windows",
  NORMAL_FRAME,
  FRAME_UNWIND_ARCH,
  default_frame_unwind_stop_reason,
  &aarch64_windows_frame_this_id,
  &aarch64_windows_frame_prev_register,
  NULL,
  default_frame_sniffer
);

/* Implement the "skip_prologue" gdbarch method using unwind data.  */

static CORE_ADDR
aarch64_windows_skip_prologue (struct gdbarch *gdbarch, CORE_ADDR pc)
{
  CORE_ADDR func_addr, image_base, start_rva;
  uint32_t packed_or_rva = 0;
  int is_packed = 0;
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);

  if (aarch64_windows_find_unwind_info (gdbarch, pc, &packed_or_rva,
					&is_packed, &image_base,
					&start_rva) == 0)
    {
      CORE_ADDR func_start = image_base + start_rva;
      int prolog_len = -1;

      if (!is_packed)
	{
	  /* Count prolog codes in the .xdata to get the prolog length.  */
	  gdb_byte hdr[8];
	  uint32_t w0;

	  if (target_read_memory (image_base + packed_or_rva, hdr, 4) == 0)
	    {
	      w0 = extract_unsigned_integer (hdr, 4, byte_order);
	      unsigned e_bit = (w0 >> 21) & 0x1;
	      unsigned epilog_count = (w0 >> 22) & 0x1f;
	      unsigned code_words = (w0 >> 27) & 0x1f;
	      int header_words = 1;

	      if (epilog_count == 0 && code_words == 0)
		{
		  if (target_read_memory (image_base + packed_or_rva + 4,
					  hdr + 4, 4) == 0)
		    {
		      uint32_t w1
			= extract_unsigned_integer (hdr + 4, 4, byte_order);
		      epilog_count = w1 & 0xffff;
		      code_words = (w1 >> 16) & 0xff;
		      header_words = 2;
		    }
		}

	      int scope_bytes = e_bit ? 0 : (int) epilog_count * 4;
	      CORE_ADDR codes_addr = image_base + packed_or_rva
				     + header_words * 4 + scope_bytes;
	      int codes_len = code_words * 4;

	      if (codes_len > 0 && codes_len <= 255 * 4)
		{
		  gdb_byte codes[255 * 4];

		  if (target_read_memory (codes_addr, codes, codes_len) == 0)
		    prolog_len
		      = aarch64_windows_count_insns (codes, codes_len, 0) * 4;
		}
	    }
	}

      if (prolog_len >= 0)
	return std::max (pc, func_start + prolog_len);
    }

  /* Fall back to the line table.  */
  if (find_pc_partial_function (pc, NULL, &func_addr, NULL))
    {
      CORE_ADDR post = skip_prologue_using_sal (gdbarch, func_addr);

      if (post != 0)
	return std::max (pc, post);
    }

  return pc;
}

/* gdbarch initialization for Windows on aarch64.  */

static void
aarch64_windows_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  windows_init_abi (info, gdbarch);

  /* aarch64_gdbarch_init has already appended the DWARF and generic
     prologue unwinders, the latter using default_frame_sniffer (always
     accepts).  Prepend the SEH unwinder so it is consulted first.  */
  frame_unwind_prepend_unwinder (gdbarch, &aarch64_windows_frame_unwind);

  set_gdbarch_skip_prologue (gdbarch, aarch64_windows_skip_prologue);
}

/* Identify aarch64 Windows PE/COFF objects by their BFD target name.  */

static enum gdb_osabi
aarch64_windows_osabi_sniffer (bfd *abfd)
{
  const char *target_name = bfd_get_target (abfd);

  if (!streq (target_name, "pei-aarch64-little"))
    return GDB_OSABI_UNKNOWN;

  return GDB_OSABI_WINDOWS;
}

INIT_GDB_FILE (aarch64_windows_tdep)
{
  gdbarch_register_osabi (bfd_arch_aarch64, bfd_mach_aarch64,
			  GDB_OSABI_WINDOWS, aarch64_windows_init_abi);

  gdbarch_register_osabi_sniffer (bfd_arch_aarch64, bfd_target_coff_flavour,
				  aarch64_windows_osabi_sniffer);
}
