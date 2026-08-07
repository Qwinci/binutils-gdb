/* Target-dependent code for RISC-V64 systems running Windows, for GDB.

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

/* This file implements SEH-based unwinding for riscv64 Windows PE/COFF
   binaries (target triple riscv64-*-windows-msvc / riscv64-*-windows-gnu).
   The .pdata/.xdata unwind-code format is documented in
   llvm/docs/RISCVWinCFI.md in the LLVM source tree.

   The UNWIND_INFO/RUNTIME_FUNCTION container is identical to x86_64's (see
   amd64-windows-tdep.c), so the generic external_pex64_unwind_info and
   external_pex64_runtime_function structures and the PEX64_* macros from
   coff/pe.h are reused as-is.  Only the meaning of the opcodes inside the
   unwind-code array, and the register-index tables, differ.  Like x86_64
   (and unlike ARM64), riscv64 has no unwind codes describing the epilogue,
   so the unwinder pattern-matches the epilogue's instruction bytes.  */

#include "event-top.h"
#include "extract-store-integer.h"
#include "osabi.h"
#include "riscv-tdep.h"
#include "gdbcore.h"
#include "windows-tdep.h"
#include "frame.h"
#include "frame-unwind.h"
#include "objfiles.h"
#include "coff/internal.h"
#include "coff/riscv64.h"
#include "coff/pe.h"
#include "libcoff.h"
#include "opcode/riscv.h"
#include <algorithm>

/* On-disk unwind opcodes.  Values 1-4 are numerically identical to, and
   share the meaning of, x86_64's UWOP_ALLOC_LARGE/UWOP_ALLOC_SMALL/
   UWOP_SET_FPREG/UWOP_SAVE_NONVOL (see coff/pe.h) so those macros are reused
   directly.  Value 5 is FPR saves: D-extension registers are 8 bytes, not
   the 16 bytes x86_64's UWOP_SAVE_XMM128 (also value 5, but reused for a
   different purpose here) assumes.  */
#define RISCV64_UWOP_SAVE_FREG 5

/* Operand-less opcodes describing OS-runtime dispatcher stubs
   (.seh_trap_frame / .seh_context / .seh_clear_unwound_to_call).  They only
   appear in hand-written kernel/dispatcher code, never in compiler output,
   and describe how a full register state was restored from an OS-synthesized
   trap frame or CONTEXT record on the stack.  Each is a single 2-byte slot
   with no register/offset operand.  A backtrace unwinder does not need to act
   on them -- in the stubs that use them they are paired with the ordinary
   save opcodes that already record where each register lives -- but it must
   still recognise and step over them rather than treat them as malformed.  */
#define RISCV64_UWOP_TRAP_FRAME		6
#define RISCV64_UWOP_CONTEXT		7
#define RISCV64_UWOP_CLEAR_UNWOUND_TO_CALL 8

/* Maps the 4-bit "GPR index" used by UWOP_SAVE_NONVOL/UWOP_SET_FPREG to the
   raw RISC-V integer register number.  The register number is directly
   usable as a GDB regnum, since GDB numbers x0-x31 as regnum 0-31.  */
static const int riscv_windows_seh_gpr[] =
{
  1,   /* ra  = x1  */
  8,   /* s0  = x8  */
  9,   /* s1  = x9  */
  18,  /* s2  = x18 */
  19,  /* s3  = x19 */
  20,  /* s4  = x20 */
  21,  /* s5  = x21 */
  22,  /* s6  = x22 */
  23,  /* s7  = x23 */
  24,  /* s8  = x24 */
  25,  /* s9  = x25 */
  26,  /* s10 = x26 */
  27,  /* s11 = x27 */
};

/* Maps the 4-bit "FPR index" used by RISCV64_UWOP_SAVE_FREG to the raw
   RISC-V floating-point register number (add RISCV_FIRST_FP_REGNUM to get a
   GDB regnum).  */
static const int riscv_windows_seh_fpr[] =
{
  8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
};

/* Return true if REGNUM (a raw x-register number) is one of the GPRs this
   unwind format can ever save (ra, s0-s11).  Used to make the epilogue
   instruction matcher conservative: an ordinary "ld reg,off(sp)" in the
   middle of a function body must not be mistaken for an epilogue reload.  */

static bool
riscv_windows_is_seh_gpr (int regnum)
{
  return (regnum == RISCV_RA_REGNUM || regnum == RISCV_FP_REGNUM
	  || regnum == 9 || (regnum >= 18 && regnum <= 27));
}

/* Likewise for FPRs (fs0-fs11), REGNUM here being a raw f-register
   number (0-31, i.e. GDB regnum minus RISCV_FIRST_FP_REGNUM).  */

static bool
riscv_windows_is_seh_fpr (int regnum)
{
  return regnum == 8 || regnum == 9 || (regnum >= 18 && regnum <= 27);
}

/* Cached information about a riscv64 Windows SEH frame.  */

struct riscv_windows_frame_cache
{
  /* ImageBase for the module.  */
  CORE_ADDR image_base;

  /* Function start and end rva.  */
  CORE_ADDR start_rva;
  CORE_ADDR end_rva;

  /* Next instruction to be executed.  */
  CORE_ADDR pc;

  /* Current sp.  */
  CORE_ADDR sp;

  /* Address at which each raw x-register (indexed by GDB regnum, i.e.
     raw register number 0-31) was saved, or 0 if it was not saved by
     this frame (in which case its previous value is the same as its
     current, live value).  */
  CORE_ADDR prev_gpr_addr[RISCV_NUM_INTEGER_REGS];

  /* Likewise for the raw f-registers (indexed by raw register number
     0-31, i.e. GDB regnum minus RISCV_FIRST_FP_REGNUM).  */
  CORE_ADDR prev_fpr_addr[32];

  /* Address of the previous frame (the CFA, i.e. the value SP had on
     entry to this function).  Unlike x86_64, riscv64's "call" does not
     implicitly push a return address, so this is not adjusted by any
     fixed constant to account for one.  */
  CORE_ADDR prev_sp;
};

/* Try to recognize and decode a riscv64 SEH epilogue sequence starting at
   CACHE->pc.  Since riscv64 has no unwind codes describing the epilogue,
   this is done by pattern-matching the actual instruction bytes.

   Return -1 if we fail to read memory for any reason.  Return 1 if an
   epilogue sequence was recognized and CACHE was filled in.  Return 0 if
   the PC is not the start of a recognized epilogue shape (the caller
   should fall back to the ordinary UNWIND_INFO-code-driven walk).  */

static int
riscv_windows_frame_decode_epilogue (const frame_info_ptr &this_frame,
				     struct riscv_windows_frame_cache *cache)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  CORE_ADDR pc = cache->pc;
  CORE_ADDR cur_sp = cache->sp;
  CORE_ADDR func_start = cache->image_base + cache->start_rva;
  CORE_ADDR func_end = cache->image_base + cache->end_rva;
  bool first = true;

  while (1)
    {
      gdb_byte buf[4];
      ULONGEST insn;
      int len;

      QUIT;

      if (pc >= func_end)
	return 0;

      if (target_read_memory (pc, buf, 2) != 0)
	return -1;
      insn = extract_unsigned_integer (buf, 2, BFD_ENDIAN_LITTLE);
      len = riscv_insn_length (insn);
      if (len != 2 && len != 4)
	return 0;
      if (len == 4)
	{
	  if (target_read_memory (pc + 2, buf + 2, 2) != 0)
	    return -1;
	  insn = extract_unsigned_integer (buf, 4, BFD_ENDIAN_LITTLE);
	}

      int rd = (insn >> 7) & 0x1f;
      int rs1 = (insn >> 15) & 0x1f;

      /* Step 1: "restore SP from FP", only legal as the very first
	 instruction of the epilogue.  addi sp, s0, -N.  */
      if (first && len == 4 && (insn & MASK_ADDI) == MATCH_ADDI
	  && rd == RISCV_SP_REGNUM && rs1 == RISCV_FP_REGNUM)
	{
	  std::array<gdb_byte, 8> fpbuf;
	  CORE_ADDR fp_val;
	  LONGEST imm = EXTRACT_ITYPE_IMM (insn);

	  get_frame_register (this_frame, RISCV_FP_REGNUM, fpbuf);
	  fp_val = extract_unsigned_integer (fpbuf, byte_order);
	  cur_sp = fp_val + imm;
	  pc += len;
	  first = false;
	  continue;
	}
      first = false;

      /* Step 2: GPR/FPR reloads.  ld/c.ldsp restoring {ra,s0-s11}, or
	 fld/c.fldsp restoring {fs0-fs11}, from sp.  */
      if (len == 4 && (insn & MASK_LD) == MATCH_LD && rs1 == RISCV_SP_REGNUM
	  && riscv_windows_is_seh_gpr (rd))
	{
	  LONGEST imm = EXTRACT_ITYPE_IMM (insn);
	  cache->prev_gpr_addr[rd] = cur_sp + imm;
	  pc += len;
	  continue;
	}
      if (len == 4 && (insn & MASK_FLD) == MATCH_FLD && rs1 == RISCV_SP_REGNUM
	  && riscv_windows_is_seh_fpr (rd))
	{
	  LONGEST imm = EXTRACT_ITYPE_IMM (insn);
	  cache->prev_fpr_addr[rd] = cur_sp + imm;
	  pc += len;
	  continue;
	}
      if (len == 2 && (insn & MASK_C_LDSP) == MATCH_C_LDSP
	  && riscv_windows_is_seh_gpr (rd))
	{
	  LONGEST imm = EXTRACT_CITYPE_LDSP_IMM (insn);
	  cache->prev_gpr_addr[rd] = cur_sp + imm;
	  pc += len;
	  continue;
	}
      if (len == 2 && (insn & MASK_C_FLDSP) == MATCH_C_FLDSP
	  && riscv_windows_is_seh_fpr (rd))
	{
	  LONGEST imm = EXTRACT_CITYPE_LDSP_IMM (insn);
	  cache->prev_fpr_addr[rd] = cur_sp + imm;
	  pc += len;
	  continue;
	}

      /* Step 3: stack deallocation(s).  addi sp, sp, +N, possibly split
	 into two consecutive instructions because of the 12-bit
	 immediate limit, or c.addi16sp +N.  */
      if (len == 4 && (insn & MASK_ADDI) == MATCH_ADDI
	  && rd == RISCV_SP_REGNUM && rs1 == RISCV_SP_REGNUM)
	{
	  LONGEST imm = EXTRACT_ITYPE_IMM (insn);
	  if (imm > 0)
	    {
	      cur_sp += imm;
	      pc += len;
	      continue;
	    }
	  return 0;
	}
      if (len == 2 && (insn & MASK_C_ADDI16SP) == MATCH_C_ADDI16SP)
	{
	  LONGEST imm = EXTRACT_CITYPE_ADDI16SP_IMM (insn);
	  if (imm > 0)
	    {
	      cur_sp += imm;
	      pc += len;
	      continue;
	    }
	  return 0;
	}

      /* Step 4: the terminating construct.  By this point the frame
	 teardown (steps 1-3) has fully run, so CUR_SP is already the CFA
	 whichever terminator form appears; recognising it just records
	 CUR_SP.  The terminator is a return or a tail call (riscv64 emits
	 tail calls routinely, at -O1 and above):

	   - Return: "ret" (jalr x0, 0(ra)) or "jr ra" (c.jr ra).
	   - Indirect tail call: the same encoding with a link register
	     other than ra -- jalr x0, 0(rN) / c.jr rN.  It differs from a
	     return only in the source register, and both end the function.
	   - Direct tail call: auipc rT, %pcrel_hi(t) followed by
	     jalr x0, rT, %pcrel_lo(t).
	   - Relaxed direct tail call: "j t" (jal x0, t) or its compressed
	     c.j form; only appears when call relaxation is enabled.

	 "j"/"c.j" is also the ordinary encoding for an in-function branch,
	 so -- exactly as amd64-windows-tdep.c does for jmp rel8/rel32 -- a
	 direct jump whose target lies inside this function is body control
	 flow, not an epilogue marker.  An indirect register jump has no
	 statically known target and is accepted as a terminator regardless
	 (matching amd64's treatment of "jmp reg").  */
      if (len == 4 && (insn & MASK_JALR) == MATCH_JALR
	  && rd == RISCV_ZERO_REGNUM && EXTRACT_ITYPE_IMM (insn) == 0)
	{
	  /* ret / jr ra / indirect tail call (jalr x0, 0(rN)).  */
	  cache->prev_sp = cur_sp;
	  return 1;
	}
      if (len == 2 && (insn & MASK_C_JR) == MATCH_C_JR
	  && rd != RISCV_ZERO_REGNUM)
	{
	  /* c.jr ra (ret) / c.jr rN (indirect tail call).  */
	  cache->prev_sp = cur_sp;
	  return 1;
	}
      if (len == 4 && (insn & MASK_AUIPC) == MATCH_AUIPC
	  && rd != RISCV_ZERO_REGNUM)
	{
	  /* Direct tail call: auipc rT must be immediately followed by
	     jalr x0, rT, imm.  A lone auipc is not epilogue-shaped.  */
	  gdb_byte nbuf[4];
	  ULONGEST next;

	  if (target_read_memory (pc + 4, nbuf, 4) != 0)
	    return -1;
	  next = extract_unsigned_integer (nbuf, 4, BFD_ENDIAN_LITTLE);
	  if ((next & MASK_JALR) == MATCH_JALR
	      && ((next >> 7) & 0x1f) == RISCV_ZERO_REGNUM
	      && ((next >> 15) & 0x1f) == rd)
	    {
	      cache->prev_sp = cur_sp;
	      return 1;
	    }
	  return 0;
	}
      if (len == 4 && (insn & MASK_JAL) == MATCH_JAL
	  && rd == RISCV_ZERO_REGNUM)
	{
	  /* "j t": a relaxed direct tail call only if it leaves the
	     function; an in-function target is an ordinary branch.  */
	  CORE_ADDR target = pc + EXTRACT_JTYPE_IMM (insn);

	  if (target >= func_start && target < func_end)
	    return 0;
	  cache->prev_sp = cur_sp;
	  return 1;
	}
      if (len == 2 && (insn & MASK_C_J) == MATCH_C_J)
	{
	  CORE_ADDR target = pc + EXTRACT_CJTYPE_IMM (insn);

	  if (target >= func_start && target < func_end)
	    return 0;
	  cache->prev_sp = cur_sp;
	  return 1;
	}

      /* Not a recognized epilogue instruction; PC is presumably still in
	 the function body, use the ordinary UNWIND_INFO-driven walk.  */
      return 0;
    }
}

/* Decode and execute unwind insns at UNWIND_INFO.  */

static void
riscv_windows_frame_decode_insns (const frame_info_ptr &this_frame,
				  struct riscv_windows_frame_cache *cache,
				  CORE_ADDR unwind_info)
{
  CORE_ADDR cur_sp = cache->sp;
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  int first = 1;

  while (1)
    {
      struct external_pex64_unwind_info ex_ui;
      /* There are at most 256 16-bit unwind insns.  */
      gdb_byte insns[2 * 256];
      gdb_byte *p;
      gdb_byte *end_insns;
      unsigned char codes_count;
      CORE_ADDR start;

      if (target_read_memory (cache->image_base + unwind_info,
			      (gdb_byte *) &ex_ui, sizeof (ex_ui)) != 0)
	return;

      frame_debug_printf ("%s: ver: %02x, plgsz: %02x, cnt: %02x",
			  paddress (gdbarch, unwind_info),
			  ex_ui.Version_Flags, ex_ui.SizeOfPrologue,
			  ex_ui.CountOfCodes);

      /* riscv64 always uses version 1; version 2's UWOP_EPILOG scheme is
	 an x86_64-only producer optimization never used on this target.  */
      if (PEX64_UWI_VERSION (ex_ui.Version_Flags) != 1)
	return;

      start = cache->image_base + cache->start_rva;
      if (first
	  && !(cache->pc >= start && cache->pc < start + ex_ui.SizeOfPrologue))
	{
	  /* We want to detect if the PC points to an epilogue.  This needs
	     to be checked only once, and an epilogue can be anywhere but in
	     the prologue.  */
	  int r = riscv_windows_frame_decode_epilogue (this_frame, cache);

	  if (r == 1)
	    return;

	  /* Not in an epilogue.  Clear possible side effects.  */
	  memset (cache->prev_gpr_addr, 0, sizeof (cache->prev_gpr_addr));
	  memset (cache->prev_fpr_addr, 0, sizeof (cache->prev_fpr_addr));
	}

      codes_count = ex_ui.CountOfCodes;

      /* Read opcodes.  */
      if (codes_count != 0
	  && target_read_memory (cache->image_base + unwind_info
				 + sizeof (ex_ui),
				 insns, codes_count * 2) != 0)
	return;

      end_insns = &insns[codes_count * 2];
      p = insns;

      for (; p < end_insns; )
	{
	  int op = PEX64_UNWCODE_CODE (p[1]);
	  int info = PEX64_UNWCODE_INFO (p[1]);

	  /* Virtually execute the operation if the pc is after the
	     corresponding instruction (that matters in case of a break
	     within the prologue).  Note that for chained info (!first),
	     the prologue has been fully executed.  */
	  if (cache->pc >= start + p[0] || cache->pc < start)
	    {
	      frame_debug_printf ("   op #%u: off=0x%02x, insn=0x%02x",
				  (unsigned) (p - insns), p[0], p[1]);

	      switch (op)
		{
		case UWOP_ALLOC_LARGE:
		  if (info == 0)
		    cur_sp
		      += 8 * extract_unsigned_integer (p + 2, 2, byte_order);
		  else if (info == 1)
		    cur_sp += extract_unsigned_integer (p + 2, 4, byte_order);
		  else
		    return;
		  break;
		case UWOP_ALLOC_SMALL:
		  cur_sp += 8 + 8 * info;
		  break;
		case UWOP_SET_FPREG:
		  {
		    /* The frame pointer, if any, is always s0: there is no
		       register sub-field on this target, so the opcode's
		       OpInfo is unused and must not be used to look up a
		       register.  FrameRegisterOffset is not split into a
		       register nibble and an offset nibble either: the entire
		       byte is the 16-scaled offset (range [0, 4080]), only
		       meaningful when this opcode is present.  */
		    std::array<gdb_byte, 8> fpbuf;
		    CORE_ADDR fp_val;

		    get_frame_register (this_frame, RISCV_FP_REGNUM, fpbuf);
		    fp_val = extract_unsigned_integer (fpbuf, byte_order);
		    cur_sp = fp_val - ex_ui.FrameRegisterOffset * 16;
		  }
		  break;
		case UWOP_SAVE_NONVOL:
		  if (info >= 0
		      && info < (int) ARRAY_SIZE (riscv_windows_seh_gpr))
		    {
		      int reg = riscv_windows_seh_gpr[info];

		      /* riscv64 GPR/FPR saves are always relative to SP at
			 the point of the save, never to the frame register,
			 regardless of whether one has been established.  */
		      cache->prev_gpr_addr[reg] = cur_sp
			+ 8 * extract_unsigned_integer (p + 2, 2, byte_order);
		    }
		  else
		    return;
		  break;
		case RISCV64_UWOP_SAVE_FREG:
		  if (info >= 0
		      && info < (int) ARRAY_SIZE (riscv_windows_seh_fpr))
		    {
		      int reg = riscv_windows_seh_fpr[info];

		      cache->prev_fpr_addr[reg] = cur_sp
			+ 8 * extract_unsigned_integer (p + 2, 2, byte_order);
		    }
		  else
		    return;
		  break;
		case RISCV64_UWOP_TRAP_FRAME:
		case RISCV64_UWOP_CONTEXT:
		case RISCV64_UWOP_CLEAR_UNWOUND_TO_CALL:
		  /* Dispatcher-stub opcodes.  In the stubs that emit these the
		     register state is also described by ordinary save opcodes,
		     so for backtrace purposes there is nothing to do here but
		     step over the slot.  */
		  break;
		default:
		  return;
		}
	    }

	  /* Adjust with the length of the opcode.  */
	  switch (op)
	    {
	    case UWOP_ALLOC_SMALL:
	    case UWOP_SET_FPREG:
	    case RISCV64_UWOP_TRAP_FRAME:
	    case RISCV64_UWOP_CONTEXT:
	    case RISCV64_UWOP_CLEAR_UNWOUND_TO_CALL:
	      p += 2;
	      break;
	    case UWOP_ALLOC_LARGE:
	      if (info == 0)
		p += 4;
	      else if (info == 1)
		p += 6;
	      else
		return;
	      break;
	    case UWOP_SAVE_NONVOL:
	    case RISCV64_UWOP_SAVE_FREG:
	      p += 4;
	      break;
	    default:
	      return;
	    }
	}

      if (PEX64_UWI_FLAGS (ex_ui.Version_Flags) != UNW_FLAG_CHAININFO)
	{
	  /* End of unwind info.  */
	  break;
	}
      else
	{
	  /* Read the chained unwind info.  The current encoder never emits
	     this (each function/funclet gets its own RUNTIME_FUNCTION/
	     UNWIND_INFO pair), but the container format allows it and a
	     decoder should not assume it can never appear.  */
	  struct external_pex64_runtime_function d;
	  CORE_ADDR chain_vma;

	  first = 0;

	  chain_vma = cache->image_base + unwind_info
	    + sizeof (ex_ui) + ((codes_count + 1) & ~1) * 2;

	  if (target_read_memory (chain_vma, (gdb_byte *) &d, sizeof (d)) != 0)
	    return;

	  cache->start_rva
	    = extract_unsigned_integer (d.rva_BeginAddress, 4, byte_order);
	  cache->end_rva
	    = extract_unsigned_integer (d.rva_EndAddress, 4, byte_order);
	  unwind_info
	    = extract_unsigned_integer (d.rva_UnwindData, 4, byte_order);
	}

      QUIT;
    }

  /* Unlike x86_64, riscv64's call sequence does not implicitly push a
     return address onto the stack (it is passed in the ra register), so
     there is no "+8" (or similar) adjustment to make here: CUR_SP is
     already the CFA.  */
  cache->prev_sp = cur_sp;

  frame_debug_printf ("   prev_sp: %s", paddress (gdbarch, cache->prev_sp));
}

/* Find SEH unwind info for PC, returning 0 on success.

   UNWIND_INFO is set to the rva of unwind info address, IMAGE_BASE
   to the base address of the corresponding image, and START_RVA/END_RVA
   to the rva range of the function containing PC.  */

static int
riscv_windows_find_unwind_info (struct gdbarch *gdbarch, CORE_ADDR pc,
				CORE_ADDR *unwind_info,
				CORE_ADDR *image_base,
				CORE_ADDR *start_rva,
				CORE_ADDR *end_rva)
{
  struct obj_section *sec;
  pe_data_type *pe;
  IMAGE_DATA_DIRECTORY *dir;
  struct objfile *objfile;
  unsigned long lo, hi;
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

  /* Binary search over .pdata, which the linker sorts by BeginAddress.
     This does not handle dynamically added entries (JIT engines), same
     caveat as amd64-windows-tdep.c.  */
  lo = 0;
  hi = dir->Size / sizeof (struct external_pex64_runtime_function);
  *unwind_info = 0;
  while (lo <= hi)
    {
      unsigned long mid = lo + (hi - lo) / 2;
      struct external_pex64_runtime_function d;
      CORE_ADDR sa, ea;

      if (target_read_memory (base + dir->VirtualAddress + mid * sizeof (d),
			      (gdb_byte *) &d, sizeof (d)) != 0)
	return -1;

      sa = extract_unsigned_integer (d.rva_BeginAddress, 4, byte_order);
      ea = extract_unsigned_integer (d.rva_EndAddress, 4, byte_order);
      if (pc < base + sa)
	hi = mid - 1;
      else if (pc >= base + ea)
	lo = mid + 1;
      else if (pc >= base + sa && pc < base + ea)
	{
	  *start_rva = sa;
	  *end_rva = ea;
	  *unwind_info =
	    extract_unsigned_integer (d.rva_UnwindData, 4, byte_order);
	  break;
	}
      else
	break;
    }

  frame_debug_printf ("image_base=%s, unwind_data=%s",
		      paddress (gdbarch, base),
		      paddress (gdbarch, *unwind_info));

  return 0;
}

/* Fill THIS_CACHE using the native riscv64-windows unwinding data
   for THIS_FRAME.  */

static struct riscv_windows_frame_cache *
riscv_windows_frame_cache (const frame_info_ptr &this_frame, void **this_cache)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  struct riscv_windows_frame_cache *cache;
  std::array<gdb_byte, 8> buf;
  CORE_ADDR pc;
  CORE_ADDR unwind_info = 0;

  if (*this_cache)
    return (struct riscv_windows_frame_cache *) *this_cache;

  cache = FRAME_OBSTACK_ZALLOC (struct riscv_windows_frame_cache);
  *this_cache = cache;

  pc = get_frame_pc (this_frame);
  get_frame_register (this_frame, RISCV_SP_REGNUM, buf);
  cache->sp = extract_unsigned_integer (buf, byte_order);
  cache->pc = pc;

  if (riscv_windows_find_unwind_info (gdbarch, pc, &unwind_info,
				      &cache->image_base,
				      &cache->start_rva,
				      &cache->end_rva)
      || unwind_info == 0)
    {
      /* No unwind info: assume a true leaf function, i.e. one that
	 changed nothing that needs undoing.  Unlike x86_64, there is no
	 implicit return-address push to account for.  */
      cache->prev_sp = cache->sp;
    }
  else
    riscv_windows_frame_decode_insns (this_frame, cache, unwind_info);

  return cache;
}

/* Implement the "prev_register" method of struct frame_unwind using the
   riscv64 Windows SEH info.  */

static struct value *
riscv_windows_frame_prev_register (const frame_info_ptr &this_frame,
				   void **this_cache, int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  struct riscv_windows_frame_cache *cache
    = riscv_windows_frame_cache (this_frame, this_cache);
  CORE_ADDR prev = 0;

  frame_debug_printf ("%s for sp=%s",
		      gdbarch_register_name (gdbarch, regnum),
		      paddress (gdbarch, cache->prev_sp));

  if (regnum == RISCV_SP_REGNUM)
    return frame_unwind_got_constant (this_frame, regnum, cache->prev_sp);

  if (regnum == gdbarch_pc_regnum (gdbarch))
    {
      /* riscv64 holds the return address in ra, not on the stack, so the
	 previous PC is defined to be whatever this frame's own ra
	 resolves to: either wherever ra was saved (if this function's
	 prologue saved it), or ra's current live value (if it never
	 touched ra at all).  This mirrors how riscv_frame_cache (the
	 generic, non-Windows unwinder) derives PC from RA in
	 riscv-tdep.c.  */
      prev = cache->prev_gpr_addr[RISCV_RA_REGNUM];
      if (prev == 0)
	return frame_unwind_got_register (this_frame, regnum,
					  RISCV_RA_REGNUM);
      return frame_unwind_got_memory (this_frame, regnum, prev);
    }

  if (regnum >= RISCV_FIRST_FP_REGNUM && regnum <= RISCV_LAST_FP_REGNUM)
    prev = cache->prev_fpr_addr[regnum - RISCV_FIRST_FP_REGNUM];
  else if (regnum >= 0 && regnum < RISCV_NUM_INTEGER_REGS)
    prev = cache->prev_gpr_addr[regnum];

  if (prev != 0)
    {
      frame_debug_printf ("  -> at %s", paddress (gdbarch, prev));
      return frame_unwind_got_memory (this_frame, regnum, prev);
    }

  /* Register is either volatile, or not modified by this frame.  */
  return frame_unwind_got_register (this_frame, regnum, regnum);
}

/* Implement the "this_id" method of struct frame_unwind using the
   riscv64 Windows SEH info.  */

static void
riscv_windows_frame_this_id (const frame_info_ptr &this_frame,
			     void **this_cache, struct frame_id *this_id)
{
  struct riscv_windows_frame_cache *cache
    = riscv_windows_frame_cache (this_frame, this_cache);

  *this_id = frame_id_build (cache->prev_sp,
			     cache->image_base + cache->start_rva);
}

/* Windows riscv64 SEH unwinder.  */

static const struct frame_unwind_legacy riscv_windows_frame_unwind (
  "riscv64 windows",
  NORMAL_FRAME,
  FRAME_UNWIND_ARCH,
  default_frame_unwind_stop_reason,
  &riscv_windows_frame_this_id,
  &riscv_windows_frame_prev_register,
  NULL,
  default_frame_sniffer
);

/* Implement the "skip_prologue" gdbarch method.  */

static CORE_ADDR
riscv_windows_skip_prologue (struct gdbarch *gdbarch, CORE_ADDR pc)
{
  CORE_ADDR func_addr;
  CORE_ADDR unwind_info = 0;
  CORE_ADDR image_base, start_rva, end_rva;
  struct external_pex64_unwind_info ex_ui;

  /* Use the prologue size from unwind info.  */
  if (riscv_windows_find_unwind_info (gdbarch, pc, &unwind_info, &image_base,
				      &start_rva, &end_rva) == 0)
    {
      if (unwind_info == 0)
	{
	  /* Leaf function.  */
	  return pc;
	}
      else if (target_read_memory (image_base + unwind_info,
				   (gdb_byte *) &ex_ui, sizeof (ex_ui)) == 0
	       && PEX64_UWI_VERSION (ex_ui.Version_Flags) == 1)
	return std::max (pc, image_base + start_rva + ex_ui.SizeOfPrologue);
    }

  /* See if we can determine the end of the prologue via the symbol
     table.  If so, then return either the PC, or the PC after
     the prologue, whichever is greater.  */
  if (find_pc_partial_function (pc, NULL, &func_addr, NULL))
    {
      CORE_ADDR post_prologue_pc
	= skip_prologue_using_sal (gdbarch, func_addr);

      if (post_prologue_pc != 0)
	return std::max (pc, post_prologue_pc);
    }

  return pc;
}

/* gdbarch initialization for Windows on riscv64.  */

static void
riscv_windows_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  windows_init_abi (info, gdbarch);

  /* riscv_gdbarch_init (riscv-tdep.c) has already appended the DWARF and
     generic prologue-analyzing unwinders by the time this osabi hook
     runs, and the latter uses default_frame_sniffer (always accepts the
     frame), so anything merely appended here would never be reached.
     Prepending -- the same mechanism tramp_frame_prepend_unwinder uses
     for OS signal-trampoline unwinders -- puts the SEH unwinder ahead of
     both.  */
  frame_unwind_prepend_unwinder (gdbarch, &riscv_windows_frame_unwind);

  set_gdbarch_skip_prologue (gdbarch, riscv_windows_skip_prologue);
}

/* Implement the "coff_flavour" osabi sniffer: identify riscv64 Windows
   PE/COFF objects by their BFD target name.  */

static enum gdb_osabi
riscv_windows_osabi_sniffer (bfd *abfd)
{
  const char *target_name = bfd_get_target (abfd);

  if (!streq (target_name, "pei-riscv64-little"))
    return GDB_OSABI_UNKNOWN;

  return GDB_OSABI_WINDOWS;
}

INIT_GDB_FILE (riscv_windows_tdep)
{
  gdbarch_register_osabi (bfd_arch_riscv, bfd_mach_riscv64, GDB_OSABI_WINDOWS,
			  riscv_windows_init_abi);

  gdbarch_register_osabi_sniffer (bfd_arch_riscv, bfd_target_coff_flavour,
				  riscv_windows_osabi_sniffer);
}
