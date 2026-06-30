/*
 * gate_bridge.cpp - Wraps break6502 (M6502Core) for single-instruction execution.
 *
 * CRITICAL DESIGN: The M6502 core is created ONCE in gate_init() and reused
 * for all subsequent calls via hardware reset (/RES pin assertion).
 *
 * Why: The M6502 constructor allocates ~273 MB for the PLA lookup table
 * (Decoder6502.bin).  Creating/destroying this per call causes heap
 * corruption and SIGSEGV after a few hundred calls.
 *
 * Strategy per gate_execute_instruction() call:
 *   1. Zero 64 KB RAM
 *   2. Write preamble + test instruction + postamble at $0200
 *   3. Assert /RES for 20 half-cycles (hardware reset)
 *   4. Release /RES, run 20 more half-cycles (reset sequence completes)
 *   5. Run until postamble halt loop detected
 *   6. Read results from zero-page $F0-$F4
 *
 * Preamble (sets all registers via real 6502 instructions):
 *   LDX #s_in ; TXS ; LDA #p_in ; PHA ; LDA #a_in ; LDX #x_in ; LDY #y_in ; PLP
 *
 * PLP is LAST so nothing clobbers P before the test instruction.
 * Stack page: all zeros except mem[$100+s_in] = p_in.
 * PLA/PLP test instructions pop 0x00 (matches ISA bridge).
 */

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "BaseLogic.h"
#include "core.h"
#include "SRAM.h"

using namespace BaseLogic;

// Persistent state (created once, reused across all calls)

static M6502Core::M6502 *core = nullptr;
static BaseBoard::SRAM  *ram  = nullptr;

// Per-step simulation variables
static uint16_t ab;
static uint8_t  db;
static TriState RnW_s;

// Half-cycle step (mirrors break6502/main.cpp Step())

static void do_half_step(TriState n_NMI, TriState n_IRQ, TriState n_RES,
                         TriState RDY,   TriState SO,    TriState CLK)
{
    bool dz = false;
    uint32_t ram_addr;

    // Memory read (from previous half-cycle's address + RnW)
    if (RnW_s == TriState::One) {
        ram_addr = ab;
        ram->sim(TriState::Zero, RnW_s, NOT(RnW_s), &ram_addr, &db, dz);
    }

    // Core simulation
    TriState inputs[(size_t)M6502Core::InputPad::Max]{};
    TriState outputs[(size_t)M6502Core::OutputPad::Max];

    inputs[(size_t)M6502Core::InputPad::n_NMI] = n_NMI;
    inputs[(size_t)M6502Core::InputPad::n_IRQ] = n_IRQ;
    inputs[(size_t)M6502Core::InputPad::n_RES] = n_RES;
    inputs[(size_t)M6502Core::InputPad::PHI0]  = CLK;
    inputs[(size_t)M6502Core::InputPad::RDY]   = RDY;
    inputs[(size_t)M6502Core::InputPad::SO]    = SO;

    core->sim(inputs, outputs, &ab, &db);

    RnW_s = outputs[(size_t)M6502Core::OutputPad::RnW];

    // Memory write
    if (RnW_s == TriState::Zero) {
        ram_addr = ab;
        ram->sim(TriState::Zero, RnW_s, NOT(RnW_s), &ram_addr, &db, dz);
    }
}

// Public API

extern "C" {

void gate_init(void)
{
    if (core != nullptr)
        return;   // idempotent, safe to call multiple times

    fprintf(stderr, "gate_init: creating M6502 core (one-time ~273 MB PLA)...\n");
    ram  = new BaseBoard::SRAM("RAM", 16, false);    // 2^16 = 64 KB
    core = new M6502Core::M6502(true, false);         // HLE=true, BCD_Hack=false
    fprintf(stderr, "gate_init: core ready.\n");

    /*
     * Initial power-on: internal latches start at zero, which is the
     * 6502's power-on state.  Run a dummy reset sequence so the core's
     * pipeline latches settle into a valid fetch state before the first
     * real test.
     */
    for (int i = 0; i < 65536; i++)
        ram->Dbg_WriteByte(i, 0xEA);   // fill with NOPs
    ram->Dbg_WriteByte(0xFFFC, 0x00);
    ram->Dbg_WriteByte(0xFFFD, 0x04);  // reset vector -> $0400 (NOP sled)

    ab    = 0;
    db    = 0;
    RnW_s = TriState::One;
    TriState clk = TriState::One;

    // Run 40 half-cycles to complete the power-on reset sequence
    for (int i = 0; i < 40; i++) {
        do_half_step(TriState::One, TriState::One, TriState::One,
                     TriState::One, TriState::Zero, clk);
        clk = NOT(clk);
    }
}

void gate_poke(uint16_t addr, uint8_t val)
{
    if (ram) ram->Dbg_WriteByte(addr, val);
}

void gate_execute_instruction(
    uint8_t opcode, uint8_t op1, uint8_t op2, int ilen,
    uint8_t a_in, uint8_t x_in, uint8_t y_in, uint8_t s_in, uint8_t p_in,
    uint8_t *a_out, uint8_t *x_out, uint8_t *y_out,
    uint8_t *s_out, uint8_t *p_out)
{
    if (!core) gate_init();

    // Zero all RAM
    for (int i = 0; i < 65536; i++)
        ram->Dbg_WriteByte(i, 0);

    // Write program at $0200
    int pc = 0x0200;
    auto W = [&](uint8_t v) { ram->Dbg_WriteByte(pc++, v); };

    // Preamble (13 bytes: $0200-$020C)
    W(0xA2); W(s_in);          // LDX #s_in
    W(0x9A);                   // TXS         -> SP = s_in
    W(0xA9); W(p_in);          // LDA #p_in
    W(0x48);                   // PHA         -> [$100+s_in]=p_in
    W(0xA9); W(a_in);          // LDA #a_in
    W(0xA2); W(x_in);          // LDX #x_in
    W(0xA0); W(y_in);          // LDY #y_in
    W(0x28);                   // PLP        -> SP=s_in, P=p_in

    // Test instruction ($020D+)
    W(opcode);
    if (ilen >= 2) W(op1);
    if (ilen >= 3) W(op2);

    // Postamble
    W(0x08);                   // PHP
    W(0x85); W(0xF0);          // STA $F0     -> save A
    W(0x86); W(0xF1);          // STX $F1     -> save X
    W(0x84); W(0xF2);          // STY $F2     -> save Y
    W(0x68);                   // PLA         -> A = pushed P
    W(0x85); W(0xF3);          // STA $F3     -> save P
    W(0xBA);                   // TSX         -> X = SP_after_test
    W(0x86); W(0xF4);          // STX $F4     -> save SP
    int halt_addr = pc;
    W(0x4C);                   // JMP halt_addr
    W(halt_addr & 0xFF);
    W((halt_addr >> 8) & 0xFF);

    // Reset vector -> $0200
    ram->Dbg_WriteByte(0xFFFC, 0x00);
    ram->Dbg_WriteByte(0xFFFD, 0x02);

    /* Hardware reset via /RES pin
     *
     * The 6502 reset works by asserting the /RES pin (active low).
     * This forces the CPU to abort whatever it was doing (e.g. the
     * JMP * halt loop from a previous test), read the reset vector
     * from $FFFC/$FFFD, and start fetching from that address.
     *
     * The real 6502 reset sequence takes 7 cycles.  We hold /RES
     * for 20 half-cycles (10 cycles) to be safe, then release and
     * let the CPU fetch the reset vector over the next 20 half-cycles.
     */
    ab    = 0;
    db    = 0;
    RnW_s = TriState::One;
    TriState clk = TriState::One;

    // Phase A: assert /RES
    for (int i = 0; i < 20; i++) {
        do_half_step(TriState::One, TriState::One,
                     TriState::Zero,    /* /RES asserted */
                     TriState::One, TriState::Zero, clk);
        clk = NOT(clk);
    }

    // Phase B: release /RES - CPU reads reset vector and prepares to fetch from $0200
    for (int i = 0; i < 20; i++) {
        do_half_step(TriState::One, TriState::One,
                     TriState::One,     /* /RES released */
                     TriState::One, TriState::Zero, clk);
        clk = NOT(clk);
    }

    // Run until halt loop detected
    for (int i = 0; i < 4000; i++) {
        do_half_step(TriState::One, TriState::One, TriState::One,
                     TriState::One, TriState::Zero, clk);
        clk = NOT(clk);

        // Halt: address bus shows halt_addr on a read cycle
        if (ab == (uint16_t)halt_addr && RnW_s == TriState::One)
            break;
    }

    // Read results from zero-page
    *a_out = ram->Dbg_ReadByte(0xF0);
    *x_out = ram->Dbg_ReadByte(0xF1);
    *y_out = ram->Dbg_ReadByte(0xF2);
    *p_out = ram->Dbg_ReadByte(0xF3) & 0xCF;   // mask B + bit 5
    *s_out = ram->Dbg_ReadByte(0xF4);
}

}  // extern "C"
