/*
 * isa_bridge.c - Wraps fake6502 for single-instruction execution.
 *
 * Strategy: Direct register set → step → read registers.
 * Memory is zeroed before each call so PLA/PLP pop 0x00 from the stack.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o libisa6502.so isa_bridge.c \
 *       /path/to/fake6502/fake6502.c -I/path/to/fake6502
 */

#include <stdint.h>
#include <string.h>
#include "fake6502.h"

// fake6502 requires the host to provide read6502 / write6502
static uint8_t mem[65536];

uint8_t read6502(uint16_t address)               { return mem[address]; }
void    write6502(uint16_t address, uint8_t value){ mem[address] = value; }

// Public API (called from Python via ctypes)

void isa_init(void) {
    // Nothing needed - fake6502 has no global init
}

void isa_poke(uint16_t addr, uint8_t val) {
    mem[addr] = val;
}

/*
 * Execute a single instruction with the given register state.
 *
 * The memory layout is:
 *   $0400          : opcode [op1] [op2]
 *   $0400 + ilen.. : NOP sled (safety)
 *   $FFFC/$FFFD    : reset vector → $0400
 *   $0100-$01FF    : stack page (all zeros)
 *
 * PLA/PLP will read 0x00 from the zeroed stack.  This matches the
 * gate bridge's preamble which also leaves untouched stack bytes at 0.
 */
void isa_execute_instruction(
    uint8_t opcode, uint8_t op1, uint8_t op2, int ilen,
    uint8_t a_in, uint8_t x_in, uint8_t y_in, uint8_t s_in, uint8_t p_in,
    uint8_t *a_out, uint8_t *x_out, uint8_t *y_out,
    uint8_t *s_out, uint8_t *p_out, uint16_t *pc_out)
{
    // Zero all memory
    memset(mem, 0, sizeof(mem));

    // Place the test instruction at $0400
    uint16_t org = 0x0400;
    mem[org] = opcode;
    if (ilen >= 2) mem[org + 1] = op1;
    if (ilen >= 3) mem[org + 2] = op2;

    // NOP sled after the instruction (catches fall-through)
    for (int i = 0; i < 16; i++)
        mem[org + ilen + i] = 0xEA;   // NOP

    // Reset vector
    mem[0xFFFC] = org & 0xFF;
    mem[0xFFFD] = (org >> 8) & 0xFF;

    // Reset the CPU and set registers directly
    reset6502();
    PC = org;
    A  = a_in;
    X  = x_in;
    Y  = y_in;
    SP = s_in;
    setP(p_in);

    // Execute exactly one instruction
    step6502();

    // Read back results
    *a_out  = A;
    *x_out  = X;
    *y_out  = Y;
    *s_out  = SP;
    *p_out  = getP() & 0xCF;   // Mask out B flag (bit 4) and bit 5
    *pc_out = PC;
}
