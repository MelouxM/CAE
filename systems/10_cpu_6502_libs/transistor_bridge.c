/*
 * transistor_bridge.c - Wraps perfect6502 for single-instruction execution.
 *
 * The transistor-level chip is created ONCE in transistor_init() and reused
 * via the /RES pin for each subsequent test. This avoids heap fragmentation
 * from repeated initAndResetChip()/destroyChip() calls.
 *
 * Same preamble/postamble convention as gate_bridge.cpp.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "perfect6502.h"

extern unsigned char memory[65536];
extern unsigned long cycle;

/* Functions from netlist_sim.c (linked into this .so).
 * state_t is already typedef'd to void by perfect6502.h. */
extern void setNode(void *state, unsigned short nn, unsigned char s);
extern void recalcNodeList(void *state);

/* Node numbers from netlist_6502.h */
#define NODE_RES  159
#define NODE_CLK0 1171
#define NODE_RDY  89
#define NODE_SO   1672
#define NODE_IRQ  103
#define NODE_NMI  1297

/* Persistent chip state */
static void *chip = NULL;

void transistor_init(void)
{
    if (chip)
        return;

    /* Fill memory with NOPs for initial stabilization */
    memset(memory, 0xEA, 65536);
    memory[0xFFFC] = 0x00;
    memory[0xFFFD] = 0x04;   /* reset vector -> $0400 (NOP sled) */

    fprintf(stderr, "transistor_init: creating chip (one-time, ~89 KB)...\n");
    chip = initAndResetChip();   /* setup + stabilize + 8-cycle reset */
    fprintf(stderr, "transistor_init: chip ready.\n");

    /* Run a few NOPs so the pipeline is fully settled */
    for (int i = 0; i < 100; i++)
        step(chip);
}

void transistor_execute_instruction(
    uint8_t opcode, uint8_t op1, uint8_t op2, int ilen,
    uint8_t a_in, uint8_t x_in, uint8_t y_in, uint8_t s_in, uint8_t p_in,
    uint8_t *a_out, uint8_t *x_out, uint8_t *y_out,
    uint8_t *s_out, uint8_t *p_out)
{
    if (!chip) transistor_init();

    /* 1. Zero all memory */
    memset(memory, 0, 65536);

    /* 2. Write program at $0200 */
    int pc = 0x0200;
#define W(v) do { memory[pc++] = (v); } while(0)

    /* Preamble */
    W(0xA2); W(s_in);
    W(0x9A);
    W(0xA9); W(p_in);
    W(0x48);
    W(0xA9); W(a_in);
    W(0xA2); W(x_in);
    W(0xA0); W(y_in);
    W(0x28);

    /* Test instruction */
    W(opcode);
    if (ilen >= 2) W(op1);
    if (ilen >= 3) W(op2);

    /* Postamble */
    W(0x08);
    W(0x85); W(0xF0);
    W(0x86); W(0xF1);
    W(0x84); W(0xF2);
    W(0x68);
    W(0x85); W(0xF3);
    W(0xBA);
    W(0x86); W(0xF4);
    int halt_addr = pc;
    W(0x4C);
    W(halt_addr & 0xFF);
    W((halt_addr >> 8) & 0xFF);

    /* Reset vector -> $0200 */
    memory[0xFFFC] = 0x00;
    memory[0xFFFD] = 0x02;

#undef W

    /* 3. Hardware reset via /RES pin */
    setNode(chip, NODE_RES, 0);
    setNode(chip, NODE_CLK0, 1);
    setNode(chip, NODE_RDY, 1);
    setNode(chip, NODE_SO, 0);
    setNode(chip, NODE_IRQ, 1);
    setNode(chip, NODE_NMI, 1);

    /* Hold /RES for 16 half-cycles (8 full cycles) */
    for (int i = 0; i < 16; i++)
        step(chip);

    /* Release /RES */
    setNode(chip, NODE_RES, 1);
    recalcNodeList(chip);
    cycle = 0;

    /* 4. Run until halt loop */
    for (int i = 0; i < 8000; i++) {
        step(chip);
        if (readAddressBus(chip) == (unsigned short)halt_addr &&
            readRW(chip))
            break;
    }

    /* 5. Read results */
    *a_out = memory[0xF0];
    *x_out = memory[0xF1];
    *y_out = memory[0xF2];
    *p_out = memory[0xF3] & 0xCF;
    *s_out = memory[0xF4];
}
