#!/bin/bash
# build_libs.sh - Build the three 6502 bridge shared libraries.
#
# Each .so uses a linker version script (*.version) to export ONLY the
# bridge API functions and hide everything else. This prevents symbol
# interposition when loaded into a Python process - perfect6502's
# 'memory', 'step', 'cycle' globals would otherwise collide with
# identically-named symbols in Python/numpy, causing SIGSEGV.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FAKE_DIR="${FAKE6502_DIR:-$HOME/fake6502}"
BREAK_DIR="${BREAK6502_DIR:-$HOME/break6502}"
PERFECT_DIR="${PERFECT6502_DIR:-$HOME/perfect6502}"

SKIP_TRANSISTOR=0
for arg in "$@"; do
    [ "$arg" = "--no-transistor" ] && SKIP_TRANSISTOR=1
done

echo "--> Building 6502 Bridge Libraries"
echo "  ISA source:        $FAKE_DIR"
echo "  Gate source:       $BREAK_DIR"
echo "  Transistor source: $PERFECT_DIR"
echo ""

# ISA (fake6502)
echo "[1/3] Building libisa6502.so ..."
gcc -O2 -shared -fPIC \
    -Wl,--version-script=isa.version \
    -o libisa6502.so \
    isa_bridge.c \
    "$FAKE_DIR/fake6502.c" \
    -I"$FAKE_DIR"
echo "  Done: libisa6502.so"

# Gate (break6502)
echo "[2/3] Building libgate6502.so ..."

if [ -f "$BREAK_DIR/test/Decoder6502.bin" ] && [ ! -f Decoder6502.bin ]; then
    cp "$BREAK_DIR/test/Decoder6502.bin" .
    echo "  Copied Decoder6502.bin"
fi

BREAK_SRCS=(
    "$BREAK_DIR/Common/BaseLogicLib/BaseLogic.cpp"
    "$BREAK_DIR/Common/BaseBoardLib/SRAM.cpp"
    "$BREAK_DIR/Chips/M6502Core/core.cpp"
    "$BREAK_DIR/Chips/M6502Core/decoder.cpp"
    "$BREAK_DIR/Chips/M6502Core/predecode.cpp"
    "$BREAK_DIR/Chips/M6502Core/ir.cpp"
    "$BREAK_DIR/Chips/M6502Core/extra_counter.cpp"
    "$BREAK_DIR/Chips/M6502Core/interrupts.cpp"
    "$BREAK_DIR/Chips/M6502Core/dispatch.cpp"
    "$BREAK_DIR/Chips/M6502Core/random_logic.cpp"
    "$BREAK_DIR/Chips/M6502Core/regs_control.cpp"
    "$BREAK_DIR/Chips/M6502Core/alu_control.cpp"
    "$BREAK_DIR/Chips/M6502Core/pc_control.cpp"
    "$BREAK_DIR/Chips/M6502Core/bus_control.cpp"
    "$BREAK_DIR/Chips/M6502Core/flags_control.cpp"
    "$BREAK_DIR/Chips/M6502Core/branch_logic.cpp"
    "$BREAK_DIR/Chips/M6502Core/address_bus.cpp"
    "$BREAK_DIR/Chips/M6502Core/regs.cpp"
    "$BREAK_DIR/Chips/M6502Core/alu.cpp"
    "$BREAK_DIR/Chips/M6502Core/pc.cpp"
    "$BREAK_DIR/Chips/M6502Core/data_bus.cpp"
    "$BREAK_DIR/Chips/M6502Core/flags.cpp"
    "$BREAK_DIR/Chips/M6502Core/debug.cpp"
)

g++ -O2 -shared -fPIC -std=c++17 \
    -Wl,--version-script=gate.version \
    -o libgate6502.so \
    gate_bridge.cpp \
    "${BREAK_SRCS[@]}" \
    -I"$BREAK_DIR" \
    -I"$BREAK_DIR/Common/BaseLogicLib" \
    -I"$BREAK_DIR/Common/BaseBoardLib" \
    -I"$BREAK_DIR/Chips/M6502Core"
echo "  Done: libgate6502.so"

# Transistor (perfect6502)
if [ "$SKIP_TRANSISTOR" -eq 1 ]; then
    echo "[3/3] Skipping libtransistor6502.so (--no-transistor)"
elif [ -d "$PERFECT_DIR" ] && [ -f "$PERFECT_DIR/perfect6502.c" ]; then
    echo "[3/3] Building libtransistor6502.so ..."
    gcc -O2 -shared -fPIC \
        -Wl,--version-script=transistor.version \
        -o libtransistor6502.so \
        transistor_bridge.c \
        "$PERFECT_DIR/netlist_sim.c" \
        "$PERFECT_DIR/perfect6502.c" \
        -I"$PERFECT_DIR" \
        -lm
    echo "  Done: libtransistor6502.so"
else
    echo "[3/3] Skipping libtransistor6502.so (perfect6502 source not found)"
fi

echo ""
echo "--> Build Complete"
ls -la *.so 2>/dev/null

# Verify symbol hiding worked
echo ""
echo "--> Exported symbols (should be ONLY bridge functions)"
for lib in libisa6502.so libgate6502.so libtransistor6502.so; do
    if [ -f "$lib" ]; then
        echo "  $lib:"
        nm -D --defined-only "$lib" 2>/dev/null | grep -v "^$" | grep " T " | awk '{print "    " $3}'
    fi
done
