# Building the MOS 6502 bridge libraries

System 10 (`systems/10_cpu_6502.py`) drives three independent 6502 emulators at three abstraction
levels (ISA, gate, transistor) through small C/C++ bridge shims compiled into shared libraries:

| Library | Level | Bridge source (this repo) | Upstream emulator |
|---|---|---|---|
| `libisa6502.so` | ISA | `isa_bridge.c` | fake6502 |
| `libgate6502.so` | gate | `gate_bridge.cpp` | break6502 / M6502Core |
| `libtransistor6502.so` | transistor | `transistor_bridge.c` | perfect6502 |

The bridge sources and `*.version` linker scripts in this directory are original glue code authored for
this project (MIT, see the repo `LICENSE`). `build_libs.sh` compiles them together with upstream
emulator sources that are **not vendored here. Both the `.so` files and `Decoder6502.bin` are
`.gitignored` and produced locally.

## Prerequisites

`gcc`, `g++` (C++17), and `make`-free POSIX shell. Then fetch the three upstream source trees. By
default `build_libs.sh` looks for them under `$HOME` (override with the `*_DIR` environment variables
shown below). The commit hashes are the ones pinned in the repo-root `THIRD_PARTY` file (re-verify and
freeze them for any public release).

```bash
# 1. fake6502 - ISA-level model (BSD 2-Clause)
git clone https://github.com/ivop/fake6502 ~/fake6502
git -C ~/fake6502 checkout b52676f840983219b0b9baa13f1d0ebc07aac9f9   # 2024-05-17

# 2. break6502 / M6502Core - gate-level model (license unknown)
git clone https://github.com/ivop/break6502 ~/break6502
git -C ~/break6502 checkout 922af6496a2fa3b0a999e24419b5f8187f0ee98e   # 2024-02-16

# 3. perfect6502 - transistor-level model (MIT)
git clone https://github.com/mist64/perfect6502 ~/perfect6502
git -C ~/perfect6502 checkout 09fc542877a84318291aa42dab143a3e2c3db974
```

## Building

```bash
cd systems/10_cpu_6502_libs
./build_libs.sh                 # builds all three libraries
./build_libs.sh --no-transistor # skip libtransistor6502.so (perfect6502 not needed)
```

Override the source locations if you did not clone into `$HOME`:

```bash
FAKE6502_DIR=/path/to/fake6502 \
BREAK6502_DIR=/path/to/break6502 \
PERFECT6502_DIR=/path/to/perfect6502 \
./build_libs.sh
```

## Artifacts produced

- `libisa6502.so`, `libgate6502.so`, `libtransistor6502.so` — staged in this directory next to the
  bridge sources; `systems/10_cpu_6502.py` loads them from here.
- `Decoder6502.bin` (~272 MB) — the 6502 decode-PLA lookup table. `build_libs.sh` copies it from
  `break6502`'s `test/` tree if present; otherwise break6502's M6502Core generates it on first
  construction (HLE mode). `GateSimulator` loads the single canonical copy staged here.

Each `.so` uses a `--version-script` (`*.version`) to export only the bridge API functions. This
hides upstream globals (e.g. perfect6502's `memory`, `step`, `cycle`) that would otherwise collide with
identically-named symbols in the Python/NumPy process and cause `SIGSEGV`.

## Licensing

The upstream emulators keep their own licenses (full texts in the repo-root `THIRD_PARTY`):

- **fake6502** — BSD 2-Clause.
- **perfect6502** — MIT.
- **break6502 / M6502Core** — license unknown.