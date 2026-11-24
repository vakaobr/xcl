# Building and testing XCL on macOS

XCL is a Windows program. On macOS (Apple Silicon) it cross-compiles to a Windows executable
and runs through CrossOver's Wine with Apple's D3DMetal as the D3D12 backend and Rosetta 2
running the x86-64 code. This is not a supported platform: the build and tests described here
help contributors who work on a Mac.

## Prerequisites

- **Rust** with the `x86_64-pc-windows-gnu` target:
  ```sh
  rustup target add x86_64-pc-windows-gnu
  ```
- **mingw-w64** (the GCC cross-compiler):
  ```sh
  brew install mingw-w64
  ```
- **CrossOver** 26.x or later (`/Applications/CrossOver.app`), with a bottle configured for
  D3DMetal (see below).
- **DXC** (DirectX Shader Compiler): download the release from
  https://github.com/microsoft/DirectXShaderCompiler/releases and note the path to
  `bin/x64/dxcompiler.dll` and `dxil.dll`.

## Build

```sh
cargo build --workspace --target x86_64-pc-windows-gnu
```

The test binaries land under `target/x86_64-pc-windows-gnu/debug/deps/`. With a custom
`CARGO_TARGET_DIR`, they land there instead.

Release builds:
```sh
cargo build --workspace --release --target x86_64-pc-windows-gnu
```

## CrossOver bottle

Create a bottle for testing:

```sh
CX="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver"
"$CX/bin/cxbottle" --bottle xcl-test --create --template win10_64
```

Enable D3DMetal: edit `~/Library/Application Support/CrossOver/Bottles/xcl-test/cxbottle.conf`
and add `"CX_GRAPHICS_BACKEND" = "d3dmetal"` right under the `[EnvironmentVariables]` heading.

Disable the crash debugger (it pops up a dialog on every test failure):

```sh
"$CX/bin/wine" --bottle xcl-test reg add \
    'HKLM\Software\Microsoft\Windows NT\CurrentVersion\AeDebug' \
    /v Debugger /t REG_SZ /d '' /f < /dev/null
"$CX/bin/wine" --bottle xcl-test reg add \
    'HKLM\Software\Microsoft\Windows NT\CurrentVersion\AeDebug' \
    /v Auto /d 0 /f < /dev/null
```

## Running tests

### Single test

```sh
CX="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver"
"$CX/bin/wine" --bottle xcl-test \
    target/x86_64-pc-windows-gnu/debug/deps/xcl-HASH.exe \
    --exact test_name --test-threads=1 --nocapture < /dev/null
```

Replace `xcl-HASH.exe` with the actual binary name (cargo prints it during the build). The
`< /dev/null` closes stdin so the process does not wait for input.

### The whole suite, one test per process

Running all tests in a single libtest process hangs under Wine when many threads contend for
the D3D12 device. Use `scripts/crossover_each.py` to run each test in its own process:

```sh
python3 -I scripts/crossover_each.py \
    target/x86_64-pc-windows-gnu/debug/deps/xcl-HASH.exe \
    results.json \
    --bottle xcl-test
```

The script lists every test, runs them one at a time with a timeout, and writes a JSON
results file. GPU tests get 60 seconds, other tests get 180 seconds. See `--help` for
options.

### DXC path

GPU tests that compile shaders need DXC. Set the path as a Wine path (Z: maps to /):

```sh
export XCL_DXC_DIR='Z:\path\to\dxc\bin\x64'
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `XCL_DXC_DIR` | Path to `dxcompiler.dll` and `dxil.dll` (Wine path, e.g. `Z:\...`) |
| `XCL_ADAPTER` | GPU selection: `auto` (default), `warp`, or a substring of the adapter name |

## Known differences from Windows

- **No double-precision float shader ops.** D3DMetal reports
  `DoublePrecisionFloatShaderOps = FALSE`. Tests that use f64 instructions skip with
  "no doubles".
- **No WARP adapter.** `EnumWarpAdapter` returns DXGI_ERROR_NOT_FOUND. Tests that need WARP
  skip.
- **No host-visible GPU heaps.** `OpenExistingHeapFromAddress` returns E_NOTIMPL. The
  `gpu_uma_import_check_at_init` test fails because the UMA import path is not available.
- **Graphics tests hang.** Tests that draw (e.g. `gpu_graphics_vs_ps_draw`,
  `gpu_rect_list_*`, `gpu_wave64_*_spill*`) may hang under D3DMetal. Use a timeout.
- **Wine error codes.** Some error codes differ between Wine and Windows (e.g. error 2 vs 3
  for a missing path).

## Reporting results

Test results on macOS go to [#258](https://github.com/ivall/xcl/issues/258), not a new issue.
Name the exact commit, the hardware (chip, RAM, macOS version, CrossOver version), the commands
and their counts, and which tool and model wrote the report if an AI helped.
