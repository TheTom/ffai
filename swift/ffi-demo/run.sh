#!/usr/bin/env bash
# Build the Rust FFI staticlib + compile/link/run the Swift driver.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUST="$HERE/../../rust"
HDR="$RUST/crates/ffai-ffi/include"
LIB="$RUST/target/debug"

( cd "$RUST" && cargo build -p ffai-ffi )

swiftc "$HERE/main.swift" \
  -import-objc-header "$HDR/ffai.h" \
  -L "$LIB" -lffai_ffi \
  -o "$HERE/ffai-ffi-demo"

exec "$HERE/ffai-ffi-demo"
