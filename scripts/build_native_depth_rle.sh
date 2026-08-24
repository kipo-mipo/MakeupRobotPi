#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ORBBEC_SDK_ROOT:-$HOME/OrbbecSDK_v2}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SRC_DIR/native_depth_rle"

if [[ ! -f "$SDK_ROOT/build/src/generated/Export.h" ]]; then
  echo "Missing generated header: $SDK_ROOT/build/src/generated/Export.h" >&2
  exit 1
fi

if [[ ! -f "$SDK_ROOT/build/linux_arm64/lib/libOrbbecSDK.so" ]]; then
  echo "Missing SDK library: $SDK_ROOT/build/linux_arm64/lib/libOrbbecSDK.so" >&2
  exit 1
fi

g++ "$SRC_DIR/native_depth_rle.cpp" \
  -std=c++17 \
  -I"$SDK_ROOT/include" \
  -I"$SDK_ROOT/build/src/generated" \
  -L"$SDK_ROOT/build/linux_arm64/lib" \
  -Wl,-rpath,"$SDK_ROOT/build/linux_arm64/lib" \
  -lOrbbecSDK \
  -o "$OUT"

echo "Built: $OUT"
