#!/usr/bin/env bash
#
# Build and install tiktoken with profile-guided optimisation (PGO) applied to
# the Rust core.
#
# Tokenising is dominated by `fancy_regex`'s backtracking VM: a branchy
# interpreter loop, its `regex-automata` delegate searches and the allocation
# churn around them. LLVM cannot lay that code out well without knowing which
# branches are hot, so a profiled rebuild is worth roughly 10% on every encode
# benchmark. Semantics are unchanged: only code layout, inlining and branch
# ordering differ.
#
# Usage:
#   scripts/pgo_build.sh [pip install arguments...]   # default: -e .
#
# Requires the `llvm-tools` rustup component (added automatically if missing);
# `llvm-profdata` must come from the same toolchain that produced the profiles.
set -euo pipefail

cd "$(dirname "$0")/.."

pip_args=("$@")
if [ ${#pip_args[@]} -eq 0 ]; then
    pip_args=(-e .)
fi

find_llvm_profdata() {
    find "$(rustc --print sysroot)" -name 'llvm-profdata' -type f | head -n 1
}

llvm_profdata=$(find_llvm_profdata)
if [ -z "$llvm_profdata" ]; then
    echo "== adding the llvm-tools rustup component"
    rustup component add llvm-tools
    llvm_profdata=$(find_llvm_profdata)
fi
if [ -z "$llvm_profdata" ]; then
    echo "error: llvm-profdata not found; cannot build with PGO" >&2
    exit 1
fi

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

echo "== PGO stage 1/2: instrumented build"
RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-generate=$workdir/raw" \
    python -m pip install "${pip_args[@]}"

echo "== PGO stage 1/2: training run"
python scripts/pgo_train.py

echo "== PGO: merging profiles"
"$llvm_profdata" merge -o "$workdir/merged.profdata" "$workdir/raw"

echo "== PGO stage 2/2: optimised build"
RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-use=$workdir/merged.profdata" \
    python -m pip install "${pip_args[@]}"
