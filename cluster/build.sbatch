#!/usr/bin/env bash
#SBATCH --job-name=ur_build
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cpu_short,cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.out

# Builds the release solver binary on a COMPUTE node, so it links against the
# compute-node glibc (2.34) rather than the login-node glibc (2.39).

set -euo pipefail
# Slurm copies the batch script into a spool directory, so BASH_SOURCE points
# there rather than at the repository. UR_ROOT is exported at submission time
# and is the only reliable way back to the tree.
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"

echo "host=$(hostname) glibc=$(ldd --version | head -1)"

if ! command -v cc >/dev/null 2>&1 && [[ ! -x ${CC:-/nonexistent} ]]; then
    echo "no C toolchain found; installing one into $UR_TOOLCHAIN"
    source "$UR_CONDA_SH"
    # Dedicated prefix: never touch the working `env` environment.
    conda create -y -p "$UR_TOOLCHAIN" -c conda-forge \
        gcc_linux-64 sysroot_linux-64=2.17 binutils_linux-64
    # Slurm copies the batch script into a spool directory, so BASH_SOURCE points
# there rather than at the repository. UR_ROOT is exported at submission time
# and is the only reliable way back to the tree.
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"
fi

echo "CC=${CC:-cc}"
"${CC:-cc}" --version | head -1

cargo build --release --manifest-path "$UR_ROOT/rust/Cargo.toml"

BINARY=$UR_ROOT/rust/target/release/royalur_analysis
ls -la "$BINARY"
echo "=== linked glibc requirements ==="
objdump -T "$BINARY" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -u -V | tail -5 || true
echo "=== smoke test (CLI usage) ==="
"$BINARY" 2>&1 | head -5 || true
echo "build complete"
