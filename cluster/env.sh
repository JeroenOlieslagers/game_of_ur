#!/usr/bin/env bash
# Shared environment for the Slurm job scripts. Source this from every job.
#
# Everything is derived from the location of this file, so the repository can
# live anywhere. Every exported variable can be overridden from the environment.

UR_CLUSTER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export UR_ROOT=${UR_ROOT:-$(cd -- "$UR_CLUSTER_DIR/.." && pwd)}
export UR_MODELS=${UR_MODELS:-$UR_ROOT/models}
export UR_LOGS=${UR_LOGS:-$UR_ROOT/logs}

# Keep cargo and rustup off a quota-limited home directory: cargo creates a lot
# of small files. On NYU Torch these belong under /scratch, which is where the
# repository itself should live.
export CARGO_HOME=${CARGO_HOME:-$UR_ROOT/.cargo}
export RUSTUP_HOME=${RUSTUP_HOME:-$UR_ROOT/.rustup}
export PATH=$CARGO_HOME/bin:$PATH

# Conda-provided C toolchain, used only to link the Rust binary. Two things make
# it necessary on NYU Torch: login nodes have no C compiler at all, and login
# nodes run glibc 2.39 while compute nodes run 2.34. The binary is therefore
# built on a compute node against a sysroot old enough to run on either.
export UR_TOOLCHAIN=${UR_TOOLCHAIN:-$UR_ROOT/.toolchain}

if [[ -d $UR_TOOLCHAIN/bin ]]; then
    export PATH=$UR_TOOLCHAIN/bin:$PATH
    if [[ -x $UR_TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc ]]; then
        export CC=$UR_TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc
        export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$CC
    fi
fi

# Conda, used only to create the toolchain above when it is missing.
export UR_CONDA_SH=${UR_CONDA_SH:-/scratch/jo2229/miniforge3/etc/profile.d/conda.sh}

mkdir -p "$UR_LOGS"
