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
#
# The default is derived from UR_ROOT, but a toolchain installed for an earlier
# layout will not be under a *new* UR_ROOT -- so prefer any existing install
# over a path that merely follows the naming convention. Getting this wrong
# fails as `cargo: command not found` a minute into a job, after every
# downstream job has already queued behind it.
for candidate in "${CARGO_HOME:-}" "$UR_ROOT/.cargo" /scratch/"$USER"/rust/cargo "$HOME/.cargo"; do
    if [[ -n $candidate && -x $candidate/bin/cargo ]]; then
        export CARGO_HOME=$candidate
        break
    fi
done
export CARGO_HOME=${CARGO_HOME:-$UR_ROOT/.cargo}
if [[ -z ${RUSTUP_HOME:-} ]]; then
    default_rustup=${CARGO_HOME%/cargo}/rustup
    if [[ -d $default_rustup ]]; then
        export RUSTUP_HOME=$default_rustup
    else
        export RUSTUP_HOME=$UR_ROOT/.rustup
    fi
fi
export PATH=$CARGO_HOME/bin:$PATH

# Conda-provided C toolchain, used only to link the Rust binary. Two things make
# it necessary on NYU Torch: login nodes have no C compiler at all, and login
# nodes run glibc 2.39 while compute nodes run 2.34. The binary is therefore
# built on a compute node against a sysroot old enough to run on either.
# Same reasoning as CARGO_HOME above: an existing toolchain beats a conventional
# path, or the job spends a quarter of an hour rebuilding one that already exists.
for candidate in "${UR_TOOLCHAIN:-}" "$UR_ROOT/.toolchain" "$UR_ROOT/../toolchain"; do
    if [[ -n $candidate && -d $candidate/bin ]]; then
        export UR_TOOLCHAIN=$(cd -- "$candidate" && pwd)
        break
    fi
done
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
