#!/usr/bin/env bash

# Usage: ./install_spack.sh [install_dir]

SPACK_DIR="${1:-$HOME}/spack"

if [[ ! -d "$SPACK_DIR" ]]; then
    git clone --depth=2 --branch=releases/v1.0 https://github.com/spack/spack.git "$SPACK_DIR"
    
    cd "$SPACK_DIR" || exit 1
    . share/spack/setup-env.sh
    cd ~ || exit 1

    # Only add to bashrc if not already present
    if ! grep -q "spack/share/spack/setup-env.sh" ~/.bashrc; then
        echo "source $SPACK_DIR/share/spack/setup-env.sh" >> ~/.bashrc
    fi
    source $SPACK_DIR/share/spack/setup-env.sh
    echo "spack installed at $SPACK_DIR"
else
    source $SPACK_DIR/share/spack/setup-env.sh
    echo "spack already exists at $SPACK_DIR"
fi
