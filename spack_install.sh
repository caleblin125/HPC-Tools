#!/bin/bash

SPACK_DIR=~/spack
if [[ ! -d "$SPACK_DIR" ]]; then
    git clone --depth=2 --branch=releases/v1.0 https://github.com/spack/spack.git $SPACK_DIR
    cd ~/spack
    . share/spack/setup-env.sh
    cd ~
    echo "source ~/spack/share/spack/setup-env.sh" >> ~/.bashrc
    echo "spack installed"
else
    echo "spack already exists"
fi 