#!/bin/bash

#Use general functions
source add_build.sh
source basic_build.sh

#Setup installing and cloning directories
export INSTALL_DIR="/mnt/spack_hpl_manila/opt"
export CLONE_DIR="/mnt/spack_hpl_manila/mysteryapp"
export BUILD_DIR="/mnt/spack_hpl_manila/build"

mkdir -p $INSTALL_DIR $CLONE_DIR $BUILD_DIR

#Set flags
export CXX="g++"
export CXXFLAGS="-O3 -march=native"

#git-lfs and boost
sudo apt-get update
sudo apt -y install git-lfs libboost-all-dev

#eigen
basic_cmake_github https://gitlab.com/libeigen/eigen.git -n eigen
add_build $INSTALL_DIR/eigen eigen

#nlohmann json
basic_cmake_github https://github.com/nlohmann/json.git -n nlohmann_json -c v3.11.3
add_build $INSTALL_DIR/json nlohmann_json

#Xerces
basic_cmake_tarball https://dlcdn.apache.org//xerces/c/3/sources/xerces-c-3.3.0.tar.gz xerces
add_build $INSTALL_DIR/xerces xerces

export XercesC_INCLUDE_DIR=$INSTALL_DIR/xerces-c-3.3.0/include
export XercesC_LIBRARY=$INSTALL_DIR/xerces-c-3.3.0/lib/libxerces-c.so
export XercesC_VERSION=3.3.0

#oneTBB
basic_cmake_github https://github.com/uxlfoundation/oneTBB.git -n oneTBB -c v2022.2.0
add_build $INSTALL_DIR/oneTBB oneTBB

#pythia
cd $CLONE_DIR
wget https://pythia.org/download/pythia83/pythia8313.tgz
tar -xzf pythia8313.tgz -C pythia
rm -rf pythia8313.tgz
cd pythia8313
mkdir $INSTALL_DIR/pythia
./configure --prefix=$INSTALL_DIR/pythia
make -j$(nproc)
sudo make install 
cd $CLONE_DIR
add_build $INSTALL_DIR/pythia pythia

#root
sudo apt -y install binutils cmake dpkg-dev g++ gcc libssl-dev git libx11-dev \
    libxext-dev libxft-dev libxpm-dev python3 libtbb-dev libvdt-dev libgif-dev

basic_cmake_github https://github.com/root-project/root.git -n root -c v6-34-04 --cmake-args \
    -Dgnuinstall=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -Dvdt=OFF \
    -Dxrootd=OFF \
    -Ddavix=OFF \
    -DCMAKE_CXX_FLAGS="-w" \
    -Droottest="OFF" \
    -Dtesting="OFF"
add_build $INSTALL_DIR/root root
source thisroot.sh

#hepmc
basic_cmake_github https://gitlab.cern.ch/hepmc/HepMC3.git -n HepMC3 -c "3.3.1" --cmake-args \
    -DHEPMC3_ENABLE_ROOTIO:BOOL=OFF             \
    -DHEPMC3_ENABLE_PROTOBUFIO:BOOL=OFF         \
    -DHEPMC3_ENABLE_TEST:BOOL=OFF               \
    -DHEPMC3_INSTALL_INTERFACES:BOOL=ON         \
    -DHEPMC3_BUILD_STATIC_LIBS:BOOL=OFF         \
    -DHEPMC3_BUILD_DOCS:BOOL=OFF                \
    -DHEPMC3_BUILD_EXAMPLES=ON                  \
    -DHEPMC3_ENABLE_PYTHON:BOOL=ON
add_build $INSTALL_DIR/HepMC3 HepMC3

#LCIO
basic_cmake_github https://github.com/iLCSoft/LCIO.git -n LCIO -c "v02-22-05"
add_build $INSTALL_DIR/LCIO LCIO

#geant4
basic_cmake_github https://github.com/Geant4/geant4.git -n geant4 -c "v11.3.0"
add_build $INSTALL_DIR/LCIO LCIO
