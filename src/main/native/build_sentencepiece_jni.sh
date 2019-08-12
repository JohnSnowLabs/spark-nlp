#! /bin/bash

set -e
set -v

if [ ! -d "./sentencepiece" ]; then
	echo "This script must be executed in the folder with the sentencepiece CMake files"
	exit 1
fi

if [ ! -d "./../resources/" ] && [$# -lt 1]; then
	echo "This script expects to be executed in <project.dir>/src/main/native or to have destination provided for the .so file"
	exit 1
fi

DEST=$1
DEST=${DEST:-"$PWD/../resources/lib"}

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../resources/lib/ $PWD/sentencepiece
make install
rm -rf CMake* cmake_install.cmake install_manifest.txt *.so Makefile sentencepiece/src sentencepiece/tmp
