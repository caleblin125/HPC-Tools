#!/bin/bash

add_build() {
    local prefix="$1"
    local name="$2"

    if [[ -z "$prefix" ]]; then
        echo "Usage: add_build <prefix> [<name>]"
        return 1
    fi

    # Normalize path
    prefix="$(cd "$prefix" 2>/dev/null && pwd)"
    if [[ -z "$prefix" ]]; then
        echo "Directory does not exist: $1"
        return 1
    fi

    # If name not provided, use basename
    [[ -z "$name" ]] && name="$(basename "$prefix")"

    # Add bin
    [[ -d "$prefix/bin" ]] && export PATH="$prefix/bin:$PATH"

    # Add lib or lib64
    if [[ -d "$prefix/lib" ]]; then
        export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$prefix/lib:${LIBRARY_PATH:-}"
        #Export Variables
        declare -x "${name^^}_LIBRARY=$prefix/lib"
    elif [[ -d "$prefix/lib64" ]]; then
        export LD_LIBRARY_PATH="$prefix/lib64:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$prefix/lib64:${LIBRARY_PATH:-}"
        #Export Variables
        declare -x "${name^^}_LIBRARY=$prefix/lib"
    fi

    # Add include
    [[ -d "$prefix/include" ]] && {
        export CPATH="$prefix/include:${CPATH:-}"
        export C_INCLUDE_PATH="$prefix/include:${C_INCLUDE_PATH:-}"
        export CPLUS_INCLUDE_PATH="$prefix/include:${CPLUS_INCLUDE_PATH:-}"
        # Export variables
        declare -x "${name^^}_INCLUDE_DIR=$prefix/include"
    }

    # Add CMake prefix path
    export CMAKE_PREFIX_PATH="$prefix:${CMAKE_PREFIX_PATH:-}"

    echo "Added build prefix: $prefix"
}

# Remove a build prefix from environment variables and unset exported variables
remove_build() {
    local prefix="$1"
    local name="$2"

    if [[ -z "$prefix" ]]; then
        echo "Usage: remove_build <prefix> [<name>]"
        return 1
    fi

    # If name not provided, use basename
    [[ -z "$name" ]] && name="$(basename "$prefix")"
    local uname="${name^^}"  # uppercase name for exported variables

    # Helper: remove a directory from a colon-separated variable
    remove_path() {
        local varname="$1"
        local dir="$2"
        local current
        eval "current=\$$varname"
        eval "export $varname=\"$(echo "$current" | tr ':' '\n' | grep -v "^$dir\$" | paste -sd ':' -)\""
    }

    # Remove bin
    remove_path PATH "$prefix/bin"

    # Remove lib/lib64
    remove_path LD_LIBRARY_PATH "$prefix/lib"
    remove_path LD_LIBRARY_PATH "$prefix/lib64"
    remove_path LIBRARY_PATH "$prefix/lib"
    remove_path LIBRARY_PATH "$prefix/lib64"

    # Remove include
    remove_path CPATH "$prefix/include"
    remove_path C_INCLUDE_PATH "$prefix/include"
    remove_path CPLUS_INCLUDE_PATH "$prefix/include"

    # Remove CMake prefix
    remove_path CMAKE_PREFIX_PATH "$prefix"

    # Unset exported variables
    unset "${uname}_LIBRARY"
    unset "${uname}_INCLUDE_DIR"

    echo "Removed build prefix: $prefix"
}