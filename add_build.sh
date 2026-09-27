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

    # Default name = basename of prefix
    [[ -z "$name" ]] && name="$(basename "$prefix")"

    # Convert name to uppercase and replace non-alphanumeric
    # characters with underscores.
    local uname
    uname="$(printf '%s' "$name" | tr '[:lower:]-.' '[:upper:]__')"

    # Add bin
    if [[ -d "$prefix/bin" ]]; then
        export PATH="$prefix/bin:$PATH"
    fi

    # Determine library directory
    local libdir=""
    if [[ -d "$prefix/lib" ]]; then
        libdir="$prefix/lib"
    elif [[ -d "$prefix/lib64" ]]; then
        libdir="$prefix/lib64"
    fi

    if [[ -n "$libdir" ]]; then
        export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$libdir:${LIBRARY_PATH:-}"

        export "${uname}_LIBRARY=$libdir"
    fi

    # Add include
    if [[ -d "$prefix/include" ]]; then
        export CPATH="$prefix/include:${CPATH:-}"
        export C_INCLUDE_PATH="$prefix/include:${C_INCLUDE_PATH:-}"
        export CPLUS_INCLUDE_PATH="$prefix/include:${CPLUS_INCLUDE_PATH:-}"

        export "${uname}_INCLUDE_DIR=$prefix/include"
    fi

    # Add CMake prefix path
    export CMAKE_PREFIX_PATH="$prefix:${CMAKE_PREFIX_PATH:-}"

    echo "Added build prefix:"
    echo "  Name:   $uname"
    echo "  Prefix: $prefix"
    [[ -n "$libdir" ]] && echo "  Library: $libdir"
    [[ -d "$prefix/include" ]] && echo "  Include: $prefix/include"
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