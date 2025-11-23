#!/bin/bash

basic_cmake_github(){
    local repo_url="$1"
    shift 1
    local name=""
    local checkout=""
    local cmake_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--checkout)
                checkout="$2"
                shift 2
                ;;
            -n|--name)
                name="$2"
                shift 2
                ;;
            --cmake-args)
                shift
                while [[ $# -gt 0 ]]; do
                    cmake_args+=("$1")
                    shift
                done
                ;;
            *)
                echo "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$repo_url" ]]; then
        echo "Usage: build_cmake_project <repo> [-c <ref>] [--cmake-args <args>]"
        return 1
    fi

    if [[ -z "$CLONE_DIR" ]]; then
        echo "WARNING: CLONE_DIR unset"
    fi

    if [[ -z "$BUILD_DIR" ]]; then
        echo "WARNING: BUILD_DIR unset"
    fi

    if [[ -z "$INSTALL_DIR" ]]; then
        echo "ERROR: INSTALL_DIR unset"
        return 1
    fi

    # use github name if name is not offered
    if [[ -z "$name" ]]; then
        local name=$(basename "$repo_url" .git)
    fi
    
    #paths
    local clonepath="$CLONE_DIR/$name"
    local buildpath="$BUILD_DIR/$name"
    local installpath="$INSTALL_DIR/$name"

    echo "=== Cloning $name ==="
    cd $CLONE_DIR
    #Check if can clone
    if [[ ! -d "$name" ]]; then
        git clone "$repo_url" "$name"
    else
        echo "Repo exists, skipping clone."
    fi
    cd $name || return 1

    #Checkout branch
    [[ -n "$checkout" ]] && git checkout "$checkout"

    #Build
    echo "=== Building $name ==="
    mkdir -p $buildpath
    sudo chown -R $USER:$USER $buildpath
    cd "$buildpath" || return 1
    cmake "$clonepath" -DCMAKE_INSTALL_PREFIX="$installpath" -DCMAKE_BUILD_TYPE=RELEASE "${cmake_args[@]}"
    cmake --build $buildpath -j$(nproc)

    #Install
    echo "=== Installing $name ==="
    mkdir -p $installpath
    sudo chown -R $USER:$USER $installpath
    sudo cmake --install $buildpath
}

basic_cmake_tarball() {
    local url="$1"
    local name="$2"
    local tarfile=""
    local src_dir=""

    if [[ -z "$url" ]]; then
        echo "Usage: build_tarball_project <url> [name]"
        return 1
    fi

    if [[ -z "$CLONE_DIR" ]]; then
        echo "WARNING: CLONE_DIR unset"
    fi

    if [[ -z "$BUILD_DIR" ]]; then
        echo "WARNING: BUILD_DIR unset"
    fi

    if [[ -z "$INSTALL_DIR" ]]; then
        echo "ERROR: INSTALL_DIR unset"
        return 1
    fi

    # Derive name from URL if not provided
    [[ -z "$name" ]] && name=$(basename "$url" | sed -E 's/\.tar\.(gz|bz2|xz|tgz)$//')

    local clonepath="$CLONE_DIR/$name"
    local buildpath="$BUILD_DIR/$name"
    local installpath="$INSTALL_DIR/$name"

    cd "$CLONE_DIR" || return 1

    # Download tarball
    tarfile=$(basename "$url")
    if [[ ! -f "$tarfile" ]]; then
        wget "$url"
    else
        echo "Tarball $tarfile already exists, skipping download."
    fi

    # Extract
    rm -rf "$clonepath"
    mkdir -p "$clonepath"
    if [[ "$tarfile" == *.tar.gz || "$tarfile" == *.tgz ]]; then
        tar -xzf "$tarfile" -C "$clonepath" --strip-components=1
    elif [[ "$tarfile" == *.tar.bz2 ]]; then
        tar -xjf "$tarfile" -C "$clonepath" --strip-components=1
    elif [[ "$tarfile" == *.tar.xz ]]; then
        tar -xJf "$tarfile" -C "$clonepath" --strip-components=1
    else
        echo "Unsupported archive format: $tarfile"
        return 1
    fi

    #Build
    echo "=== Building $name ==="
    mkdir -p $buildpath
    sudo chown -R $USER:$USER $buildpath
    cd "$buildpath" || return 1
    cmake "$clonepath" -DCMAKE_INSTALL_PREFIX="$installpath" "${cmake_args[@]}" || return 1
    cmake --build $buildpath -j$(nproc) || return 1

    #Install
    echo "=== Installing $name ==="
    mkdir -p $installpath
    sudo chown -R $USER:$USER $installpath
    sudo cmake --install $buildpath || return 1
}