#!bin/bash

# View the stdout file (.out)
catout() {
    local dir="outputs/slurm"
    if [ -n "$1" ]; then
        # If a job number is provided, cat that specific file
        cat "${dir}/job_${1}.out"
    else
        # Find and cat the highest numbered .out file
        local latest_file
        latest_file=$(ls -v "${dir}"/job_*.out 2>/dev/null | tail -n 1)
        if [ -n "$latest_file" ]; then
            echo "=== Displaying: $latest_file ==="
            cat "$latest_file"
        else
            echo "No .out files found in ${dir}/"
        fi
    fi
}

# View the stderr file (.err)
caterr() {
    local dir="outputs/slurm"
    if [ -n "$1" ]; then
        # If a job number is provided, cat that specific file
        cat "${dir}/job_${1}.err"
    else
        # Find and cat the highest numbered .err file
        local latest_file
        latest_file=$(ls -v "${dir}"/job_*.err 2>/dev/null | tail -n 1)
        if [ -n "$latest_file" ]; then
            echo "=== Displaying: $latest_file ==="
            cat "$latest_file"
        else
            echo "No .err files found in ${dir}/"
        fi
    fi
}
