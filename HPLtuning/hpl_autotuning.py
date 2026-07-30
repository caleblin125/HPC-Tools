
import random
import re
import os
import subprocess
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
try:
    import statsmodels.api as sm
    import pandas as pd
except ModuleNotFoundError:
    # Regression is diagnostic only; the search and plotting paths do not
    # require these heavier optional packages.
    sm = None
    pd = None
import argparse
from pathlib import Path
from parameter_search import EliteRandomSearch, Parameter, Param

script_location = Path(__file__).resolve().parent

sample = os.environ.get("HPL_TEMPLATE", os.path.join(script_location, "HPL.dat"))
sampleName = os.environ.get("HPL_BENCHMARK_NAME", "HPL")
sampleExtension = "dat"
OUTPUT_SUBDIR = os.environ.get("HPL_OUTPUT_SUBDIR", "HPL_RUN")
CONFIG_SUBDIR = os.environ.get("HPL_CONFIG_SUBDIR", "hpl_gen_configs")
RESULTS_FILE = os.environ.get("HPL_RESULTS_FILE", "hpl_config_results.json")

with open(sample, "r") as file:
    sampleLines = file.readlines()
    sampleText = "".join(sampleLines)
    
batch = os.environ.get("HPL_JOB_TEMPLATE", os.path.join(script_location, "hpl.job"))
with open(batch, "r") as file:
    batchLines = file.readlines()
    batchText = "".join(batchLines)

ROOT = os.getcwd()
HPL_LOCATION = os.environ.get("HPL_EXECUTABLE", "/haydean/caleb/opt/HPL/bin/xhpl")
print("ROOT", ROOT)

NODES = 4
CORES_PER_NODE = 6
TOTAL_TASKS = NODES * CORES_PER_NODE

RAM = 16 #Gigabytes
PERCENT = 0.98
MAX_NS = int((NODES * PERCENT * RAM * 1e9 / 8) ** 0.5)

NUDGE_NUM = 3
NUDGE_JUMP = 0.05

params = [
    Param("NS",         [int(i) for i in range(int(0.7*MAX_NS), MAX_NS)]), #Testing small first
    Param("NB",         [i for i in range(150, 400)]),
    # Param("PMAP",       [0, 1]),
    Param("PMAP",       [0]),
    Param("PFACT",      [0, 1, 2]),
    Param("NBMIN",      [i for i in range(1, 8)]),
    Param("NDIV",       [i for i in range(2, 6)]),
    Param("RFACT",      [0, 1, 2]),
    # Param("BCAST",      [0, 1, 2, 3, 4, 5]),
    Param("BCAST",      [0, 1, 2]),
    Param("DEPTH",      [i for i in range(1, 7)]),
    Param("SWAP",       [0, 1, 2]),
    Param("L1",         [0, 1]),
    Param("U",          [0, 1]),
    # Param("P",          [i for i in range(1, TOTAL_TASKS+1) if (TOTAL_TASKS % i == 0)]),
    # Param("Q",          [i for i in range(1, TOTAL_TASKS+1) if (TOTAL_TASKS % i == 0)])
    Param("P",          [4]),
    Param("Q",          [6])
]

# Intel's distributed HPL-AI binary applies optimized internal values for
# NB, PFACT, RFACT, BCAST, DEPTH, SWAP, L1, U, and EQUIL.  Do not report a
# fake search over knobs the executable ignores; retain only fields observed
# in its output to affect the run.
if os.environ.get("HPL_AI_MODE") == "1":
    params = [
        Param("NS", [int(i) for i in range(int(0.7 * MAX_NS), MAX_NS)]),
        Param("PMAP", [0, 1]),
        Param("NBMIN", [i for i in range(1, 8)]),
        Param("NDIV", [i for i in range(2, 6)]),
        Param("P", [4]),
        Param("Q", [6]),
    ]

batchParams = [
    Param("NODES", [NODES]),
    Param("TASKS", [TOTAL_TASKS]),
    Param("HPL_LOCATION", [HPL_LOCATION]),
    Param("ROOT", [ROOT]),
    Param("DESCRIPTION", ["autotuning"]),
    Param("TIME", ["1:00:00"]),
    Param("CONFIG_ID", ["0"]),
    Param("CONFIG_PATH", [""])
]

def getParam(params:"list[Param]", name):
    for p in params:
        if p.name == name:
            return p
    # raise Exception(f"Param {name} not found in {params}")
    return None

def writeFile(filename, params:"list[Param]", lines:"list[str]"):
    newLines = []
    for line in lines:
        newLine = line
        for p in params:
            newLine = p.replace(newLine)
        newLines.append(newLine)
    
    with open(filename, "w") as file:
        file.writelines(newLines)

def heuristic(entry):
    if "Possible" not in entry:
        return 0
    elif ("GFlops" in entry) and entry["Possible"]:
        return entry["GFlops"]
    else:
        return 0

def parseHPL(filename):
    with open(filename, "r") as file:
        text = file.read()
    # A completed Slurm job is not a valid benchmark without HPL's residual
    # check passing.
    if not re.search(r"\.\.\.\.\.\. PASSED", text):
        return []
    lines = iter(text.splitlines(keepends=True))

    nextline = next(lines, "")
    while not re.match(r"The following parameter values will be used", nextline):
        nextline = next(lines, "")
        if nextline == "":
            return []

    values = {}
    while not re.match(r"-+", nextline):
        nextline = next(lines, "")
        if nextline == "":
            return []
        match = re.match(r"(.*):(.*)", nextline)
        if match == None:
            continue
        arg = match.group(1).strip()
        val = match.group(2).strip()
        if arg not in ("PMAP", "SWAP", "L1", "U", "EQUIL", "ALIGN"):
            val = [s.strip() for s in val.split(" ")]
            val = [v for v in val if v != ""]
        values[arg] = val

    names = ["T/V","N","NB","P","Q","Time","GFlops"]
    dataTypes = [str, int, int, int, int, float, float]
    index = 0
    runs = []
    while nextline != "":
        nextline = next(lines, "")
        match = re.match(r"T\/V +N +NB +P +Q +Time +Gflops", nextline)
        if match == None:
            continue
        next(lines, "")
        nextline = next(lines, "")
        vals = [v.strip() for v in nextline.split(" ")]
        vals = [v for v in vals if v != ""]
        entry = {name : dType(val) for name, val, dType in zip(names, vals, dataTypes)}
        entry["index"] = index
        index += 1
        runs.append(entry)

    if len(runs) == 0:
        return []

    # best = sorted(runs, key = lambda x:x["GFlops"], reverse=True)
    # with open(filename+".res", "w") as file:
    #     file.write("\n".join([f"{k:<15}: {str(v)}" for k, v in values.items()]))
    #     file.write("\n\nsorted:\n")
    #     file.write("         "+"".join([f"{n:<13}" for n in names])+"\n")
    #     for r in best:
    #         file.write(f"run {r["index"]:<3}  ")
    #         file.write("".join([f"{str(r[k]):<13}" for k in names]))
    #         file.write("\n")

    #     file.write("\n\nunsorted:\n")
    #     file.write("         "+"".join([f"{n:<13}" for n in names])+"\n")
    #     for r in runs:
    #         file.write(f"run {r["index"]:<3}  ")
    #         file.write("".join([f"{str(r[k]):<13}" for k in names]))
    #         file.write("\n")
    
    return runs

def run_output_path(config_id, job_id):
    return os.path.join(ROOT, "output", OUTPUT_SUBDIR, f"hpl_config_{config_id}_job_{job_id}.out")

def submit_config(config_id, config_path, description):
    getParam(batchParams, "DESCRIPTION").rand = description
    getParam(batchParams, "CONFIG_ID").rand = str(config_id)
    getParam(batchParams, "CONFIG_PATH").rand = str(Path(config_path).resolve())
    # Keep each generated Slurm script beside its HPL configuration instead of
    # cluttering the workspace root.  sbatch reads the script at submission,
    # so this location is also safe to retain as run provenance.
    os.makedirs(os.path.join(ROOT, CONFIG_SUBDIR), exist_ok=True)
    batch_file = os.path.join(ROOT, CONFIG_SUBDIR, f"hpl_config_{config_id}.job")
    writeFile(batch_file, batchParams, batchLines)
    batch_result = subprocess.run(["sbatch", batch_file], capture_output=True, text=True, check=False)
    match = re.search(r"Submitted batch job (\d+)", batch_result.stdout)
    if not match:
        raise RuntimeError(f"Batch submission failed:\n{batch_result.stdout}{batch_result.stderr}")
    return match.group(1)

def wait_for_result(config_id, job_id):
    while True:
        time.sleep(1.0)
        watch_result = subprocess.run(["scontrol", "show", "job", str(job_id)], capture_output=True, text=True, check=False)
        output = watch_result.stdout + watch_result.stderr
        match = re.search(r"JobState=(\w+)", output)
        if not match:
            raise RuntimeError(f"Could not determine job state:\n{output}")
        state = match.group(1)
        if state == "COMPLETED":
            break
        if state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}:
            return False, None, state
    output_path = run_output_path(config_id, job_id)
    results = parseHPL(output_path) if os.path.exists(output_path) else []
    return bool(results), (results[0]["GFlops"] if results else None), state

search = EliteRandomSearch(
    [Parameter(parameter.name, parameter.values) for parameter in params],
    elite_count=7,
    mutation_count=NUDGE_NUM,
    mutation_fraction=NUDGE_JUMP,
)

def hpl_params(values):
    """Translate a generic candidate dictionary into HPL template values."""
    candidate = [parameter.copy() for parameter in params]
    for parameter in candidate:
        parameter.rand = values[parameter.name]
    return candidate

def generateRand(index:int, values:dict, origin:str):
    file = f"{sampleName}_config_{index}.{sampleExtension}"
    print(f"Run {index}", file)

    newP = hpl_params(values)
        
    #Adjust P and Q to match count
    p = getParam(newP, "P")
    q = getParam(newP, "Q")
    
    if (p is not None) and (q is not None):
        q.rand = TOTAL_TASKS // p.rand
    
    print("New Parameters:", newP)
    os.makedirs(os.path.join(ROOT, CONFIG_SUBDIR), exist_ok=True)
    config_path = os.path.join(ROOT, CONFIG_SUBDIR, file)
    writeFile(config_path, newP, sampleLines)
    #writeFile(os.path.join(os.path.dirname(HPL_LOCATION), "HPL.dat"), newP, sampleLines)
    
    job_id = submit_config(index, config_path, f"autotuning {origin}")
    succeeded, GFlops, state = wait_for_result(index, job_id)
    
    print("succeed", succeeded,"; GFlops: ", GFlops)

    entry = {
        "Config Id" : index,
        "Config File" : os.path.relpath(config_path, ROOT),
        "Output File" : os.path.relpath(run_output_path(index, job_id), ROOT),
        "GFlops" : GFlops,
        "Job Id" : int(job_id),
        "State" : state,
        "Possible" : succeeded
    }
    for p in newP:
        entry[p.name] = p.rand

    # Write file
    write = RESULTS_FILE
    if os.path.exists(write):
        with open(write, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)

    with open(write, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    plot()

    return entry

def plot():
    if not os.path.exists(RESULTS_FILE):
        return
    
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("highest gflops:", max([d["GFlops"] for d in data if "GFlops" in d], key = lambda x : 0 if x is None else x))

    def plot2Var(x, y, file, succeeded=True):
        plt.cla()
        dataX = []
        dataY = []
        badDataX = []
        badDataY = []
        for d in data:
            if (not succeeded) or d["Possible"]:
                if (x in d) and (y in d) and d[y] is not None:
                    dataX.append(d[x])
                    dataY.append(d[y])
            else:
                if (x in d) and (y in d) and d[y] is not None:
                    badDataX.append(d[x])
                    badDataY.append(d[y])
        
        plt.scatter(badDataX, badDataY, c="red")
        plt.scatter(dataX, dataY, c="green")
        plt.xlabel(x)
        plt.ylabel(y)
        
        # Auto-scale based on actual finite data
        all_y = [v for v in dataY + badDataY if v is not None]
        if all_y:
            plt.ylim(0, max(all_y) * 1.1)  # 10% headroom above max
        
        plt.savefig(file, dpi=300)
        
    os.makedirs("plots", exist_ok = True)
    for p in params:
        plot2Var(p.name, "GFlops", f"plots/{p.name}.png")

    # Linear Regression
    try:
        if sm is None or pd is None:
            return
        variables = [p.name for p in params]
        usable = []
        yVar = "GFlops"
        for d in data:
            if all([v in d for v in variables]):
                if "Possible" in d and d["Possible"]:
                    usable.append(d)
        
        df = pd.DataFrame(usable)
        # Separate X and y properly
        X = df[variables].drop(columns=[yVar])
        X = sm.add_constant(X)
        y = df[yVar]*df["NumCores"]

        model = sm.OLS(y, X).fit()
        print(model.summary())
    except:
        pass
    
epochs = 1000
batches = 10
top = 7
initRand = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-config", type=int, help="rerun a saved configuration by its stable configuration index")
    parser.add_argument("--epochs", type=int, default=epochs, help="number of sequential tuning batches")
    parser.add_argument("--batches", type=int, default=batches, help="candidates per batch")
    parser.add_argument("--seed-known", action="store_true", help="seed the first batch with mutations of the current known-best HPL point")
    args = parser.parse_args()
    os.makedirs(CONFIG_SUBDIR, exist_ok = True)

    if args.validate_config is not None:
        legacy = Path(ROOT) / CONFIG_SUBDIR / f"{sampleName}_{args.validate_config}.dat"
        current = Path(ROOT) / CONFIG_SUBDIR / f"{sampleName}_config_{args.validate_config}.dat"
        config_path = current if current.exists() else legacy
        if not config_path.exists():
            raise FileNotFoundError(f"No saved configuration for index {args.validate_config}")
        job_id = submit_config(args.validate_config, config_path, "validation rerun")
        succeeded, gflops, state = wait_for_result(args.validate_config, job_id)
        print(json.dumps({"Config Id": args.validate_config, "Job Id": int(job_id), "State": state,
                          "Valid": succeeded, "GFlops": gflops,
                          "Output File": os.path.relpath(run_output_path(args.validate_config, job_id), ROOT)}, indent=2))
        raise SystemExit(0 if succeeded else 1)
    tested = os.listdir(CONFIG_SUBDIR)
    
    #Find current index
    index = -1
    for t in tested:
        match=re.match(rf"{sampleName}_(?:config_)?(\d+)\.{sampleExtension}$", t)
        if match:
            if int(match.group(1)) > index:
                index = int(match.group(1))
    index += 1

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data:"list[dict]" = json.load(f)
        plot()
    else:
        data = []

    known_seed = {
        "NS": 84731, "NB": 250, "PMAP": 0, "PFACT": 2, "NBMIN": 2,
        "NDIV": 5, "RFACT": 0, "BCAST": 0, "DEPTH": 1, "SWAP": 2,
        "L1": 0, "U": 1, "P": 4, "Q": 6,
        "Possible": True, "GFlops": 0.0,
    }
    for e in range(args.epochs):
        proposal_history = data
        if args.seed_known and not data:
            proposal_history = [known_seed] * search.elite_count
        candidates = search.propose(proposal_history, args.batches, score_key="GFlops", valid_key="Possible")
        for offset, candidate in enumerate(candidates):
            data.append(generateRand(index + offset, candidate.values, candidate.origin))
        index = index + args.batches
        
        
