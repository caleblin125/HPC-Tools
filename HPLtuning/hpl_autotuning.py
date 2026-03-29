
import random
import re
import os
import subprocess
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import argparse
from pathlib import Path

script_location = Path(__file__).resolve().parent

sample = os.path.join(script_location, "HPL.dat")
sampleName = "HPL"
sampleExtension = "dat"

with open(sample, "r") as file:
    sampleLines = file.readlines()
    sampleText = "".join(sampleLines)
    
batch = os.path.join(script_location, "hpl.job")
with open(batch, "r") as file:
    batchLines = file.readlines()
    batchText = "".join(sampleLines)

class Param():
    def __init__(self, name:str, values:list):
        self.name = name
        self.values = values
        self.rand = random.choice(values)
    
    def generate(self):
        self.rand = random.choice(self.values)

    def move(self, jump:float) -> bool:
        """Moves parameter by some jump percentage

        Args:
            jump (float): jump percent [0.0, 0.5]

        Returns:
            bool: if the value actually moved
        """
        index = self.values.index(self.rand)
        thresh = int(max(1, jump*len(self.values)))
        newIndex = index + random.randint(-thresh, thresh)
        if newIndex < 0:
            newIndex = 0
        if newIndex >= len(self.values):
            newIndex = len(self.values)-1
        self.rand = self.values[newIndex]
        return newIndex != index

    def zero(self):
        self.rand = self.values[0]
        
    def next(self, jump:float=0.0, overflow=True) -> bool:
        """
        Moves this parameter forward in value list by jump percentage (minimum 1)

        Args:
            jump (float, optional): jump percent [0.0, 1.0]. Defaults to 0.0.
            overflow (bool, optional): allow wrapping when at the end of values. Defaults to True.

        Returns:
            bool: _description_
        """
        jumpMax = int(max(jump * len(self.values), 1))
        nextIndex = self.values.index(self.rand) + random.randint(1, jumpMax)
        if overflow:
            self.rand = self.values[nextIndex % len(self.values)]
        else:
            self.rand = self.values[min(nextIndex, len(self.values) - 1)]
        return len(self.values) <= nextIndex

    def replace(self, fileText:str):
        fileText = re.sub(f"<{self.name}>", f"{self.rand}", fileText)
        return fileText
    
    def copy(self):
        clone = Param(self.name, self.values)
        clone.rand = self.rand
        return clone

    def __str__(self) -> str:
        return f"{self.name}: {self.rand}"
    
    def __repr__(self) -> str:
        return str(self)

ROOT = os.getcwd()
HPL_LOCATION = "/global/common/software/m4007/opt/hpl-2.3/bin/xhpl"
print(ROOT)


NODES = 1
CORES_PER_NODE = 128
TOTAL_TASKS = NODES * CORES_PER_NODE

RAM = 512 #Gigabytes
PERCENT = 0.9
MAX_NS = int((PERCENT * RAM * 10e9 / 8) ** 0.5)


NUDGE_NUM = 3
NUDGE_JUMP = 0.05

params = [
    Param("NS",         [int(i * 0.3) for i in range(int(0.7*MAX_NS), MAX_NS)]), #Testing small first
    Param("NB",         [i for i in range(50, 600)]),
    Param("PMAP",       [0, 1]),
    Param("PFACT",      [0, 1, 2]),
    Param("NBMIN",      [i for i in range(1, 8)]),
    Param("NDIV",       [i for i in range(1, 5)]),
    Param("RFACT",      [0, 1, 2]),
    Param("BCAST",      [0, 1, 2, 3, 4, 5]),
    Param("DEPTH",      [i for i in range(6)]),
    Param("SWAP",       [0, 1, 2]),
    Param("L1",         [0, 1]),
    Param("U",          [0, 1]),
    Param("P",          [i for i in range(1, TOTAL_TASKS+1) if (TOTAL_TASKS % i == 0)]),
    Param("Q",          [i for i in range(1, TOTAL_TASKS+1) if (TOTAL_TASKS % i == 0)])
]

batchParams = [
    Param("NODES", [NODES]),
    Param("TASKS", [TOTAL_TASKS]),
    Param("HPL_LOCATION", [HPL_LOCATION]),
    Param("ROOT", [ROOT]),
    Param("DESCRIPTION", ["autotuning"]),
]

#Method to generate random parameters
def genRandom(params = params):
    copy = [p.copy() for p in params]
    for p in copy:
        p.generate()
    return copy

#Method to shift parameters by a random amount
def nudge(params:"list[Param]", values:int, jump:float):
    copy = [p.copy() for p in params]
    while values > 0:
        p = random.choice(copy)
        if p.move(jump):
            values -= 1
    return copy

def getParam(params:"list[Param]", name):
    for p in params:
        if p.name == name:
            return p
    raise Exception(f"Param {name} not found in {params}")

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
    file = open(filename, "r")

    nextline = file.readline()
    while not re.match(r"The following parameter values will be used", nextline):
        nextline = file.readline()
        if nextline == "":
            return []

    values = {}
    while not re.match(r"-+", nextline):
        nextline = file.readline()
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
        nextline = file.readline()
        match = re.match(r"T\/V +N +NB +P +Q +Time +Gflops", nextline)
        if match == None:
            continue
        file.readline()
        nextline = file.readline()
        vals = [v.strip() for v in nextline.split(" ")]
        vals = [v for v in vals if v != ""]
        entry = {name : dType(val) for name, val, dType in zip(names, vals, dataTypes)}
        entry["index"] = index
        index += 1
        runs.append(entry)

    file.close()

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

def generateRand(index:int, params:"list[Param]" = params, nudgeP=False):
    file = f"{sampleName}_{index}.{sampleExtension}"

    if nudgeP:
        newP = nudge(params, NUDGE_NUM, NUDGE_JUMP)
    else:
        newP = genRandom(params)
        
    #Adjust P and Q to match count
    p = getParam(newP, "P")
    q = getParam(newP, "Q")
    
    q.rand = TOTAL_TASKS // p.rand
    
    print("New Parameters:", newP)
    print(os.path.join(ROOT, file))
    writeFile(os.path.join(ROOT, file), newP, sampleLines)
    writeFile(os.path.join(os.path.dirname(HPL_LOCATION), "HPL.dat"), newP, sampleLines)
    
    description = getParam(batchParams, "DESCRIPTION")
    description.rand = f"Autotuning test {index} {'modified' if nudgeP else 'random'}"
    writeFile(os.path.join(ROOT, "hpl.job"), batchParams, batchLines)
    
    #Run batch script
    batch_result = subprocess.run(
        ["sbatch", "hpl.job"],
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    # batch_result = subprocess.run(["sbatch", "hpl.job"], capture_output=True, text=True)
    
    text_output = batch_result.stdout
    match = re.match(r"Submitted batch job (\d+)", text_output)
    
    if match:
        job_id = match.group(1)
        
        #Wait till job finishes
        while True:
            time.sleep(10.0)
            watch_result = subprocess.run(
                ["scontrol", "show", "job", str(job_id)],
                env={**os.environ},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            output = watch_result.stdout + watch_result.stderr

            # Extract job state
            match = re.search(r"JobState=(\w+)", output)
            if not match:
                raise RuntimeError(f"Could not determine job state:\n{output}")

            state = match.group(1)

            if state == "COMPLETED":
                succeeded = True
                break
            elif state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}:
                succeeded = False
                GFlops = None
                break
        
        if succeeded:
            res = parseHPL(f"output/HPL_RUN/hpl_{job_id}.out")
            if len(res) == 0:
                succeeded = False
                GFlops = None
            else:
                succeeded = True
                GFlops = res[0]["GFlops"]
    else:
        succeeded = False
        GFlops = None
        job_id = None
    
    print("succeed", succeeded,"; GFlops: ", GFlops)

    os.system(f"mv {file} hpl_gen_configs")

    entry = {
        "Filename" : file,
        "GFlops" : GFlops,
        "Job Id" : job_id,
        "Possible" : succeeded
    }
    for p in newP:
        entry[p.name] = p.rand

    # Write file
    write = "hpl_config_results.json"
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
    if "hpl_config_results.json" not in os.listdir():
        return
    
    with open("hpl_config_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    def plot2Var(x, y, file, succeeded=True, scaleAxi=False):
        plt.cla()
        dataX = []
        dataY = []
        badDataX = []
        badDataY = []
        for d in data:
            if (not succeeded) or d["Possible"]:
                if (x in d) and (y in d):
                    dataX.append(d[x])
                    dataY.append(d[y])
            else:
                if (x in d) and (y in d):
                    badDataX.append(d[x])
                    badDataY.append(d[y])
        plt.scatter(badDataX, badDataY, c= "red")
        plt.scatter(dataX, dataY, c= "green")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.savefig(file, dpi=300)

    plot2Var("NS", "GFlops", "plots/NS.png")
    plot2Var("NB", "GFlops", "plots/NB.png")
    plot2Var("BCAST", "GFlops", "plots/BCAST.png")

    # Linear Regression
    try: 
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
    
epochs = 3
batches = 10
top = 7
initRand = True
if __name__ == "__main__":
    os.makedirs("hpl_gen_configs", exist_ok = True)
    tested = os.listdir("hpl_gen_configs")
    
    #Find current index
    index = 0
    for t in tested:
        match=re.match(f"{sampleName}(.*).{sampleExtension}", t)
        if match:
            if int(match.group(1)) > index:
                index = int(match.group(1)) + 1

    if os.path.exists("hpl_config_results.json"):
        with open("hpl_config_results.json", "r", encoding="utf-8") as f:
            data:"list[dict]" = json.load(f)
        plot()
    else:
        data = []

    for e in range(epochs):
        try:
            if (e == 0) and initRand: #first gen is random
                for i in range(index, index + batches):
                    data.append(generateRand(i))
            else:
                data.sort(key = heuristic, reverse=True)
                for i in range(index, index + top):
                    for k, v in data[i - index].items():
                        p = getParam(params, k)
                        if p is not None:
                            p.rand = v
                    data.append(generateRand(i, params, True))
                for i in range(index + top, index + batches):
                    data.append(generateRand(i))
            index = index + batches
            
        except KeyboardInterrupt:
            pass
        finally:
            plot()
        