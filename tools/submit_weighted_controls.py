#!/usr/bin/env python3
"""Submit exactly 30 declared controls, recording accepted jobs for safe resume."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path("/Data4/mi_25026207/drone/v8_wireflynet")
FIELDS = ["run","graph","seed","job_id","status","scheduler_message"]

def command(graph, seed):
    return ["qsub","-v",f"GRAPH={graph},SEED={seed}","jobs/v8_weighted_controls.pbs"]

def save_attempt(ledger, row, update=False):
    rows=[]
    if ledger.exists():
        with ledger.open(newline="") as f:
            rows=list(csv.DictReader(f))
    matches=[i for i,r in enumerate(rows) if r["run"]==row["run"]]
    if update:
        if len(matches)!=1:
            raise RuntimeError("Ledger update is ambiguous; stop and review.")
        rows[matches[0]]=row
    else:
        if matches:
            raise RuntimeError("Run already has a ledger entry; do not duplicate it.")
        rows.append(row)
    tmp=ledger.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,ledger)

def submit_one(run, submit, ledger):
    cmd = command(run["graph"],run["seed"])
    if not submit:
        print(" ".join(cmd))
        return True
    row = dict(run=run["run"],graph=run["graph"],seed=run["seed"],job_id="",
               status="SUBMITTING_CHECK_QUEUE",scheduler_message="Submission attempt started; verify queue if interrupted.")
    # Record intent BEFORE qsub: even an interruption between acceptance and
    # receipt of the job ID will block blind resubmission of this run.
    save_attempt(ledger,row)
    try:
        result = subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    except OSError as exc:
        print(f"qsub could not be started: {exc}",file=sys.stderr)
        return False
    message = (result.stdout+"\n"+result.stderr).strip()
    candidates = [line.strip() for line in result.stdout.splitlines()
                  if re.fullmatch(r"\d+(?:\[[^\]]*\])?(?:\.[A-Za-z0-9._-]+)?",line.strip())]
    ok = result.returncode == 0 and len(candidates) == 1
    # An unexpected/failed scheduler response is recorded and blocked until
    # reviewed, rather than risking a duplicate if submission status is unclear.
    row.update(job_id=candidates[0] if len(candidates)==1 else "",
               status="SUBMITTED" if ok else "CHECK_QUEUE_BEFORE_RETRY",scheduler_message=message)
    save_attempt(ledger,row,update=True)
    if ok:
        print(f"{row['job_id']} | {run['run']}",flush=True)
        return True
    print(f"STOP: scheduler response for {run['run']}:\n{message}\n"
          "Check qstat and the ledger before retrying. Earlier accepted jobs are saved.",file=sys.stderr)
    return False

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit",action="store_true",help="Actually submit; default only prints planned commands")
    args=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/"tools/preflight_weighted_controls.py"),"--verify"],check=True)
    folder=ROOT/"experiments_weighted_loss/_control_submission"
    plan=json.loads((folder/"inventory.json").read_text())
    ledger=folder/"jobs.csv"
    with (folder/"submit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        known={}
        if ledger.exists():
            with ledger.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row["run"] in known:
                        raise RuntimeError("Duplicate ledger records; stop and review.")
                    known[row["run"]]=row
        if any(r["status"]!="SUBMITTED" for r in known.values()):
            raise RuntimeError("A scheduler response needs review. Do not resubmit; inspect jobs.csv and qstat.")
        count=0
        for run in plan["runs"]:
            if run["run"] in known:
                print(f"SKIP already submitted: {known[run['run']]['job_id']} | {run['run']}")
                continue
            target=ROOT/"experiments_weighted_loss"/run["run"]
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f"Existing output without a ledger entry: {target}. Review; do not duplicate.")
            if not submit_one(run,args.submit,ledger):
                return 1
            count+=1
        print(f"{'Submitted' if args.submit else 'Planned'} {count} new jobs; {len(known)} previously submitted.")
        if args.submit:
            print(f"Job ledger: {ledger}")
    return 0

if __name__=="__main__":
    sys.exit(main())