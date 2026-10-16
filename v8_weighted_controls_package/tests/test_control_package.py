#!/usr/bin/env python3
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PKG=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,PKG/"tools"/f"{name}.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def fixture(module, root):
    import numpy as np
    module.ROOT=root
    module.REFERENCE=root/"reference.json"
    cfg=json.loads((PKG/"protocol_reference.json").read_text())["config_snapshot"]
    cfg.update(v8_root=str(root),exp_root=str(root/"experiments_weighted_loss"),
               dataset_root=str(root/"dataset"),image_subroot=str(root/"images"),graphs_dir=str(root/"graphs"))
    source=root/"models/wireflynet.py"
    source.parent.mkdir(parents=True)
    source.write_text("# mock source for preflight unit tests\n")
    cp=root/"configs/wireflynet_g1_weighted_loss.json"
    cp.parent.mkdir()
    cp.write_text(json.dumps(cfg))
    module.REFERENCE.write_text(json.dumps(dict(script_sha256=module.digest(source),config_sha256=module.digest(cp))))
    hashes={}
    meta={"counts":{}}
    for split in ["train","val"]:
        p=Path(cfg["dataset_root"])/split/"pairs.csv"
        p.parent.mkdir(parents=True)
        p.write_text("prev_frame,curr_frame,label_id\np,c,0\n")
        hashes[split]=module.digest(p)
        meta["counts"][split]={"manifest_sha256":hashes[split]}
    (Path(cfg["dataset_root"])/"split_metadata.json").write_text(json.dumps(meta))
    for model,graph in [("wireflynet","bio"),("cnn_matched","none")]:
        for seed in [42,43,44]:
            d=Path(cfg["exp_root"])/f"{model}_{graph}_seed{seed}"
            d.mkdir(parents=True)
            m=dict(config=cfg,config_sha256=module.digest(cp),script_sha256=module.digest(source),
                   eval_test=False,train_sampler="shuffle",class_weighting="inverse_frequency",
                   resolved_arguments=dict(model=model,graph=graph,seed=seed,epochs=40,batch=64,lr=.002,patience=12,workers=4),
                   train_manifest_sha256=hashes["train"],val_manifest_sha256=hashes["val"],
                   loss_class_weights={"FORWARD":.37,"LEFT":6.65,"RIGHT":6.65})
            (d/"run_metadata.json").write_text(json.dumps(m))
            (d/"validation_result.json").write_text(json.dumps(dict(run=d.name,model=model,graph=graph,seed=seed)))
    graphdir=Path(cfg["graphs_dir"])
    entry=np.array([0,1],dtype=np.int64)
    output=np.array([2,3,4],dtype=np.int64)
    def graphfile(p, offset):
        p.parent.mkdir(parents=True,exist_ok=True)
        mask=np.zeros((6,6),dtype=bool)
        for i in range(6):
            mask[i,(i+offset)%6]=True
        np.savez(p,mask=mask,syn_init=mask.astype(float)*2,entry_idx=entry,output_idx=output)
    graphfile(graphdir/"biological/g1_bio.npz",1)
    for kind in ["rewired","random"]:
        for k in range(5):
            graphfile(module.graph_path(graphdir,f"{kind}:{k}"),k+1)
    return cfg

class ControlTests(unittest.TestCase):
    def test_exact_scope_and_commands(self):
        m=load("preflight_weighted_controls")
        s=load("submit_weighted_controls")
        self.assertEqual(len(m.ARMS),30)
        self.assertEqual(len(set(m.ARMS)),30)
        self.assertEqual(sum(g.startswith("rewired:") for g,_ in m.ARMS),15)
        self.assertEqual(sum(g.startswith("random:") for g,_ in m.ARMS),15)
        self.assertEqual(m.run_name("random:4",44),"wireflynet_random4_seed44")
        self.assertEqual(s.command("rewired:0",42),
                         ["qsub","-v","GRAPH=rewired:0,SEED=42","jobs/v8_weighted_controls.pbs"])
        for g,sid in [("bio",42),("none",43),("random:5",42),("rewired:0",45)]:
            with self.assertRaises(ValueError):
                m.run_name(g,sid)

    def test_syntax_and_pbs_protocol(self):
        for p in PKG.rglob("*.py"):
            compile(p.read_text(),str(p),"exec")
        text=(PKG/"jobs/v8_weighted_controls.pbs").read_text()
        for part in ["#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1","#PBS -l walltime=24:00:00",
                     "--skip-test-eval","--workers 4","--model wireflynet",
                     'mkdir "$OUT/.weighted_control_lock"',"preflight_weighted_controls.py --verify"]:
            self.assertIn(part,text)
        self.assertNotIn("--mode eval",text)

    def test_prepare_verify_and_changed_input(self):
        m=load("preflight_weighted_controls")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            fixture(m,Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                m.prepare()
                plan=m.verify()
            self.assertEqual(len(plan["runs"]),30)
            self.assertEqual(len(plan["graphs"]),10)
            self.assertFalse(any("/test/" in p for p in plan["sha256"]))
            p=m.ROOT/"models/wireflynet.py"
            p.write_text(p.read_text()+"# changed\n")
            with self.assertRaisesRegex(RuntimeError,"differs"):
                m.verify()

    def test_existing_control_target_rejected(self):
        m=load("preflight_weighted_controls")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            fixture(m,Path(tmp))
            p=m.ROOT/"experiments_weighted_loss/wireflynet_rewired0_seed42/best.pt"
            p.parent.mkdir()
            p.write_text("do not touch")
            with self.assertRaisesRegex(RuntimeError,"already exists"):
                m.prepare()
            self.assertEqual(p.read_text(),"do not touch")

    def test_pilot_worker_mismatch_rejected(self):
        m=load("preflight_weighted_controls")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            fixture(m,Path(tmp))
            p=m.ROOT/"experiments_weighted_loss/wireflynet_bio_seed42/run_metadata.json"
            x=json.loads(p.read_text())
            x["resolved_arguments"]["workers"]=3
            p.write_text(json.dumps(x))
            with self.assertRaisesRegex(RuntimeError,"CLI/budget"):
                m.prepare()

    def test_successful_submission_is_recorded_and_duplicates_blocked(self):
        s=load("submit_weighted_controls")
        run=dict(run="wireflynet_random0_seed42",graph="random:0",seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            s.ROOT=Path(tmp)
            ledger=s.ROOT/"jobs.csv"
            result=SimpleNamespace(returncode=0,stdout="900.master\n",stderr="")
            with patch.object(s.subprocess,"run",return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(s.submit_one(run,True,ledger))
                with self.assertRaisesRegex(RuntimeError,"already has"):
                    s.submit_one(run,True,ledger)
            with ledger.open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["job_id"],"900.master")
            self.assertEqual(rows[0]["status"],"SUBMITTED")

    def test_interruption_and_rejection_require_review(self):
        s=load("submit_weighted_controls")
        run=dict(run="wireflynet_rewired1_seed43",graph="rewired:1",seed=43)
        for result in [OSError("interrupted"),SimpleNamespace(returncode=1,stdout="",stderr="queue limit")]:
            with tempfile.TemporaryDirectory() as tmp:
                s.ROOT=Path(tmp)
                ledger=s.ROOT/"jobs.csv"
                kwargs=dict(side_effect=result) if isinstance(result,Exception) else dict(return_value=result)
                with patch.object(s.subprocess,"run",**kwargs),contextlib.redirect_stderr(io.StringIO()):
                    self.assertFalse(s.submit_one(run,True,ledger))
                with ledger.open(newline="") as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),1)
                self.assertNotEqual(rows[0]["status"],"SUBMITTED")

if __name__=="__main__":
    unittest.main(verbosity=2)