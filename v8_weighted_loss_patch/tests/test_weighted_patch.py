#!/usr/bin/env python3
"""Standard-library tests; do not import torch or touch any HPC dataset."""
import ast
import copy
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

PKG = Path(__file__).resolve().parents[1]
SOURCE = PKG / "models/wireflynet.py"
TREE = ast.parse(SOURCE.read_text())

def extract_function(name, namespace):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Remove torch decorators for isolated pure/control-flow tests.
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]

def pure_helpers():
    ns = dict(Counter=Counter, NUM_CLS=3, TRAIN_HFLIP_P=0.5)
    return (extract_function("validate_balancing_config", ns),
            extract_function("training_loss_weights", ns))

class PatchTests(unittest.TestCase):
    def test_syntax_and_config(self):
        compile(SOURCE.read_text(), str(SOURCE), "exec")
        for p in PKG.rglob("*.py"):
            compile(p.read_text(), str(p), "exec")
        cfg = json.loads((PKG / "configs/wireflynet_g1_weighted_loss.json").read_text())
        self.assertFalse(cfg["eval_test"])
        self.assertEqual(cfg["train_sampler"], "shuffle")
        self.assertEqual(cfg["class_weighting"], "inverse_frequency")
        self.assertEqual(cfg["epochs"], 40)
        self.assertEqual(cfg["patience"], 12)
        self.assertTrue(cfg["exp_root"].endswith("/experiments_weighted_loss"))
        commands = (PKG / "jobs/weighted_loss_commands.txt").read_text().splitlines()
        commands = [c for c in commands if c.startswith("python3 ")]
        self.assertEqual(len(commands), 6)
        for c in commands:
            self.assertIn("--skip-test-eval", c)
            self.assertIn("wireflynet_g1_weighted_loss.json", c)
        self.assertEqual(sum("--graph bio " in c for c in commands), 3)
        self.assertEqual(sum("--graph none " in c for c in commands), 3)
        for seed in [42, 43, 44]:
            self.assertEqual(sum(f"--seed {seed} " in c for c in commands), 2)

    def test_original_defaults_and_double_weight_guard(self):
        validate, _ = pure_helpers()
        self.assertEqual(validate({}), ("balanced", "none"))
        self.assertEqual(validate({"train_sampler": "shuffle", "class_weighting": "inverse_frequency"}),
                         ("shuffle", "inverse_frequency"))
        with self.assertRaises(ValueError):
            validate({"train_sampler": "balanced", "class_weighting": "inverse_frequency"})
        for cfg in [{"train_sampler": "invalid"}, {"class_weighting": "invalid"}]:
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_weights_derive_from_training_labels_and_flip(self):
        _, weights_fn = pure_helpers()
        labels = [0] * 90 + [1] * 6 + [2] * 4
        weights, raw, expected = weights_fn(labels, "inverse_frequency")
        self.assertEqual(raw, [90, 6, 4])
        self.assertEqual(expected, [90.0, 5.0, 5.0])
        self.assertAlmostEqual(weights[1], weights[2])
        weighted_mass = [n * w for n, w in zip(expected, weights)]
        for mass in weighted_mass:
            self.assertAlmostEqual(mass, 100 / 3)
        no_flip_weights, _, counts = weights_fn(labels, "inverse_frequency", 0)
        self.assertEqual(counts, [90.0, 6.0, 4.0])
        self.assertNotEqual(no_flip_weights[1], no_flip_weights[2])
        self.assertIsNone(weights_fn(labels, "none")[0])
        with self.assertRaises(ValueError):
            weights_fn([0, 0], "inverse_frequency")
        with self.assertRaises(ValueError):
            weights_fn([0, 3], "inverse_frequency")

    def test_sampler_routing_with_mock_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = {}
            for split in ["train", "val", "test"]:
                p = root / split / "pairs.csv"
                p.parent.mkdir()
                p.write_text("prev_frame,curr_frame,label_id\np,c,0\n")
                counts[split] = {"manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            (root / "split_metadata.json").write_text(json.dumps({"counts": counts}))
            class FakeDataset:
                def __init__(self, split, augment, limit, **kwargs):
                    self.split = split
                def labels(self):
                    return [0, 0, 0, 1, 2]
                def __len__(self):
                    return 5
            def loader(ds, **kwargs):
                return SimpleNamespace(dataset=ds, kwargs=kwargs)
            ns = dict(
                CFG={}, json=json, DATASET_ROOT=root, IMG_SUBROOT=root / "images",
                MidAirDataset=FakeDataset, args=SimpleNamespace(batch=64, workers=0),
                _worker_init=lambda _: None, Counter=Counter,
                DataLoader=loader,
                WeightedRandomSampler=lambda weights, length, replacement: {
                    "weights": weights, "length": length, "replacement": replacement
                },
                TRAIN_SAMPLER="balanced"
            )
            make_loader = extract_function("make_loader", ns)
            balanced, _ = make_loader("train", augment=True)
            self.assertIn("sampler", balanced.kwargs)
            self.assertTrue(balanced.kwargs["sampler"]["replacement"])
            self.assertNotIn("shuffle", balanced.kwargs)
            ns["TRAIN_SAMPLER"] = "shuffle"
            shuffled, _ = make_loader("train", augment=True)
            self.assertTrue(shuffled.kwargs["shuffle"])
            self.assertNotIn("sampler", shuffled.kwargs)
            self.assertTrue(shuffled.kwargs["drop_last"])
            validation, _ = make_loader("val")
            self.assertFalse(validation.kwargs["shuffle"])
            self.assertNotIn("sampler", validation.kwargs)

    def test_eval_guard_blocks_before_model_or_data_loading(self):
        ns = dict(CFG={"eval_test": False}, args=SimpleNamespace(skip_test_eval=False))
        evaluate = extract_function("evaluate", ns)
        with self.assertRaisesRegex(RuntimeError, "validation-only"):
            evaluate()
        ns["CFG"] = {"eval_test": True}
        ns["args"].skip_test_eval = True
        with self.assertRaises(RuntimeError):
            evaluate()

    def test_model_implementations_and_selection_rule_unchanged(self):
        original = PKG / "original_source_for_test.py.txt"
        if original.is_file():
            old_tree = ast.parse(original.read_text())
            def definition(tree, name):
                return ast.dump(next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name),
                                include_attributes=False)
            for name in ["Retina", "Encoder", "WireFlyNet", "CNNMatched", "EMA"]:
                self.assertEqual(definition(old_tree, name), definition(TREE, name))
        text = SOURCE.read_text()
        self.assertIn('if vm["macro_f1"] > best:', text)
        self.assertIn('CFG.get("eval_test", True) and not args.skip_test_eval', text)
        self.assertIn('make_loader("test") if do_eval_test else (None, None)', text)
        self.assertIn('"run_metadata": run_metadata', text)
        self.assertIn('out / "validation_result.json"', text)

    def test_summary_tool_against_real_baseline_and_synthetic_followup(self):
        spec = importlib.util.spec_from_file_location("summary", PKG / "tools/summarize_weighted_validation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        real_baseline = Path("/data/raw/v8_diagnostics/experiments")
        if real_baseline.is_dir():
            rows, labels = mod.collect(real_baseline, "baseline")
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(labels), 13221)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, graph in [("wireflynet", "bio"), ("cnn_matched", "none")]:
                for seed in [42, 43, 44]:
                    name = f"{model}_{graph}_seed{seed}"
                    folder = root / name
                    folder.mkdir()
                    (folder / "val_predictions.csv").write_text("true,pred\n0,0\n1,1\n2,2\n")
                    metric, _ = mod.prediction_metrics(folder / "val_predictions.csv")
                    result = dict(run=name, seed=seed, model=model, graph=graph, params=1, best_epoch=1, val=metric)
                    (folder / "validation_result.json").write_text(json.dumps(result))
                    (folder / "history.csv").write_text("epoch,val_f1\n1,1.0\n2,0.8\n")
                    (folder / "run_metadata.json").write_text(json.dumps(dict(
                        train_sampler="shuffle", class_weighting="inverse_frequency", eval_test=False)))
            rows, labels = mod.collect(root, "weighted_loss")
            self.assertEqual(len(rows), 6)
            self.assertEqual(labels, [0, 1, 2])
            (root / "wireflynet_bio_seed42/test_predictions.csv").write_text("true,pred\n")
            with self.assertRaisesRegex(ValueError, "test outputs"):
                mod.collect(root, "weighted_loss")

if __name__ == "__main__":
    unittest.main(verbosity=2)