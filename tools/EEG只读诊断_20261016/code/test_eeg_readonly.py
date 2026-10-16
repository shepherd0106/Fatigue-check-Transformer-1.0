"""Stdlib-only safety and arithmetic checks; never substitutes for CUDA replay."""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import types
import sys
import zipfile

import eeg_readonly as d


class ReadonlyTests(unittest.TestCase):
    def test_output_rejects_ancestors_descendants_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "old"
            evidence.mkdir()
            evidence.joinpath("weight.pt").write_bytes(b"preserve")
            before = d.sha(evidence / "weight.pt")
            for target in (root, evidence, evidence / "new"):
                with self.assertRaises(ValueError):
                    d.validate_output(target, [evidence])
            existing = root / "existing_empty"
            existing.mkdir()
            with self.assertRaises(ValueError):
                d.validate_output(existing, [evidence])
            self.assertEqual(d.validate_output(root / "fresh", [evidence]), (root / "fresh").resolve())
            self.assertEqual(d.sha(evidence / "weight.pt"), before)

    def test_child_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                d.child(temp, "../escape")

    def test_execution_prefix_excludes_suffix_even_invalid_suffix(self):
        stream = [[1] * 64, [2] * 64, [999] * 64]
        counts = d.executed_counts(stream, 2, [1, 2, 3])
        self.assertEqual(counts, {1: 64, 2: 64})
        self.assertNotIn(999, counts)
        with self.assertRaises(ValueError):
            d.executed_counts(stream, 3, [1, 2, 3])

    def test_invalid_prefix_and_batch(self):
        for steps in (0, -1, 2, True, 1.0):
            with self.assertRaises(ValueError):
                d.executed_counts([[1] * 64], steps, [1])
        with self.assertRaises(ValueError):
            d.executed_counts([[1] * 63], 1, [1])
        with self.assertRaises(ValueError):
            d.executed_counts([[1.5] * 64], 1, [1])

    def test_coverage_denominators_zero_probe_and_people(self):
        counts = d.Counter({1: 3, 2: 1})
        result = d.coverage(counts, [1, 2, 3, 4], [1, 3, 4], lambda i: "1" if i < 3 else "3")
        self.assertEqual(result["total"]["unique_window_coverage"], .5)
        self.assertEqual(result["total"]["probe_exposure"], dict(count=3, min=0, median=0, max=3, zero=2))
        self.assertEqual(result["per_subject"]["3"]["draws"], 0)
        self.assertEqual(result["per_subject"]["1"]["unique_sampled"], 2)

    def test_exclusive_json_write_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "evidence.json"
            d.write_new(target, {"old": 1})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                d.write_new(target, {"replacement": 2})
            self.assertEqual(target.read_bytes(), before)

    def predictions(self, path, a=1., b=2., source_index="7", step="64"):
        row = dict(model="C_E", fold="2", seed="17", step=step, source_index=source_index,
                   subject_id="1", run_id="101", window_in_run="0", block_info="block1", label="1", logits0=a, logits1=b)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return [dict(source_index=7, subject_id="1", run_id="101", window_in_run=0,
                     block_info="block1", label=1, logits=[1., 2.])]

    def test_reproduction_tolerance_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "old.csv"
            rows = self.predictions(target, a=1.00005)
            before = target.read_bytes()
            self.assertTrue(d.compare_rows(rows, target, 64)["passed"])
            self.assertEqual(target.read_bytes(), before)
            self.predictions(target, a=1.01)
            with self.assertRaises(ValueError):
                d.compare_rows(rows, target, 64)
            self.predictions(target, source_index="8")
            with self.assertRaises(ValueError):
                d.compare_rows(rows, target, 64)
            self.predictions(target, step="128")
            with self.assertRaises(ValueError):
                d.compare_rows(rows, target, 64)

    def test_nonfinite_reference_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "old.csv"
            rows = self.predictions(target, a="nan")
            with self.assertRaises(ValueError):
                d.compare_rows(rows, target, 64)

    def test_fixed_data_checksum_scope_and_mutation(self):
        class Guard:
            def __init__(self, dataset, allowed):
                self.dataset = dataset
                self.allowed = allowed
            def _check(self, indices):
                if any(self.dataset.windows[i].subject_id not in self.allowed for i in indices):
                    raise ValueError("Forbidden before file access")
                return indices
        module = types.ModuleType("fatigue_training.development")
        module.SubjectGuard = Guard
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(sys.modules, {"fatigue_training.development": module}):
            root = Path(temp)
            root.joinpath("train.npy").write_bytes(b"read-only training bytes")
            # A nonexistent withheld file proves that the routine never accesses it.
            dataset = types.SimpleNamespace(root=root, windows=[types.SimpleNamespace(run_id="a", subject_id="1"), types.SimpleNamespace(run_id="b", subject_id="2")],
                runs={"a": {"eeg": {"array": "train.npy"}}, "b": {"eeg": {"array": "FORBIDDEN-MISSING.npy"}}})
            target = root / "checksums.json"
            d.write_new(target, [dict(path="train.npy", bytes=(root / "train.npy").stat().st_size, sha256=d.sha(root / "train.npy"))])
            with mock.patch.object(d, "DATA_CHECKSUMS_SHA256", d.sha(target)):
                report = d.data_snapshot(dataset, [0], ["1"])
                self.assertEqual(len(report), 2)
                with self.assertRaises(ValueError):
                    d.data_snapshot(dataset, [1], ["1"])
                root.joinpath("train.npy").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    d.data_snapshot(dataset, [0], ["1"])

    def test_export_contains_only_new_artifacts_and_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            d.write_new(root / "failure.json", dict(state="failed"))
            d.export_audit(root)
            with zipfile.ZipFile(root / "eeg_readonly_audit.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"failure.json", "audit_index.json"})
                self.assertEqual(json.loads(archive.read("audit_index.json"))["failure.json"], d.sha(root / "failure.json"))

    def test_no_training_calls_and_safe_checkpoint_loader(self):
        tree = ast.parse(Path(d.__file__).read_text(encoding="utf-8"))
        forbidden = {"backward", "step", "zero_grad", "optimizer", "training_one", "execute_one", "run", "prepare", "save", "train", "fingerprint"}
        loads = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                self.assertNotIn(name, forbidden)
                if name == "update" and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(ast.unparse(node.func.value), ("m", "model", "opt", "torch"))
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and name == "load":
                    loads.append(node)
                    self.assertTrue(any(k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords))
        self.assertEqual(len(loads), 1)
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module and "optim" in n.module for n in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
