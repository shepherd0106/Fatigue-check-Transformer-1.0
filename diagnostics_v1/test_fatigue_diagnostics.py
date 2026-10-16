"""Pure metadata/metric/accounting tests; no research fits or dependency installs."""
import argparse
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import fatigue_diagnostics as d


def completed_fixture(folder,spec,marker):
    """Explicit fake binary/text receipts for metadata-only reuse guard tests."""
    names=["best.pt","last.pt","initial_base.pt","sampling_indices.npy","curves.json","lr_before_update.json","initialization.json","optimizer_groups.json","evaluations/best_reloaded.csv"]
    names += [f"evaluations/step{s:04d}_probe.csv" for s in range(64,spec["steps"]+1,64)]
    for name in names:
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"metadata fixture; no actual model")
    d.write(folder/"result.json",dict(state="completed",spec=spec,executed_steps=spec["steps"],tiny_fit_marker=marker,needs_review=not marker,numerical_and_eval_state_checks=True))


class PureChecks(unittest.TestCase):
    def test_plan_has_no_numpy_torch_or_signal_reads(self):
        command="import sys;import fatigue_diagnostics as d;p=d.plan();assert p['actual_optimizer_steps']==0;assert p['protocol']['total_step_cap']==16384;assert not {'numpy','torch','scipy','mamba_ssm'}&sys.modules.keys()"
        done=subprocess.run([sys.executable,"-B","-c",command],cwd=d.HERE,capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
    def test_schedule_before_update_and_matching_budget(self):
        self.assertEqual(d.learning_rate("cosine",1),.001)
        self.assertAlmostEqual(d.learning_rate("cosine",3840),.0001,places=16)
        values=[d.learning_rate("cosine",k) for k in range(1,3841)]
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertTrue(all(d.learning_rate("constant",k)==.001 for k in (1,64,3840)))
        self.assertEqual(sum(s["steps"] for s in d.run_specs()),16384)
        for k in (0,3841):
            with self.assertRaises(ValueError):d.learning_rate("cosine",k)
    def test_virtual_patience_keeps_first_stop_and_continues_best(self):
        state=dict(best=None,best_step=None,stale=0,virtual_stop_step=None)
        for epoch in range(1,12):state=d.virtual_update(state,epoch*64,.5)
        self.assertEqual(state["virtual_stop_step"],704)
        state=d.virtual_update(state,768,.6)
        self.assertEqual((state["best_step"],state["virtual_stop_step"]),(768,704))
        improved=d.virtual_update(state,832,.60009)
        self.assertEqual(improved["best_step"],768)
    def test_stratified_lists_numeric_order_and_single_class_fallback(self):
        windows=[]
        for sid in ("10","2","1","3","4","5"):
            for label in (0,1):
                for j in range(25 if sid!="5" else (40 if label==0 else 0)):
                    windows.append(SimpleNamespace(paired_valid=True,subject_id=sid,label_2class=label,source_index=len(windows)))
        windows.append(SimpleNamespace(paired_valid=True,subject_id="99",label_2class=0,source_index=999))
        dataset=SimpleNamespace(windows=windows);selected=d.select_lists(dataset,["1","2","3","4","5","10"])
        self.assertEqual(selected["tiny_subjects"],["1","2","3","4"])
        self.assertEqual(len(selected["tiny"]),128);self.assertEqual(len(selected["probe"]),192)
        self.assertTrue(all(windows[i].subject_id!="99" for k in ("tiny","probe") for i in selected[k]))
        self.assertEqual(sum(windows[i].subject_id=="5" for i in selected["probe"]),32)
    def test_metrics_ce_and_support_equal_people_not_batches(self):
        rows=[dict(subject_id="a",label=0,logits=[0,0])]+[dict(subject_id="b",label=1,logits=[0,math.log(3)])]*3
        result=d.classification_metrics(rows)
        self.assertEqual(result["primary"],1)
        self.assertEqual(result["per_subject"]["a"]["recall"],[1,None])
        self.assertAlmostEqual(result["window_equal_ce"],(math.log(2)+3*math.log(4/3))/4)
        self.assertAlmostEqual(result["subject_equal_ce"],(math.log(2)+math.log(4/3))/2)
        self.assertEqual(result["predicted_fatigue_fraction"],.75)
        with self.assertRaises(ValueError):d.classification_metrics([dict(subject_id="x",label=0,logits=[float("nan"),0])])
    def test_ledger_failure_reserves_before_step_and_no_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=d.Budget(root,{"fixture":True});spec=d.run_specs()[0]
            ledger.begin(spec);ledger.reserve(spec["key"]);ledger.finish(spec["key"],"failed","simulated error after reserve")
            other=d.Budget(root,{"fixture":True})
            self.assertEqual((other.state["runs"],other.state["steps"]),(1,1))
            with self.assertRaises(ValueError):other.begin(spec)
            with self.assertRaises(ValueError):d.Budget(root,{"fixture":False})
            self.assertEqual(d.read(root/"ledger.json")["steps"],1)
    def test_phase_and_slot_caps_exact_not_free_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=d.Budget(root,{})
            for spec in d.run_specs():
                ledger.begin(spec)
                # Set to final allowed reservation to avoid thousands of writes.
                remaining=spec["steps"]-1;ledger.state["steps"]+=remaining
                ledger.state["phases"][spec["phase"]]["steps"]+=remaining
                ledger.state["slots"][spec["key"]]["steps"]=remaining
                ledger.reserve(spec["key"])
                with self.assertRaises(ValueError):ledger.reserve(spec["key"])
                ledger.finish(spec["key"],"completed")
            self.assertEqual((ledger.state["runs"],ledger.state["steps"]),(6,16384))
            with self.assertRaises(ValueError):ledger.begin(dict(d.run_specs()[0],key="invented"))
    def test_existing_writer_lock_is_never_removed_by_failed_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with d.Writer(root):
                with self.assertRaises(ValueError):
                    with d.Writer(root):pass
                self.assertTrue((root/"writer.lock").exists())
            self.assertFalse((root/"writer.lock").exists())
    def test_completed_reuse_checks_all_hashes_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=d.run_specs()[0];folder=root/"runs"/spec["key"];folder.mkdir(parents=True)
            binding={"fixture":1};completed_fixture(folder,spec,True)
            d.write(folder/"status.json",dict(state="completed",spec=spec,binding=binding,artifacts=d.artifact_hashes(folder)))
            self.assertTrue(d.verify_completed(root,spec,binding)["tiny_fit_marker"])
            with self.assertRaises(ValueError):d.verify_completed(root,spec,{"fixture":2})
            d.write(folder/"result.json",{"tiny_fit_marker":False})
            with self.assertRaises(ValueError):d.verify_completed(root,spec,binding)
    def test_run_rejects_missing_execute_before_heavy_import(self):
        with self.assertRaisesRegex(ValueError,"explicit --execute"):d.run(SimpleNamespace(execute=False))
    def test_protected_output_and_unrelated_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);baseline=parent/"baseline";preflight=parent/"e0/receipt.json"
            for path in (parent,baseline,baseline/"nested",preflight.parent,d.HERE/"nested"):
                with self.subTest(path=path),self.assertRaises(ValueError):d.validate_output(path,baseline,preflight)
            unrelated=parent/"unrelated";unrelated.mkdir();(unrelated/"unique.txt").write_text("retain",encoding="utf-8")
            with self.assertRaises(ValueError):d.Budget(unrelated,{})
            self.assertEqual((unrelated/"unique.txt").read_text(encoding="utf-8"),"retain")
    def test_tiny_review_binds_actual_results_no_prefilled_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binding={"fixture":1};ledger=d.Budget(root,binding)
            for spec in d.run_specs()[:2]:
                ledger.begin(spec);folder=root/"runs"/spec["key"];folder.mkdir(parents=True)
                ledger.state["slots"][spec["key"]]["steps"]=512
                ledger.state["steps"]+=512;ledger.state["phases"]["tiny"]["steps"]+=512
                completed_fixture(folder,spec,False)
                d.write(folder/"status.json",dict(state="completed",spec=spec,binding=binding,artifacts=d.artifact_hashes(folder)))
                ledger.finish(spec["key"],"completed")
            with self.assertRaises(ValueError):d.tiny_gate(root,ledger,binding)
            review=dict(reviewed_by="test actual review",reviewed_at="2026-10-15",reason="fixture",allow_pilot=True,binding_sha256=d.digest(binding),
                tiny_result_sha256={s["key"]:d.sha(root/"runs"/s["key"]/"result.json") for s in d.run_specs()[:2]})
            d.write(root/"review.json",review);d.tiny_gate(root,ledger,binding,root/"review.json")
            review["tiny_result_sha256"]={};d.write(root/"badreview.json",review)
            with self.assertRaises(ValueError):d.tiny_gate(root,ledger,binding,root/"badreview.json")

    def test_corrupt_or_missing_ledger_never_resets_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=d.Budget(root,{})
            original=copy.deepcopy(ledger.state);bad=copy.deepcopy(original);bad["steps"]=1;d.write(root/"ledger.json",bad)
            with self.assertRaises(ValueError):d.Budget(root,{})
            d.write(root/"ledger.json",original);ledger=d.Budget(root,{})
            ledger.begin(d.run_specs()[0])
            with self.assertRaises(ValueError):ledger.finish(d.run_specs()[0]["key"],"completed")
            (root/"ledger.json").unlink();d.write(root/"prepared.json",{"binding":{}})
            with self.assertRaises(ValueError):d.Budget(root,{})
            with self.assertRaises(ValueError):d.status_report(root)
            self.assertFalse((root/"ledger.json").exists())

    def test_eval_csv_independently_reconstructs_metrics(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            rows=[dict(source_index=i,subject_id="1",run_id="a",window_in_run=i,block_info="block1",label=i%2,logits=[.1,float(i)]) for i in range(3)]
            path=Path(tmp)/"eval.csv";d.write_evaluation(path,rows,d.run_specs()[2],64,{"baseline":{"data_fingerprint":"fixture"}})
            with path.open(encoding="utf-8",newline="") as stream:saved=list(csv.DictReader(stream))
            reconstructed=[dict(subject_id=r["subject_id"],label=int(r["label"]),logits=[float(r["logits0"]),float(r["logits1"])]) for r in saved]
            self.assertEqual(d.classification_metrics(rows),d.classification_metrics(reconstructed))
            self.assertEqual([int(r["source_index"]) for r in saved],[0,1,2])

    def test_real_prepare_reads_only_metadata_and_is_idempotent(self):
        project=d.HERE.parents[1]
        baseline=project/"work/D2_audit_20261015/dev-v2"
        manifest=project/"outputs/真实数据试验/M2_outer1_20261012/时间轴v2预览/fatigue_time_axis_v2_preview/manifest.json"
        preflight=project/"work/D2_audit_20261015/dev-E0/receipt.json"
        if not all(p.exists() for p in (baseline,manifest,preflight)):self.skipTest("local audit metadata fixtures absent")
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output=str(Path(tmp)/"new"),baseline=str(baseline),manifest=str(manifest),preflight=str(preflight))
            with patch("numpy.load",side_effect=AssertionError("array access in prepare")),patch("fatigue_training.data.WindowDataset.fingerprint",side_effect=AssertionError("signal hashing in prepare")):
                first=d.prepare(args);second=d.prepare(args)
            self.assertEqual(first,second)
            self.assertEqual(first["selection_counts"]["tiny"],128);self.assertEqual(first["selection_counts"]["probe"],576)
            self.assertFalse(first["full_data_fingerprint_verified"])
            self.assertEqual(d.read(Path(args.output)/"ledger.json")["steps"],0)

    def test_export_has_index_and_refuses_active_writer(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/"diag";root.mkdir()
            binding=dict(source=d.source_binding(),baseline_root=str(parent/"old"),preflight_path=str(parent/"e0/receipt.json"))
            d.Budget(root,binding);d.write(root/"prepared.json",dict(binding=binding))
            target=parent/"audit.zip"
            with d.Writer(root),self.assertRaises(ValueError),patch("sys.stdout",new_callable=io.StringIO):
                d.main(["export","--output",str(root),"--zip",str(target)])
            self.assertFalse(target.exists())
            with patch("sys.stdout",new_callable=io.StringIO):d.main(["export","--output",str(root),"--zip",str(target)])
            with zipfile.ZipFile(target) as archive:
                index=json.loads(archive.read("AUDIT_INDEX.json"))
                for name,h in index["files"].items():self.assertEqual(d.hashlib.sha256(archive.read(name)).hexdigest(),h)


@unittest.skipUnless(importlib.util.find_spec("torch"),"Torch unavailable: no runtime evaluation/state verification claimed")
class TorchChecks(unittest.TestCase):
    def test_eval_preserves_mixed_module_modes_buffers_and_rng(self):
        import torch
        import numpy as np
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm1d(2);self.drop=torch.nn.Dropout(.3)
            def forward(self,e,f):return self.drop(self.bn(e))
        class Dataset:
            def __init__(self):self.windows=[SimpleNamespace(subject_id="1",paired_valid=True,source_index=i,run_id="run",window_in_run=i,block_info="block1") for i in range(3)]
            def batch(self,indices,stats):return np.ones((len(indices),2),dtype=np.float32),np.ones((len(indices),2)),np.zeros(len(indices),dtype=np.int64)
        model=Model().train();model.drop.eval();before=d.rng_snapshot(torch,"cpu")
        result,rows=d.evaluate_fixed(model,Dataset(),[0,1,2],{},["1"],"cpu")
        self.assertTrue(model.bn.training);self.assertFalse(model.drop.training)
        self.assertTrue(d.rng_equal(torch,before,d.rng_snapshot(torch,"cpu")))
        self.assertEqual(result["n"],3)


if __name__=="__main__":unittest.main()
