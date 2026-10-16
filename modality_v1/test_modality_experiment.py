"""Metadata, budget and active-only guard tests. No research fits or installs."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import modality_experiment as m


class PureChecks(unittest.TestCase):
    def test_plan_stdlib_only_and_independent_parameter_arithmetic(self):
        cmd="import sys;import modality_experiment as m;p=m.plan();assert p['protocol']['total_step_cap']==31752;assert p['parameter_arithmetic']['C_E']['trainable']==1802;assert p['parameter_arithmetic']['C_F']['trainable']==1634;assert not {'torch','numpy','scipy','mamba_ssm','modality_models'}&sys.modules.keys()"
        done=subprocess.run([sys.executable,"-B","-c",cmd],cwd=m.HERE,capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(sum(s["max_steps"] for s in m.specs()),31752)
        self.assertEqual(len(m.specs()),12)
    def test_research_completion_semantics_differ_from_preflight_and_tiny(self):
        for spec in m.specs():
            if spec["phase"]=="research":
                self.assertTrue(m.valid_completed_steps(spec,704));self.assertTrue(m.valid_completed_steps(spec,3840))
                for n in (0,65,3841):self.assertFalse(m.valid_completed_steps(spec,n))
            else:
                self.assertTrue(m.valid_completed_steps(spec,spec["max_steps"]))
                self.assertFalse(m.valid_completed_steps(spec,spec["max_steps"]-1))
    def test_budget_full_caps_metadata_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=m.Budget(Path(tmp),{"fixture":True})
            for spec in m.specs():
                b.begin(spec);n=spec["max_steps"]-1
                b.state["slots"][spec["key"]]["steps"]=n;b.state["steps"]+=n;b.state["phases"][spec["phase"]]["steps"]+=n
                b.reserve(spec["key"])
                with self.assertRaises(ValueError):b.reserve(spec["key"])
                b.finish(spec["key"],"completed")
            self.assertEqual((b.state["runs"],b.state["steps"]),(12,31752))
            m.Budget(Path(tmp),{"fixture":True}).validate()
            with self.assertRaises(ValueError):b.begin(m.specs()[0])
    def test_failures_keep_reservations_and_missing_or_corrupt_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=m.Budget(root,{});s=m.specs()[0];b.begin(s);b.reserve(s["key"]);b.finish(s["key"],"failed","fixture")
            other=m.Budget(root,{})
            self.assertEqual(other.state["steps"],1)
            with self.assertRaises(ValueError):other.begin(s)
            bad=copy.deepcopy(other.state);bad["steps"]=0;m.write(root/"ledger.json",bad)
            with self.assertRaises(ValueError):m.Budget(root,{})
            (root/"ledger.json").unlink();m.write(root/"prepared.json",{"binding":{}})
            with self.assertRaises(ValueError):m.Budget(root,{})
            with self.assertRaises(ValueError):m.status_report(root)
            self.assertFalse((root/"ledger.json").exists())
    def test_completed_research_can_finish_early_without_freeing_reserved_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=m.Budget(root,{});s=next(s for s in m.specs() if s["phase"]=="research")
            b.begin(s);b.state["slots"][s["key"]]["steps"]=704;b.state["steps"]=704;b.state["phases"]["research"]["steps"]=704;b.finish(s["key"],"completed")
            self.assertEqual(m.Budget(root,{}).state["steps"],704)
            with self.assertRaises(ValueError):b.begin(s)
    def test_run_needs_execute_before_importing_heavy_dependencies(self):
        with self.assertRaisesRegex(ValueError,"explicit --execute"):m.run(SimpleNamespace(execute=False))
    def test_completed_result_must_match_ledger_step_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=m.specs()[0];folder=root/"runs"/spec["key"];folder.mkdir(parents=True)
            for name in ("initialization.json","optimizer_groups.json","repeat0.pt","repeat1.pt","repeat_comparison.json","shape_checks.json"):
                (folder/name).write_bytes(b"metadata fixture; no model execution")
            m.write(folder/"result.json",dict(state="completed",spec=spec,executed_steps=4,numerical_and_eval_state_checks=True))
            m.write(folder/"status.json",dict(state="completed",spec=spec,binding={},artifacts=m.old.artifact_hashes(folder)))
            self.assertEqual(m.verify_completed(root,spec,{},4)["executed_steps"],4)
            with self.assertRaisesRegex(ValueError,"ledger reservations"):m.verify_completed(root,spec,{},3)
    def test_protected_output_roots_and_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);baseline=p/"old";preflight=p/"e0/receipt.json"
            for target in (p,baseline,baseline/"nested",preflight.parent,m.HERE/"out",Path("/root/fatigue-diagnostics-v1/new")):
                with self.subTest(target=target),self.assertRaises(ValueError):m.validate_output(target,baseline,preflight)
    def test_aggregate_keeps_unsupported_class_na_and_ef_ce_na(self):
        people={"a":dict(n=2,primary=1,recall=[1,None],predicted_fatigue_fraction=0,ce=.1),"b":dict(n=4,primary=.5,recall=[0,1],predicted_fatigue_fraction=1,ce=.8)}
        result=m.aggregate_people(people,True)
        self.assertEqual(result["primary"],.75);self.assertEqual(result["recall"],[.5,1]);self.assertEqual(result["fatigue_supported_subjects"],1)
        self.assertAlmostEqual(result["subject_equal_ce"],.45);self.assertAlmostEqual(result["window_equal_ce"],3.4/6)
        self.assertIsNone(m.aggregate_people(people,False)["subject_equal_ce"])
    def test_csv_uses_actual_fold_and_inactive_modality_not_in_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            import csv
            spec=next(s for s in m.specs() if s["phase"]=="research" and s["fold"]==4)
            row=dict(source_index=7,subject_id="1",run_id="r",window_in_run=2,block_info="block1",label=1,logits=[0,1])
            path=Path(tmp)/"rows.csv";m.write_predictions(path,[row],spec,64)
            with path.open(encoding="utf-8",newline="") as stream:records=list(csv.DictReader(stream))
            self.assertEqual(records[0]["fold"],"4");self.assertEqual(records[0]["seed"],"17");self.assertEqual(records[0]["prediction"],"1")
    def test_active_batch_never_calls_dual_batch_or_inactive_slice(self):
        import numpy as np
        class Spy:
            def __init__(self):
                self.windows=[SimpleNamespace(subject_id="1",paired_valid=True,label_2class=0),SimpleNamespace(subject_id="2",paired_valid=True,label_2class=1)];self.manifest={};self.calls=[]
            def batch(self,*args):raise AssertionError("Dual batch must never be used")
            def slice(self,w,modality,start=None,end=None):
                self.calls.append((w.subject_id,modality));return np.full((4,5),3,dtype=np.float32)
        data=Spy();stats={"modalities":{"eeg":{"mean":[1]*4,"std":[2]*4}}}
        x,y=m.active_batch(data,[0],stats,["1"],"eeg")
        self.assertTrue(np.array_equal(x,np.ones((1,4,5),dtype=np.float32)));self.assertEqual(data.calls,[("1","eeg")])
        with self.assertRaises(ValueError):m.active_batch(data,[1],stats,["1"],"eeg")
        self.assertEqual(data.calls,[("1","eeg")])
    def test_real_prepare_four_folds_metadata_only_and_idempotent(self):
        project=m.HERE.parents[1];baseline=project/"work/D2_audit_20261015/dev-v2";preflight=project/"work/D2_audit_20261015/dev-E0/receipt.json"
        manifest=project/"outputs/真实数据试验/M2_outer1_20261012/时间轴v2预览/fatigue_time_axis_v2_preview/manifest.json"
        if not all(p.exists() for p in (baseline,preflight,manifest)):self.skipTest("Original local metadata fixtures absent")
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output=str(Path(tmp)/"new"),baseline=str(baseline),preflight=str(preflight),manifest=str(manifest))
            with patch("numpy.load",side_effect=AssertionError("array load in metadata prepare")),patch("fatigue_training.data.WindowDataset.fingerprint",side_effect=AssertionError("signal hash in prepare")):
                first=m.prepare(args);second=m.prepare(args)
            self.assertEqual(first,second);self.assertEqual(set(first["counts"]),{"2","3","4","5"})
            self.assertTrue(all(v["probe"]==576 for v in first["counts"].values()));self.assertEqual(first["counts"]["2"]["tiny"],128)
            self.assertEqual(m.read(Path(args.output)/"ledger.json")["steps"],0)
            report=m.status_report(args.output);self.assertEqual(report["models"]["C_EF"]["summary"]["subjects"],24)
            self.assertIsNone(report["models"]["C_EF"]["summary"]["subject_equal_ce"])
            self.assertAlmostEqual(report["models"]["C_EF"]["summary"]["primary"],.5579603477204764)
            self.assertEqual(report["models"]["C_EF"]["summary"]["fatigue_supported_subjects"],23)
            self.assertFalse(report["complete_four_fold_ranking_ready"])
            # Metadata-only simulated completion: ensure partial folds are never
            # presented as a comparable 24-person single-modality summary.
            root=Path(args.output);binding=first["binding"];spec=next(s for s in m.specs() if s["phase"]=="research")
            budget=m.Budget(root,binding);budget.begin(spec)
            budget.state["slots"][spec["key"]]["steps"]=704;budget.state["steps"]=704;budget.state["phases"]["research"]["steps"]=704;budget.finish(spec["key"],"completed")
            ref=m.read(root/"references.json")["2"]["result"]["evaluation"]["per_subject"]
            people={sid:dict(n=r["n"],primary=r["support_recall"],recall=r["recall"],predicted_fatigue_fraction=.5,ce=.7) for sid,r in ref.items()}
            with patch.object(m,"verify_completed",return_value=dict(best_reloaded_metrics={"per_subject":people})):
                partial=m.status_report(root)
            self.assertIsNone(partial["models"]["C_E"]["summary"])
            self.assertEqual(partial["models"]["C_E"]["partial_summary"]["subjects"],6)
            self.assertEqual(partial["matched_folds_summary"]["C_E"]["C_EF"]["subjects"],6)
            self.assertEqual(partial["matched_folds_summary"]["C_E"]["folds"],[2])
            self.assertFalse(partial["complete_four_fold_ranking_ready"])
    def test_export_and_report_refuse_active_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new";root.mkdir()
            with WriterAlias(root):
                for command in ("report","export"):
                    args=[command,"--output",str(root)]+(["--zip",str(Path(tmp)/"out.zip")] if command=="export" else [])
                    with self.subTest(command=command),self.assertRaises(ValueError):m.main(args)


WriterAlias=m.Writer


@unittest.skipUnless(importlib.util.find_spec("torch"),"Torch unavailable; actual single-modality model/preflight/GPU checks pending")
class TorchChecks(unittest.TestCase):
    def test_only_active_parameters_and_exact_head_slice_are_registered(self):
        import torch
        from modality_models import ModalityModel
        from fatigue_training.development_models import DevelopmentModel,optimizer_groups
        reference=DevelopmentModel("C_EF",17,2)
        for variant,active,start in (("C_E","eeg",0),("C_F","fnirs",32)):
            model=ModalityModel(variant);self.assertTrue(torch.equal(model.head.weight,reference.head.weight[:,start:start+32]));self.assertTrue(torch.equal(model.head.bias,reference.head.bias))
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),m.static_counts()[variant]["trainable"])
            self.assertFalse(hasattr(model,"fnirs" if active=="eeg" else "eeg"))
            self.assertFalse(model.branch.front.bn1.bias.requires_grad)
            groups,names=optimizer_groups(model);self.assertTrue(all(active in n or n.startswith("head.") for n in names["decay"]+names["no_decay"]))
            shape=(2,4,2500) if active=="eeg" else (2,24,250);x=torch.ones(shape)
            model.eval();self.assertEqual(tuple(m.forward(model,x).shape),(2,2))
            with self.assertRaises(ValueError):model(eeg=x,fnirs=x)
    def test_eval_preserves_buffers_rng_and_mixed_modes(self):
        import numpy as np
        import torch
        from modality_models import ModalityModel
        class Data:
            def __init__(self):self.windows=[SimpleNamespace(subject_id="1",paired_valid=True,label_2class=0,source_index=i,run_id="r",window_in_run=i,block_info="block1") for i in range(2)];self.manifest={}
            def slice(self,w,modality,start=None,end=None):
                if modality!="eeg":raise AssertionError("inactive signal accessed")
                return np.ones((4,2500),dtype=np.float32)
        model=ModalityModel("C_E").train();model.dropout.eval();before=m.old.rng_snapshot(torch,"cpu")
        metrics,rows=m.evaluate_fixed(model,Data(),[0,1],{"modalities":{"eeg":{"mean":[0]*4,"std":[1]*4}}},["1"],"cpu")
        self.assertTrue(model.branch.training);self.assertFalse(model.dropout.training);self.assertTrue(m.old.rng_equal(torch,before,m.old.rng_snapshot(torch,"cpu")));self.assertEqual(len(rows),2)


if __name__=="__main__":unittest.main()
