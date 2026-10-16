"""Protocol/isolation/accounting checks need numpy only; actual GPU tests are explicit."""
import contextlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from fatigue_training.common import digest,file_hash,read_json,write_json
from fatigue_training.development import (MODELS,COUNTS,WITHHELD,CAPS,Task,Ledger,SubjectGuard,
    configuration,plan,tasks,subjects,bin_boundaries,disjoint_bin_means,choose_lrs,
    diagnostic_gate,sample_stream,artifact_reuse,interaction_bootstrap)
from fatigue_training.development_features import (fit_normalization,fit_feature_scaler,apply_scaler,
    subject_weights,signal_features)
from fatigue_training.development_training import prediction_rows,record_repair,reviewed_source

TEST_TMP=Path(__file__).parent/".tmp";TEST_TMP.mkdir(exist_ok=True)


@unittest.skipUnless(importlib.util.find_spec("torch"),"torch unavailable; no frozen-bias runtime verification claimed")
class FrozenBiasChecks(unittest.TestCase):
    def test_cnn_freezes_zero_bias_preserves_gamma_and_checkpoint_keys(self):
        import torch
        from fatigue_training.development_models import DevelopmentModel,optimizer_groups
        model=DevelopmentModel("C_EF")
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),COUNTS["C_EF"])
        groups,names=optimizer_groups(model)
        for name,branch in (("eeg",model.eeg),("fnirs",model.fnirs)):
            bias=branch.front.bn1.bias
            self.assertFalse(bias.requires_grad);self.assertEqual(int(torch.count_nonzero(bias)),0)
            self.assertTrue(branch.front.bn1.weight.requires_grad)
            self.assertIn(name+".front.bn1.bias",model.state_dict())
            self.assertNotIn(name+".front.bn1.bias",names["decay"]+names["no_decay"])
            self.assertFalse(any(p is bias for g in groups for p in g["params"]))
        with torch.no_grad():
            eeg=torch.randn(2,4,2500);fnirs=torch.randn(2,24,250)
        model.train();model(eeg,fnirs).sum().backward()
        self.assertTrue(all(branch.front.bn1.bias.grad is None for branch in (model.eeg,model.fnirs)))
        state=model.state_dict();model.load_state_dict(state)
        state["eeg.front.bn1.bias"]=torch.ones_like(state["eeg.front.bn1.bias"])
        with self.assertRaises(ValueError):model.load_state_dict(state)


class FakeData:
    """In-memory spy: forbidden subjects fail before any signal access."""
    def __init__(self):
        self.windows=[];self.accesses=[]
        self.manifest=dict(kind="synthetic",data_version="spy")
        for group in configuration()["outer_groups"]:
            for sid in group:
                for j in range(2 if sid%2 else 3):
                    self.windows.append(SimpleNamespace(subject_id=str(sid),paired_valid=True,
                        source_index=1000+len(self.windows),run_id=str(sid),block_info="block1",
                        window_in_run=j,label_2class=j%2,eeg_start=j*1250,eeg_end=j*1250+2500,
                        fnirs_start=j*125,fnirs_end=j*125+250))
    def indices(self,people,nonoverlap=False):
        people=set(map(str,people))
        return [i for i,w in enumerate(self.windows) if w.subject_id in people and (not nonoverlap or w.window_in_run%2==0)]
    def fingerprint(self):return "spy_fingerprint"
    def check_unchanged(self):pass
    def slice(self,w,modality,start=None,end=None):
        if w.subject_id in WITHHELD:raise AssertionError("Forbidden data actually read")
        self.accesses.append(w.subject_id)
        start=getattr(w,modality+"_start") if start is None else start
        end=getattr(w,modality+"_end") if end is None else end
        rows=4 if modality=="eeg" else 24
        x=np.tile((np.arange(start,end,dtype=np.float32)/100+int(w.subject_id)),(rows,1))
        x[-1]=np.float32(7.1)
        return x
    def batch(self,indices,stats):
        windows=[self.windows[int(i)] for i in indices]
        if any(w.subject_id in WITHHELD for w in windows):raise AssertionError("Forbidden batch reached data")
        self.accesses.extend(w.subject_id for w in windows)
        out=[]
        for modality in ("eeg","fnirs"):
            st=stats["modalities"][modality]
            out.append(np.stack([((self.slice(w,modality).astype(np.float64)-np.array(st["mean"])[:,None])/np.array(st["std"])[:,None]).astype(np.float32) for w in windows]))
        return (*out,np.array([w.label_2class for w in windows],dtype=np.int64))


def completed_tuning(primary=.7,fatigue=.5):
    return {t.key:dict(state="completed",primary=primary,fatigue_recall=fatigue) for t in tasks("D1")}


class ProtocolChecks(unittest.TestCase):
    def test_legacy_and_new_routes_budgets(self):
        from fatigue_training.protocol import load_config,plan as old_plan
        self.assertEqual(old_plan(load_config())["fit_count"],945)
        p=plan();self.assertEqual(p["fit_count"],131);self.assertEqual(p["research_fit_count"],116)
        self.assertEqual([len(tasks(s)) for s in ("D0","D1","D2")],[15,60,56])
        for t in p["tasks"]:
            self.assertEqual((len(t["train"]),len(t["validation"])),(18,6))
            self.assertFalse((set(t["train"])|set(t["validation"]))&WITHHELD)
            self.assertNotIn("test",t)
        self.assertEqual(p["with_E0_neural_step_cap"],430172)
    def test_cli_planning_no_heavy_import(self):
        from fatigue_training.__main__ import main
        code="import contextlib,io,sys,numpy as np; from fatigue_training.__main__ import main; np.load=lambda *a,**k: (_ for _ in ()).throw(AssertionError('array load')); out=io.StringIO(); ctx=contextlib.redirect_stdout(out); ctx.__enter__(); result=main(['dev-plan','--phase','D1']); ctx.__exit__(None,None,None); assert result==0; assert not {'torch','scipy','sklearn','mamba_ssm'} & sys.modules.keys()"
        done=subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).parent.parent,capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(["--config","missing.json","dev-plan"])
    def test_dev_train_missing_execute_rejected_before_dataset(self):
        from fatigue_training.__main__ import main
        with patch("fatigue_training.data.WindowDataset",side_effect=AssertionError("dataset instantiated")):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(["dev-train","--manifest","missing","--output","not-created","--phase","D1","--source-root","missing","--preflight","missing","--max-new-fits","1","--mode","interface"])
    def test_interaction_unit_is_subject(self):
        r=interaction_bootstrap({"5":.3,"14":.1},{"5":.1,"14":.2},replicates=1000)
        self.assertAlmostEqual(r["mean_difference"],.05)
        self.assertEqual(r["bootstrap_seed"],20261014)
        self.assertEqual(len(r["subject_seed_mean_interaction"]),2)
    def test_bins_disjoint_weighted_global_mean(self):
        self.assertEqual(bin_boundaries(50),[0,12,25,37,50]);self.assertEqual(bin_boundaries(100),[0,25,50,75,100])
        z=np.arange(50,dtype=float).reshape(1,50,1)
        means=disjoint_bin_means(z).reshape(4)
        self.assertAlmostEqual(np.dot(means,[12,13,12,13])/50,z.mean())
        self.assertNotAlmostEqual(means.mean(),z.mean())
        with self.assertRaises(ValueError):bin_boundaries(3)
    def test_sampler_shared_and_guard_blocks_before_io(self):
        d=FakeData();train,val=subjects(2)
        a=sample_stream(d,train,17,2,steps=3)
        np.random.seed(777);np.random.random(100)
        self.assertTrue(np.array_equal(a,sample_stream(d,train,17,2,steps=3)))
        self.assertFalse(np.array_equal(a,sample_stream(d,train,29,2,steps=3)))
        self.assertFalse(d.accesses)
        guard=SubjectGuard(d,train)
        forbidden=next(i for i,w in enumerate(d.windows) if w.subject_id in WITHHELD)
        with self.assertRaises(ValueError):guard.batch([forbidden],{})
        with self.assertRaises(ValueError):guard.slice(d.windows[forbidden],"eeg")
        with self.assertRaises(ValueError):guard.indices(val)
        self.assertFalse(d.accesses)
        self.assertTrue(all(d.windows[int(i)].subject_id in train for i in a.ravel()))
    def test_selection_gate_and_missing_fatigue(self):
        r=completed_tuning();selected=choose_lrs(r)
        self.assertTrue(all(lr==.0003 for lr in selected.values()))
        self.assertTrue(diagnostic_gate(r,selected)["passed"])
        for f in (2,3,4):r[Task("D1","F_EF",f,17,.0003).key]["fatigue_recall"]=.1
        self.assertFalse(diagnostic_gate(r,selected)["passed"])
        r=completed_tuning()
        r[Task("D1","F_EF",2,17,.0003).key]["fatigue_recall"]=None
        self.assertTrue(diagnostic_gate(r,selected)["missing_fatigue_support"])
        r=completed_tuning()
        for m in ("C_EF","M0_EF","T_EF"):
            for f in (2,3,4,5):r[Task("D1",m,f,17,.0003).key]["primary"]=.52
        self.assertTrue(diagnostic_gate(r,selected)["low_backbones"])
    def test_explicit_argmax_prediction_survives_probability_rounding(self):
        d=FakeData();i=d.indices(["5"])[0]
        r=prediction_rows(d,[i],[.5],Task("D1","F_EF",2,17,.0003),{"binding":{"data_fingerprint":"f"}},[1])
        self.assertEqual(r[0]["prediction"],1)
        self.assertEqual(r[0]["source_index"],d.windows[i].source_index)
        self.assertNotEqual(r[0]["source_index"],i)
    def test_stable_D1_review_hash_ignores_scheduler_or_D2_changes(self):
        from fatigue_training.development_training import write_review_report
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            r=completed_tuning();selection=choose_lrs(r);gate=diagnostic_gate(r,selection)
            provenance={k:"oldhash" for k in r}
            path=write_review_report(tmp,r,selection,gate,provenance);saved_hash=file_hash(path)
            write_json(Path(tmp)/"development_report.json",dict(state="completed",requested_phase="D1",executed_new_fits=60))
            self.assertEqual(file_hash(write_review_report(tmp,r,selection,gate,provenance)),saved_hash)
            write_json(Path(tmp)/"development_report.json",dict(state="budget_reached",requested_phase="D2",executed_new_fits=1))
            self.assertEqual(file_hash(write_review_report(tmp,r,selection,gate,provenance)),saved_hash)


class AccountingChecks(unittest.TestCase):
    def test_failure_reservation_persists_and_retry_never_resets(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            path=Path(tmp)/"ledger.json";ledger=Ledger(path,{"protocol_id":"test"})
            ledger.begin("D0","x","old");ledger.reserve("x",steps=2);ledger.finish("x","failed","fault after reservation")
            other=Ledger(path,{"protocol_id":"test"})
            self.assertEqual(other.state["phases"]["D0"]["steps"],2)
            with self.assertRaises(ValueError):other.begin("D0","x","new")
            receipt=dict(reason="repair",approved_at="2026",old_code_hash="old",new_code_hash="new")
            other.begin("D0","x","new",receipt);other.reserve("x",steps=2);other.finish("x","failed")
            self.assertEqual(other.state["phases"]["D0"]["steps"],4)
            with self.assertRaises(ValueError):other.begin("D0","x","third",receipt)
    def test_phase_hard_cap_and_E0_retries_inside_run_cap(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            ledger=Ledger(Path(tmp)/"ledger.json",{})
            ledger.begin("D1","x","h")
            ledger.state["phases"]["D1"]["steps"]=CAPS["D1"]["steps"]-1;ledger.save()
            ledger.reserve("x",steps=1)
            with self.assertRaises(ValueError):ledger.reserve("x",steps=1)
            for i in range(12):ledger.begin("E0",str(i),"h");ledger.finish(str(i),"completed")
            with self.assertRaises(ValueError):ledger.begin("E0","extra","h")
    def test_failed_fit_immutable_binding_and_changed_code_repair(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            path=Path(tmp);b=dict(protocol_id="p",task={"x":1},data_fingerprint="d",code_hash="old")
            write_json(path/"status.json",dict(state="failed",binding=b))
            self.assertIsNone(artifact_reuse(path,dict(b,code_hash="new")))
            with self.assertRaises(ValueError):artifact_reuse(path,dict(b,data_fingerprint="different"))
            write_json(path/"result.json",{"state":"completed"})
            write_json(path/"status.json",dict(state="completed",binding=b,artifacts={"result.json":file_hash(path/"result.json")}))
            self.assertEqual(artifact_reuse(path,b)["state"],"completed")
            with self.assertRaises(ValueError):artifact_reuse(path,dict(b,code_hash="new"))
            write_json(path/"result.json",{"state":"changed"})
            with self.assertRaises(ValueError):artifact_reuse(path,b)
    def test_receipt_empty_artifacts_rejected(self):
        from fatigue_training.development_preflight import verify_receipt
        from fatigue_training.development import source_hash
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            path=Path(tmp)/"receipt.json";env={"test":True}
            results={m:dict(state="completed",parameters=COUNTS[m],steps=4,backend_verified=True,resource_measurement={"batch":64}) for m in MODELS}
            write_json(path,dict(state="passed",protocol_id=configuration()["protocol_id"],config_hash=digest(configuration()),code_hash=source_hash(),environment=env,environment_hash=digest(env),runtime_verified=True,results=results,artifacts={}))
            with self.assertRaises(ValueError):verify_receipt(path,env)


class NormalizationChecks(unittest.TestCase):
    def test_train_only_union_subject_equal_and_constant_scale_one(self):
        d=FakeData();train,val=subjects(2)
        s=fit_normalization(d,train,val)
        self.assertFalse(set(d.accesses)&(set(val)|WITHHELD))
        for name in ("eeg","fnirs"):
            self.assertEqual(s["modalities"][name]["std"][-1],1)
            self.assertEqual(s["modalities"][name]["zero_variance_rows"],[(4 if name=="eeg" else 24)-1])
            counts=s["modalities"][name]["unique_samples_per_subject"]
            for sid in train:
                n=2 if int(sid)%2 else 3
                expected=(2500+(n-1)*1250) if name=="eeg" else (250+(n-1)*125)
                self.assertEqual(counts[sid],expected)
        with self.assertRaises(ValueError):fit_normalization(d,train+[val[0]],val)
    def test_feature_constant_anchor_unequal_windows_and_weights(self):
        ids=["5"]*2+["14"]*5
        x=np.zeros((7,104));x[:,0]=[0,2,10,10,10,10,10];x[:,1]=7.1
        s=fit_feature_scaler(x,ids,["5","14"])
        self.assertEqual(s["scale"][1],1);self.assertEqual(s["mean"][1],7.1)
        self.assertAlmostEqual(s["mean"][0],5.5)
        w=subject_weights(ids,mean_one=True)
        self.assertAlmostEqual(w.mean(),1)
        self.assertAlmostEqual(sum(w[:2]),sum(w[2:]))
        self.assertTrue(np.isfinite(apply_scaler(x,s)).all())
        with self.assertRaises(ValueError):fit_feature_scaler(x,ids,["5","14","21"])


@unittest.skipUnless(importlib.util.find_spec("scipy"),"scipy not installed; no synthetic spectral validation claimed")
class FeatureChecks(unittest.TestCase):
    def test_welch_band_power_ratio_and_fnirs_slope_units(self):
        t=np.arange(2500)/250
        eeg=np.tile(np.sin(2*np.pi*10*t)[None,None,:],(2,4,1))
        tf=np.arange(250)/25
        fnirs=np.tile((3+2*tf)[None,None,:],(2,24,1))
        values=signal_features(eeg,fnirs)
        self.assertEqual(values.shape,(2,104))
        per_channel=values[:,:32].reshape(2,4,8)
        self.assertTrue(np.all(per_channel[:,:,6]>.99))  # 8..13Hz relative band
        self.assertTrue(np.allclose(per_channel[:,:,4:].sum(-1),1))
        f=values[:,32:].reshape(2,24,3)
        self.assertTrue(np.allclose(f[:,:,0],fnirs.mean(-1)))
        self.assertTrue(np.allclose(f[:,:,1],fnirs.std(-1,ddof=0)))
        self.assertTrue(np.allclose(f[:,:,2],2))


@unittest.skipUnless(importlib.util.find_spec("torch"),"torch absent; runtime model tests skipped")
class TorchModelChecks(unittest.TestCase):
    def test_C_T_shared_front_head_and_optimizer(self):
        import torch
        from fatigue_training.development_models import DevelopmentModel,optimizer_groups
        c=DevelopmentModel("C_EF");t=DevelopmentModel("T_EF")
        for name,p in c.state_dict().items():
            if ".front." in name or name.startswith("head."):
                self.assertTrue(torch.equal(p,t.state_dict()[name]))
        for model in (c,t):
            groups,names=optimizer_groups(model)
            self.assertEqual(sum(len(g["params"]) for g in groups),sum(p.requires_grad for p in model.parameters()))
            self.assertFalse(set(names["decay"])&set(names["no_decay"]))
    @unittest.skipUnless(importlib.util.find_spec("mamba_ssm") and importlib.util.find_spec("selective_scan_cuda"),"official CUDA Mamba absent; no fallback")
    def test_official_cuda_variants(self):
        import torch
        if not torch.cuda.is_available():self.skipTest("actual CUDA unavailable")
        from fatigue_training.development_models import DevelopmentModel,optimizer_groups
        e=torch.randn(2,4,2500,device="cuda");f=torch.randn(2,24,250,device="cuda")
        base=DevelopmentModel("M0_EF").cuda().eval()
        with torch.no_grad():target=base(e,f)
        for name in ("Q_EF","A_EF","F_EF","K_EF"):
            model=DevelopmentModel(name).cuda().eval()
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),COUNTS[name])
            with torch.no_grad():self.assertTrue(torch.allclose(target,model(e,f),rtol=1e-4,atol=1e-5))
            _,names=optimizer_groups(model)
            self.assertTrue(all(n in names["no_decay"] for n,p in model.named_parameters() if n.endswith(("A_log",".D"))))


if __name__=="__main__":unittest.main()
