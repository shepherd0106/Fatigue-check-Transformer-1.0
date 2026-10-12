import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from fatigue_training.common import archive_incomplete_attempt, digest, read_json, write_json
from fatigue_training.data import CSV_FIELDS, MODALITIES, WindowDataset, adapt_legacy, merge_intervals
from fatigue_training.metrics import binary_metrics, evaluate, optional_evaluate, paired_bootstrap, stage_prior
from fatigue_training.normalization import fit_normalization
from fatigue_training.protocol import SubjectSampler, choose_grid, load_config, plan, refit_epochs, splits
from fatigue_training.synthetic import generate

TEST_TMP = Path(__file__).resolve().parent / ".tmp"
TEST_TMP.mkdir(exist_ok=True)


def mini_fixture(root):
    """Two unequal-length subjects; overlapping windows and one constant row."""
    root = Path(root)
    runs, windows = [], []
    for sid, count in [("2", 2), ("9", 3)]:
        directory = root/sid
        directory.mkdir()
        run = dict(run_id=sid, subject_id=sid, block_info="block1")
        for modality, (rows, length) in MODALITIES.items():
            n = length+(count-1)*(length//2)
            x = np.stack([np.linspace(int(sid), int(sid)+3, n, dtype=np.float32)+r for r in range(rows)])
            x[-1] = 7
            np.save(directory/f"{modality}.npy", x)
            np.save(directory/f"{modality}_coverage.npy", np.ones(n, dtype=bool))
            run[modality] = dict(array=f"{sid}/{modality}.npy", coverage=f"{sid}/{modality}_coverage.npy")
        runs.append(run)
        for j in range(count):
            windows.append(dict(source_index=len(windows), subject_id=sid, run_id=sid, block_info="block1", window_in_run=j,
                eeg_start=1250*j, eeg_end=1250*j+2500, fnirs_start=125*j, fnirs_end=125*j+250,
                label_2class=j%2, paired_valid=1, outer_test_fold=1, invalid_reasons=""))
    write_csv(root/"windows.csv", windows)
    write_json(root/"manifest.json", dict(schema_version=1, kind="synthetic", data_version="test", data_root=".",
        window_index="windows.csv", source=dict(generator="tests"), runs=runs))
    return root/"manifest.json"


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ProtocolTests(unittest.TestCase):
    def test_splits_and_budget(self):
        cfg = load_config()
        p = plan(cfg)
        self.assertEqual(p["fit_count"], 945)
        self.assertEqual(p["maximum_updates"], 945*120*128)
        self.assertEqual(plan(cfg, ["M2"], [1])["fit_count"], 27)
        seen = []
        for outer in range(1,6):
            train, _, test = splits(cfg, outer)
            seen.extend(test)
            self.assertEqual((len(train), len(test)), (24,6))
            for inner in range(1,6):
                if inner == outer:
                    continue
                train, val, test = splits(cfg, outer, inner)
                self.assertEqual((len(train),len(val),len(test)), (18,6,6))
                self.assertFalse(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
        self.assertEqual(len(set(seen)), 30)

    def test_invalid_config_and_scope(self):
        cfg = load_config()
        with self.assertRaises(ValueError):
            plan(cfg, ["M2", "M2"])
        cfg["transformer"]["norm_first"] = False
        from fatigue_training.protocol import validate_config
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_rounding_and_tuning_tolerance(self):
        self.assertEqual(refit_epochs([1,2,3,4]),3)
        self.assertEqual(refit_epochs([1,1,1,2]),1)
        with self.assertRaises(ValueError):
            refit_epochs([1,2,3])
        cfg = load_config()
        scores = {tuple(k): [.5]*4 for k in cfg["training"]["grid_tie_order"]}
        scores[(.001,.001)] = [.5+5e-13]*4
        self.assertEqual(choose_grid(scores,cfg),(.0003,.01))
        scores[(.001,.001)] = [.5+2e-12]*4
        self.assertEqual(choose_grid(scores,cfg),(.001,.001))

    def test_data_contract_and_interface_budget(self):
        from fatigue_training.protocol import validate_config, validate_formal, protocol_deviations
        cfg = load_config()
        for field, value in [("window_seconds",30),("stride_seconds",15),("input_dtype","float64")]:
            changed=copy.deepcopy(cfg)
            changed["data"][field]=value
            with self.assertRaisesRegex(ValueError, "data."+field):
                validate_formal(changed)
        cfg["training"].update(batch_size=2,updates_per_epoch=1,max_epochs=2)
        validate_config(cfg)
        self.assertEqual(plan(cfg,["M2"],[1])["maximum_updates"],54)
        self.assertEqual(len(protocol_deviations(cfg)),3)
        with self.assertRaisesRegex(ValueError,"Formal protocol controls changed"):
            validate_formal(cfg)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.root = Path(self.tmp.name)
        self.manifest = mini_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_centered_equal_subject_coverage_union(self):
        d = WindowDataset(self.manifest)
        stats = fit_normalization(d,["2","9"],cache=self.root/"cache")
        self.assertEqual(stats, fit_normalization(d,["9","2"],cache=self.root/"cache"))
        for modality in MODALITIES:
            arrays = [np.load(self.root/sid/f"{modality}.npy").astype(np.float64) for sid in ["2","9"]]
            means = np.stack([x.mean(axis=1) for x in arrays])
            mean = means.mean(axis=0)
            var = np.mean(np.stack([x.var(axis=1) for x in arrays])+(means-mean)**2,axis=0)
            np.testing.assert_allclose(stats["modalities"][modality]["mean"],mean,rtol=1e-12)
            np.testing.assert_allclose(stats["modalities"][modality]["variance"],var,rtol=1e-12,atol=1e-12)
            self.assertNotAlmostEqual(mean[0], np.concatenate(arrays,axis=1).mean(axis=1)[0])
            self.assertEqual(stats["modalities"][modality]["floored_rows"], [MODALITIES[modality][0]-1])
        self.assertEqual(stats["modalities"]["eeg"]["unique_samples_per_subject"], {"2":3750,"9":5000})
        self.assertEqual(merge_intervals([(0,10),(5,15),(15,20),(22,25)]),[[0,20],[22,25]])

    def test_normalization_leakage_and_test_data_not_fitted(self):
        d=WindowDataset(self.manifest)
        with self.assertRaises(ValueError):
            fit_normalization(d,["2"],forbidden_subjects=["2"])
        with self.assertRaises(ValueError):
            fit_normalization(d,["2","2"])
        stats=fit_normalization(d,["2"],forbidden_subjects=["9"])
        self.assertEqual(stats["binding"]["train_subjects"],["2"])
        self.assertEqual(stats["modalities"]["eeg"]["unique_samples_per_subject"],{"2":3750})
        batch=d.batch(d.indices(["9"]),stats)
        self.assertEqual(batch[0].shape,(3,4,2500))
        self.assertEqual(batch[2].dtype,np.dtype("int64"))

    def test_original_invalidity_and_nonoverlap_parity(self):
        with (self.root/"windows.csv").open() as f:
            rows=list(csv.DictReader(f))
        rows[0].update(paired_valid="0",invalid_reasons="original_missing")
        write_csv(self.root/"windows.csv",rows)
        d=WindowDataset(self.manifest)
        self.assertEqual([d.windows[i].window_in_run for i in d.indices(["2"])],[1])
        self.assertEqual(d.indices(["2"],nonoverlap=True),[])
        with self.assertRaises(ValueError):
            d.batch([0],fit_normalization(d,["2"]))
        self.assertEqual(optional_evaluate([]),{"n":0,"evaluation":None})

    def test_sampler_independent_rng_and_subject_weights(self):
        d=WindowDataset(self.manifest)
        first=SubjectSampler(d,["2","9"],17,1,2).draw(12000)
        np.random.seed(999)
        np.random.normal(size=500)
        second=SubjectSampler(d,["9","2"],17,1,2).draw(12000)
        np.testing.assert_array_equal(first,second)
        ratio=np.mean([d.windows[i].subject_id=="2" for i in first])
        self.assertAlmostEqual(ratio,.5,delta=.02)
        self.assertFalse(np.array_equal(first[:100],SubjectSampler(d,["2","9"],17,1,3).draw(100)))

    def test_shared_signal_alias_rejected(self):
        m=read_json(self.manifest)
        m["runs"][1]["eeg"]["array"]="2/../2/eeg.npy"
        write_json(self.manifest,m)
        with self.assertRaisesRegex(ValueError,"same continuous signal"):
            WindowDataset(self.manifest)

    def test_csv_change_after_parse_rejected(self):
        d=WindowDataset(self.manifest)
        with (self.root/"windows.csv").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError,"changed after loading"):
            d.fingerprint()

    def test_array_change_after_fingerprint_rejected(self):
        d=WindowDataset(self.manifest)
        d.fingerprint()
        path=self.root/"2/eeg.npy"
        with path.open("ab") as f:
            f.write(b"change")
        with self.assertRaisesRegex(ValueError,"changed after loading"):
            d.fingerprint()

    def test_wrong_dtype_hole_and_label_rejected(self):
        path=self.root/"2/eeg.npy"
        np.save(path,np.load(path).astype(np.float64))
        d=WindowDataset(self.manifest)
        with self.assertRaises(ValueError):
            d.slice(d.windows[0],"eeg")
        np.save(path,np.load(path).astype(np.float32))
        coverage=np.load(self.root/"2/eeg_coverage.npy")
        coverage[0]=False
        np.save(self.root/"2/eeg_coverage.npy",coverage)
        d=WindowDataset(self.manifest)
        with self.assertRaises(ValueError):
            fit_normalization(d,["2"])
        with (self.root/"windows.csv").open() as f:
            rows=list(csv.DictReader(f))
        rows[0]["label_2class"]="2"
        write_csv(self.root/"windows.csv",rows)
        with self.assertRaises(ValueError):
            WindowDataset(self.manifest)

    def test_stage_prior_tie_uses_only_train(self):
        d=WindowDataset(self.manifest)
        counts,rows=stage_prior(d,["2"],["9"],1)
        self.assertEqual(counts["block1"],[1,1])
        self.assertEqual(len(counts),16)
        self.assertTrue(all(r["probability_1"]==.5 and r["prediction"]==0 for r in rows))
        with self.assertRaises(ValueError):
            stage_prior(d,["2"],["2"],1)

    def test_formal_acceptance_bound_to_data(self):
        m=read_json(self.manifest)
        m["kind"]="real"
        write_json(self.manifest,m)
        d=WindowDataset(self.manifest)
        receipt=self.root/"receipt.json"
        a=dict(data_fingerprint=d.fingerprint(),data_version="test",approved_by="human-reviewer",approved_at="2026-10-11",
               time_axis="passed",fnirs_conversion="passed",labels="passed")
        write_json(receipt,a)
        self.assertEqual(d.acceptance(receipt)["receipt"],a)
        a["data_fingerprint"]="other"
        write_json(receipt,a)
        with self.assertRaises(ValueError):
            d.acceptance(receipt)

    def test_adapter_preserves_csv_without_arrays(self):
        legacy=self.root/"old.json"
        write_json(legacy,dict(runs=[dict(run_id=4,subject_id="2",block_info="block1",relative_directory="subject_2/block1_run_0004")]))
        result=adapt_legacy(legacy,self.root/"windows.csv",self.root/"adapted",self.root/"absent_npy_root")
        self.assertEqual((result.parent/"windows.csv").read_bytes(),(self.root/"windows.csv").read_bytes())
        self.assertEqual(read_json(result)["kind"],"real")


class MetricsTests(unittest.TestCase):
    def test_single_class_fixed_f1_and_auc(self):
        m=binary_metrics([0,0],[0,0],[.1,.5])
        self.assertEqual(m["support_recall"],1)
        self.assertEqual(m["macro_f1"],.5)
        self.assertIsNone(m["auroc"])
        self.assertEqual(m["recall"],[1,None])
        self.assertEqual(binary_metrics([0,1],[0,0],[.5,.5])["auroc"],.5)
        self.assertEqual(binary_metrics([0,1,0,1],[0,1,0,1],[.1,.7,.2,.9])["auroc"],1)

    def rows(self):
        left,right=[],[]
        for seed in [17,29,43]:
            for i,subject in enumerate(["a","b"]):
                row=dict(seed=seed,subject_id=subject,source_index=i,outer=1,run_id=subject,window_in_run=0,
                         block_info="block1",label=i,prediction=i,probability_1=.1 if i==0 else .9,data_fingerprint="f")
                left.append(row)
                bad=dict(row)
                if subject=="a":
                    bad.update(prediction=1,probability_1=.9)
                right.append(bad)
        return left,right

    def test_subject_weight_and_paired_seed_bootstrap(self):
        a,b=self.rows()
        result=paired_bootstrap(a,b,2000)
        self.assertEqual(result["mean_difference"],.5)
        self.assertEqual(result,paired_bootstrap(a,b,2000))
        self.assertEqual(result["subject_seed_mean_differences"],{"a":1.0,"b":0.0})
        part=[a[0]]*100+[b[1]]
        part[-1]=dict(part[-1],prediction=0,probability_1=.1)
        self.assertEqual(evaluate(part)["primary"],.5)
        self.assertGreater(evaluate(part)["pooled"]["accuracy"],.9)

    def test_pair_mismatch_errors(self):
        a,b=self.rows()
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b[:-1])
        b[0]["label"]=1
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b)
        a,b=self.rows()
        a[0]["source_index"]=33
        b[0]["source_index"]=33
        with self.assertRaisesRegex(ValueError,"across reporting seeds"):
            paired_bootstrap(a,b)
        with self.assertRaises(ValueError):
            binary_metrics([],[],[])


class EvidenceTests(unittest.TestCase):
    def test_report_scope_flags_and_archived_previous_report(self):
        from fatigue_training.reporting import scope_status, publish_report
        cfg=load_config()
        full=list(cfg["variants"])
        a=scope_status(cfg,"synthetic","interface",["M2"],[1,2,3,4,5],"completed","f")
        self.assertTrue(a["requested_scope_completed"])
        self.assertTrue(a["five_outer_folds_completed"])
        self.assertFalse(a["full_formal_protocol_completed"])
        self.assertFalse(scope_status(cfg,"real","formal",["M2"],[1,2,3,4,5],"completed","f")["full_formal_protocol_completed"])
        self.assertTrue(scope_status(cfg,"real","formal",full,[1,2,3,4,5],"completed","f")["full_formal_protocol_completed"])
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as root:
            root=Path(root)
            publish_report(root,a)
            b=scope_status(cfg,"synthetic","interface",full,[1,2,3,4,5],"budget_reached","f")
            publish_report(root,b)
            self.assertFalse(read_json(root/"report.json")["requested_scope_completed"])
            archived=list((root/"reports").glob("*.json"))
            self.assertEqual(len(archived),1)
            self.assertEqual(read_json(archived[0]),a)

    def test_failed_attempt_preserved(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as root:
            root=Path(root)
            write_json(root/"status.json",dict(state="failed",error="deterministic kernel unavailable",seconds=12))
            (root/"epochs.jsonl").write_text('{"epoch":1}\n')
            dest=archive_incomplete_attempt(root)
            self.assertEqual(read_json(dest/"status.json")["seconds"],12)
            self.assertEqual((dest/"epochs.jsonl").read_text(),'{"epoch":1}\n')
            self.assertEqual(archive_incomplete_attempt(root).name,"2")
            write_json(root/"status.json",dict(state="completed"))
            with self.assertRaises(ValueError):
                archive_incomplete_attempt(root)

    def test_completed_fit_reuse_checks_artifact_hash(self):
        from fatigue_training.training import reusable
        from fatigue_training.common import file_hash
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as root:
            root=Path(root)
            binding=dict(data="test")
            write_json(root/"result.json",dict(best_epoch=2))
            write_json(root/"status.json",dict(state="completed",fingerprint=digest(binding),artifacts={"result.json":file_hash(root/"result.json")}))
            self.assertEqual(reusable(root,binding),dict(best_epoch=2))
            with self.assertRaises(ValueError):
                reusable(root,dict(data="changed"))
            write_json(root/"result.json",dict(best_epoch=3))
            with self.assertRaises(ValueError):
                reusable(root,binding)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable: server forward/backward and fit checks pending")
class TorchChecks(unittest.TestCase):
    def test_all_models(self):
        from fatigue_training.smoke import model_checks
        self.assertEqual(model_checks()["status"],"passed")

    def test_synthetic_fit_checkpoint_reload_reuse(self):
        from fatigue_training.smoke import synthetic_fit
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as root:
            self.assertEqual(synthetic_fit(root)["status"],"passed")


if __name__=="__main__":
    unittest.main()
