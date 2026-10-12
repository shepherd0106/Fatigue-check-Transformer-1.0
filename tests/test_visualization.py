import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fatigue_training.common import digest,read_json,write_json
from fatigue_training.monitoring import EpochLogger,check_logging,export_logs
from fatigue_training.protocol import load_config
from fatigue_training.metrics import paired_bootstrap
from fatigue_training.visualization import collect_results,finite,read_curve,read_rows,roc_pr,render,safe_json,PREDICTION_FIELDS
from fatigue_training.visualization_demo import generate_demo

TMP=Path(__file__).resolve().parent/".tmp"
TMP.mkdir(exist_ok=True)


def csv_rows(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(source=0,subject="2",label=0,model="M2",seed=17,outer=1):
    return dict(model=model,seed=seed,outer=outer,subject_id=subject,source_index=source,run_id="run_"+subject,
                window_in_run=source,block_info="block1",label=label,prediction=label,
                probability_1=.8 if label else .2,data_fingerprint="data-v1")


def fixture(root,state="completed",models=None):
    cfg=load_config()
    binding=dict(config_hash=digest(cfg),data_fingerprint="data-v1",mode="interface")
    write_json(root/"run.json",dict(binding=binding))
    write_json(root/"report.json",dict(state=state,scope=dict(binding,models=models or ["M2"],outer_folds=[1],data_kind="synthetic"),
        requested_scope_completed=state=="completed",full_formal_protocol_completed=False))
    write_json(root/"progress.json",dict(state=state))
    return cfg,binding


def fit_fixture(root,cfg,binding,name,stage="refit",rows=None,test=None,model="M2",outer=1,state="completed"):
    directory=root/model/f"outer_{outer}"/name
    directory.mkdir(parents=True)
    task=dict(stage=stage,model=model,seed=17,outer=outer,lr=.0003,weight_decay=.01,
              test=test if test is not None else ["2","9"],validation=[] if stage=="refit" else ["5"],train=["1"])
    b=dict(binding,task=task)
    write_json(directory/"configuration.json",dict(config=cfg,task=task,binding=b))
    write_json(directory/"status.json",dict(state=state,binding=b,fingerprint=digest(b)))
    write_json(directory/"result.json",dict(best_epoch=1))
    record=dict(epoch=1,mean_loss=.6,seconds=2,best_epoch=1,primary_validation=None if stage=="refit" else .7)
    (directory/"epochs.jsonl").write_text(json.dumps(record)+"\n",encoding="utf-8")
    if rows is not None:
        csv_rows(directory/"predictions.csv",rows)
    return directory


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=TMP)
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_partial_scope(self):
        fixture(self.root,"budget_reached")
        d=collect_results(self.root)
        self.assertEqual(d["state"],"budget_reached")
        self.assertFalse(d["requested_scope_completed"])
        self.assertFalse(d["runs"])
        render(self.root,self.root/"empty.html")
        self.assertTrue((self.root/"empty.html").exists())

    def test_inner_predictions_do_not_enter_test_panels(self):
        cfg,b=fixture(self.root,"budget_reached")
        fit_fixture(self.root,cfg,b,"tune",stage="tune",rows=[row()])
        d=collect_results(self.root)
        self.assertEqual(len(d["fits"]),1)
        self.assertEqual(d["fits"][0]["evaluation_scope"],"inner_validation")
        self.assertFalse(d["summaries"]["M2"])

    def test_current_scope_filters_old_models_and_folds(self):
        cfg,b=fixture(self.root,"budget_reached")
        fit_fixture(self.root,cfg,b,"current",rows=[row(),row(1,"9",1)])
        fit_fixture(self.root,cfg,b,"oldmodel",model="F1",rows=[row(model="F1")])
        fit_fixture(self.root,cfg,b,"oldfold",outer=2,rows=[row(outer=2)])
        csv_rows(self.root/"M2_oof.csv",[row(),row(1,"9",1),row(2,"12",0,outer=2)])
        d=collect_results(self.root)
        self.assertEqual(len(d["fits"]),1)
        self.assertEqual(d["summaries"]["M2"]["17"]["windows"],2)
        self.assertEqual(d["summaries"]["M2"]["17"]["outer_folds"],[1])

    def test_failed_and_running_fits_not_test_evidence(self):
        cfg,b=fixture(self.root,"running")
        fit_fixture(self.root,cfg,b,"running",rows=[row()],state="running")
        d=collect_results(self.root)
        self.assertEqual(d["fits"][0]["state"],"running")
        self.assertFalse(d["runs"])

    def test_heldout_mismatch_does_not_resurrect_root_oof(self):
        cfg,b=fixture(self.root)
        fit_fixture(self.root,cfg,b,"bad",rows=[row()],test=["2","9"])
        csv_rows(self.root/"M2_oof.csv",[row(),row(1,"9",1)])
        d=collect_results(self.root)
        self.assertEqual(d["fits"][0]["state"],"invalid")
        self.assertFalse(d["summaries"]["M2"])

    def test_late_conflict_is_transactional(self):
        cfg,b=fixture(self.root)
        fit_fixture(self.root,cfg,b,"a_valid",rows=[row(),row(1,"9",1)])
        fit_fixture(self.root,cfg,b,"z_bad",rows=[row(2),row(0,"9",1)])
        d=collect_results(self.root)
        self.assertEqual(d["summaries"]["M2"]["17"]["windows"],2)
        self.assertEqual([f["state"] for f in d["fits"]],["completed","invalid"])
        self.assertTrue(d["warnings"])

    def test_invalid_first_fit_does_not_bind_legacy_scope(self):
        cfg,b=fixture(self.root)
        (self.root/"report.json").unlink()
        (self.root/"run.json").unlink()
        bad=fit_fixture(self.root,cfg,dict(b,data_fingerprint="stale-data"),"a_bad")
        (bad/"epochs.jsonl").write_text(json.dumps(dict(epoch=1,mean_loss=.5,seconds=1,
            primary_validation=.8,best_epoch=1)),encoding="utf-8")
        fit_fixture(self.root,cfg,b,"z_valid",rows=[row(),row(1,"9",1)])
        d=collect_results(self.root)
        self.assertEqual(d["scope"]["data_fingerprint"],"data-v1")
        self.assertEqual(d["summaries"]["M2"]["17"]["windows"],2)

    def test_saved_bootstrap_interval_and_parameters_match_predictions(self):
        cfg,b=fixture(self.root,models=["M2","M1"])
        cfg["evaluation"]["subject_bootstrap_replicates"]=8
        write_json(self.root/"run.json",dict(binding=b,protocol=cfg))
        left=[row(i,subject,label,seed=seed) for seed in (17,29,43)
              for i,subject,label in [(0,"2",0),(1,"9",1)]]
        right=[dict(r,model="M1",prediction=0,probability_1=.2) for r in left]
        csv_rows(self.root/"M2_oof.csv",left)
        csv_rows(self.root/"M1_oof.csv",right)
        report=read_json(self.root/"report.json")
        saved=paired_bootstrap(left,right,8,cfg["evaluation"]["bootstrap_seed"])
        report["contrasts"]={"M2-M1":saved}
        write_json(self.root/"report.json",report)
        self.assertEqual(len(collect_results(self.root)["contrasts"]),1)
        for change in [dict(percentile_95=[-.9,.9]),dict(replicates=9),dict(bootstrap_seed=1)]:
            report["contrasts"]={"M2-M1":dict(saved,**change)}
            write_json(self.root/"report.json",report)
            d=collect_results(self.root)
            self.assertFalse(d["contrasts"])
            self.assertTrue(any("配对比较" in w for w in d["warnings"]))

    def test_bootstrap_rejects_truth_or_provenance_changes_across_seeds(self):
        rows=[row(i,subject,label,seed=seed) for seed in (17,29,43)
              for i,subject,label in [(0,"2",0),(1,"9",1)]]
        self.assertEqual(paired_bootstrap(rows,rows,1)["mean_difference"],0)
        for field,value in [("label",1),("outer",2),("run_id","different"),
                            ("window_in_run",3),("block_info","rest1"),("data_fingerprint","other")]:
            changed=[dict(r) for r in rows]
            changed[-2][field]=value
            with self.assertRaisesRegex(ValueError,"across reporting seeds"):
                paired_bootstrap(changed,changed,1)

    def test_same_seed_metrics_roc_pr_ties_and_singleclass(self):
        cfg,b=fixture(self.root)
        rows=[row(),row(1,"9",1)]
        for r in rows:r["probability_1"]=.5;r["prediction"]=0
        fit_fixture(self.root,cfg,b,"valid",rows=rows)
        d=collect_results(self.root)
        s=d["summaries"]["M2"]["17"]
        self.assertEqual(s["metrics"]["primary"],.5)
        self.assertEqual(s["metrics"]["pooled"]["auroc"],.5)
        self.assertEqual(s["curves"]["roc"],[[0.,0.],[1.,1.]])
        self.assertEqual(s["curves"]["pr"],[[0.,1.],[1.,.5]])
        self.assertFalse(roc_pr([row()])["available"])
        self.assertFalse(d["comparison"])

    def test_bad_prediction_format_nonfinite_and_provenance(self):
        path=self.root/"rows.csv"
        for key,value in [("label",2),("probability_1",float("nan")),("probability_1",1.5),
                          ("model","unknown"),("seed",3),("outer",0)]:
            r=row();r[key]=value;csv_rows(path,[r])
            with self.assertRaises(ValueError):read_rows(path)
        csv_rows(path,[row()])
        with self.assertRaises(ValueError):read_rows(path,expected_fingerprint="another")
        path.write_text("label,prediction\n0,0\n")
        with self.assertRaises(ValueError):read_rows(path)

    def test_curve_partial_and_refit_leakage(self):
        path=self.root/"epochs.jsonl"
        r=dict(epoch=1,mean_loss=.4,primary_validation=None,seconds=1,best_epoch=1)
        path.write_text(json.dumps(r)+'\n{"epoch":2',encoding="utf-8")
        self.assertEqual(len(read_curve(path,"refit",allow_partial=True)),1)
        with self.assertRaises(json.JSONDecodeError):read_curve(path,"refit")
        r["primary_validation"]=.9;path.write_text(json.dumps(r),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Refit epoch logs"):read_curve(path,"refit")

    def test_safe_json_and_no_external_assets(self):
        dangerous='</script><script>alert("x")</script>\u2028\u2029<&>'
        safe=safe_json(dict(label=dangerous))
        self.assertNotIn("<",safe)
        self.assertNotIn("\u2028",safe)
        self.assertEqual(json.loads(safe)["label"],dangerous)
        fixture(self.root)
        render(self.root,self.root/"report.html")
        html=(self.root/"report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script src=",html)
        self.assertNotIn("fetch(",html)
        self.assertNotIn("innerHTML",html)
        self.assertNotIn("__REPORT_PAYLOAD__",html)

    def test_demo_full_chain_no_torch_signals(self):
        result=generate_demo(self.root,windows_per_run=2)
        self.assertEqual(result["fits"],106)
        self.assertEqual(result["prediction_runs"],1260)
        self.assertFalse(list(self.root.rglob("*.npy")))
        d=collect_results(self.root)
        self.assertTrue(d["demo"])
        self.assertEqual(len(d["comparison"]),7)
        self.assertEqual(len(d["contrasts"]),5)
        self.assertFalse(d["full_formal_protocol_completed"])


class FakeWriter:
    instances=[]
    def __init__(self,log_dir):
        self.directory=log_dir;self.scalars=[];self.flushed=0;self.closed=0
        self.instances.append(self)
    def add_scalar(self,*args):self.scalars.append(args)
    def flush(self):self.flushed+=1
    def close(self):self.closed+=1


class MonitoringTests(unittest.TestCase):
    def test_opt_in_only_dependency_check(self):
        with patch("fatigue_training.monitoring.writer_factory",side_effect=ImportError("missing")) as factory:
            check_logging(None);check_logging(dict(tensorboard=False))
            factory.assert_not_called()
            with self.assertRaises(ImportError):check_logging(dict(tensorboard=True))

    def test_writer_scalar_tags_refit_isolation_flush_close_attempts(self):
        with tempfile.TemporaryDirectory(dir=TMP) as root:
            logger=EpochLogger(root,FakeWriter)
            record=dict(epoch=1,mean_loss=.5,primary_validation=.8,seconds=2)
            logger.write(record,.0003,is_refit=True)
            self.assertNotIn("inner_validation/primary",[v[0] for v in logger.writer.scalars])
            logger.write(record,.0003,is_refit=False)
            self.assertIn("inner_validation/primary",[v[0] for v in logger.writer.scalars])
            logger.close();logger.close()
            self.assertEqual(logger.writer.closed,1)
            self.assertGreaterEqual(logger.writer.flushed,3)
            second=EpochLogger(root,FakeWriter)
            self.assertNotEqual(second.directory,logger.directory)
            second.close()

    def test_writer_failure_cleanup_and_export_old_logs(self):
        class Broken(FakeWriter):
            def add_scalar(self,*args):raise OSError("disk full")
        with tempfile.TemporaryDirectory(dir=TMP) as root:
            logger=EpochLogger(root,Broken)
            try:
                with self.assertRaisesRegex(RuntimeError,"TensorBoard log write failed"):
                    logger.write(dict(epoch=1,mean_loss=.5,seconds=2))
            finally:logger.close()
            self.assertEqual(logger.writer.closed,1)
        with tempfile.TemporaryDirectory(dir=TMP) as root:
            root=Path(root);cfg,b=fixture(root)
            fit_fixture(root,cfg,b,"refit",rows=[row(),row(1,"9",1)])
            result=export_logs(root,root/"export",FakeWriter)
            self.assertEqual(result["count"],1)
            self.assertFalse(result["signals_or_checkpoints_loaded"])
            writer=FakeWriter.instances[-1]
            self.assertEqual(writer.closed,1)
            self.assertNotIn("inner_validation/primary",[v[0] for v in writer.scalars])

    def test_default_cli_without_torch_tensorboard(self):
        from fatigue_training.__main__ import main
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main([]),0)
            self.assertEqual(main(["plan"]),0)
        self.assertIn("945",captured.getvalue())


if __name__=="__main__":unittest.main()
