import copy
import csv
import importlib.metadata
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from fatigue_training.common import file_hash, read_json, write_json
from fatigue_training.data import CSV_FIELDS, WindowDataset
from fatigue_training.time_axis import (CORRECTED_LOADER, FIXED_POLICY, RAW_LOADER,
    SourceRejected, checked_hash, exact_sample_identity, execute,
    interpolate_supported, load_plan, load_source, main, prepare_run, stream_map)


def stream(sid, times, values):
    info=dict(stream_id=sid)
    layout={1:("EEG",4,250),2:("FNIRS",16,25),3:("Markers",1,0)}
    if sid in layout:
        kind,channels,rate=layout[sid]
        info.update(type=[kind],channel_count=[str(channels)],nominal_srate=[str(rate)])
    return dict(info=info,time_stamps=np.asarray(times),time_series=np.asarray(values))


def fixture(root):
    root = Path(root)
    source = root/"source.xdf"
    source.write_bytes(b"synthetic source used by mocked loader")
    streams = {}
    mapping = dict(run_id="r1",subject_id="10",block_info="block1",source_id="source",
        markers_stream_id=3,start_marker_index=0,end_marker_index=1,
        expected_marker_start_value=1,expected_marker_end_value=1,marker_raw_time_tolerance_seconds=.01,
        historical_blk=dict(Task="NBack",BlockID="block1",start_time=.2,end_time=20.2,source_xdf="source",KSS=2))
    for modality, sid, channels, rate in (("eeg",1,4,250),("fnirs",2,24,25)):
        # Distinct device clocks/sampling drift, with the same linear physical signal.
        times = np.arange(0,21,1/rate)*(1.0005 if sid==1 else .9995)
        processed = np.tile(times,(channels,1)).astype(np.float32)
        raw_channels=4 if modality=="eeg" else 16
        values = np.tile(times,(raw_channels,1)).astype(np.float32)
        raw_path, proc_path = root/(modality+"_raw.mat"), root/(modality+"_processed.mat")
        savemat(raw_path,dict(raw=values,blk=mapping["historical_blk"]))
        savemat(proc_path,dict(processed=processed))
        mapping[modality] = dict(stream_id=sid,start_index=0,stop_index_exclusive=len(times),
            raw_mat=str(raw_path),raw_variable="raw",processed_mat=str(proc_path),processed_variable="processed")
        streams[sid] = stream(sid,times,values.T)
    streams[3] = stream(3,[.2,20.2],[[1],[1]])
    fields, rows = CSV_FIELDS, []
    for rid, count in (("r1",3),("r2",1)):
        for j in range(count):
            rows.append(dict(source_index=str(len(rows)),subject_id="10",run_id=rid,block_info="block1",
                window_in_run=str(j),eeg_start=str(j*1250),eeg_end=str(j*1250+2500),
                fnirs_start=str(j*125),fnirs_end=str(j*125+250),label_2class=str(j%2),
                paired_valid="0" if rid=="r1" and j==1 else "1",outer_test_fold="3",
                invalid_reasons="legacy_missing" if rid=="r1" and j==1 else ""))
    csv_path = root/"legacy.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = root/"legacy.json"
    write_json(manifest_path,dict(schema_version=1,kind="real",runs=[
        dict(run_id=rid,subject_id="10",block_info="block1") for rid in ("r1","r2")]))
    plan = dict(schema_version=1,legacy_manifest=str(manifest_path),legacy_window_csv=str(csv_path),
        scope_subject_ids=["10"],policy=dict(**FIXED_POLICY,max_interpolation_gap_seconds=dict(eeg=.04,fnirs=.2)),
        sources=dict(source=dict(path=str(source),sha256=file_hash(source),
            expected_stream_samples={str(sid):len(s["time_stamps"]) for sid,s in streams.items()})),
        runs=[mapping,dict(run_id="r2",subject_id="10",block_info="block1",rejection_reason="source_not_traced")])
    path = root/"plan.json"
    write_json(path,plan)
    return path, plan, rows, streams


class TimeAxisTests(unittest.TestCase):
    def test_marker_values_nearest_boundary_and_actual_blk_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            _,plan,_,streams=fixture(root)
            original=plan["runs"][0]
            for side in ("raw","corrected"):
                damaged=copy.deepcopy(streams)
                damaged[3]["time_series"][1,0]=200
                raw,corrected=(damaged,streams) if side=="raw" else (streams,damaged)
                with self.assertRaisesRegex(SourceRejected,"marker_sample_value_mismatch"):
                    prepare_run(original,raw,corrected,{}, {})
            shifted=copy.deepcopy(streams)
            shifted[3]=stream(3,[.2,5,20.2],[[1],[1],[1]])
            with self.assertRaisesRegex(SourceRejected,"marker_index_not_unique_nearest"):
                prepare_run(original,shifted,shifted,{}, {})
            for field,value in (("Task","SART"),("KSS",9),("source_xdf","wrong"),("BlockID","rest1"),("start_time",.3)):
                mapping=copy.deepcopy(original)
                mapping["historical_blk"][field]=value
                with self.assertRaisesRegex(SourceRejected,"historical_blk_source_metadata_mismatch"):
                    prepare_run(mapping,streams,streams,{}, {})
            # Stored BlockID can legitimately differ from the global stage.
            remapped=copy.deepcopy(original)
            remapped["historical_blk"]["BlockID"]="rest1"
            savemat(remapped["eeg"]["raw_mat"],dict(raw=streams[1]["time_series"].T,blk=remapped["historical_blk"]))
            diagnostics=prepare_run(remapped,streams,streams,{}, {})[3]
            self.assertTrue(diagnostics["historical_block_id_remapped"])
            self.assertEqual(diagnostics["marker_mapping_raw_residuals_seconds"],[0.,0.])
            remapped["historical_blk"]["start_time"]=.22
            savemat(remapped["eeg"]["raw_mat"],dict(raw=streams[1]["time_series"].T,blk=remapped["historical_blk"]))
            with self.assertRaisesRegex(SourceRejected,"residual_exceeds_tolerance"):
                prepare_run(remapped,streams,streams,{}, {})

    def test_pyxdf_warnings_and_loader_metadata_persist_in_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,_,_,streams=fixture(root)
            for s in streams.values():
                s["info"].update(effective_srate=25.,segments=[(0,len(s["time_stamps"])-1)],clock_segments=[])
                s["clock_times"]=np.array([1.,2.])
                s["clock_values"]=np.array([.001,.002])
            logger=logging.getLogger("pyxdf")
            original_handlers=list(logger.handlers)
            def noisy_loader(*args,**kwargs):
                logging.getLogger("pyxdf.pyxdf").warning("Stream 1: Segments and clock-segments differ")
                return list(streams.values()),dict(info=dict(version=["1.0"]))
            with patch("pyxdf.load_xdf",side_effect=noisy_loader):
                report=execute(load_plan(path),root/"out")
            self.assertEqual(logger.handlers,original_handlers)
            receipt=read_json(root/"out/preview_receipt.json")
            for checks in (report["source_checks"],receipt["checks"]["raw_corrected_identity_by_source"]):
                source=checks["source"]
                self.assertEqual(len(source["raw_warnings"]),1)
                self.assertEqual(len(source["corrected_warnings"]),1)
                self.assertEqual(source["warning_decision"],"retained_for_nonformal_preview_manual_review_required")
                self.assertEqual(source["corrected_file_header"]["info"]["version"],["1.0"])
                self.assertEqual(source["corrected_stream_summaries"]["1"]["clock_values"]["max"],.002)
                self.assertIn("segments",source["corrected_stream_summaries"]["1"]["info"])
            self.assertEqual(receipt["formal_acceptance"],"pending")

    def test_noninteger_stream_ids_rejected(self):
        for sid in (1.5,"1.5",[1.5],True):
            s=stream(7,[0,1],[[1],[1]])
            s["info"]["stream_id"]=sid
            with self.assertRaisesRegex(SourceRejected,"noninteger_stream_id"):
                stream_map([s])
        for sid in (7,"7",["7"],np.int64(7)):
            s=stream(7,[0,1],[[1],[1]])
            s["info"]["stream_id"]=sid
            self.assertEqual(set(stream_map([s])),{7})

    def test_fnirs_raw16_processed24_and_processed_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,plan,_,streams=fixture(root)
            self.assertEqual(streams[2]["time_series"].shape[1],16)
            item=plan["runs"][0]["fnirs"]
            reference=root/"fnirs_reference.mat"
            savemat(reference,dict(data=np.tile(streams[2]["time_stamps"],(24,1)).astype(np.float32)))
            item.update(processed_reference_mat=str(reference),processed_reference_variable="data")
            write_json(path,plan)
            with patch("fatigue_training.time_axis.load_source",return_value=(streams,streams,{})):
                report=execute(load_plan(path),root/"out")
            self.assertEqual(report["after_eligible_count"],2)
            self.assertEqual(report["run_diagnostics"][0]["fnirs"]["raw_channel_count"],16)
            self.assertEqual(report["run_diagnostics"][0]["fnirs"]["processed_channel_count"],24)
            dataset=WindowDataset(root/"out/manifest.json")
            self.assertEqual(dataset.arrays("r1","fnirs")[0].shape[0],24)
            del dataset  # Release Windows mmap handles before temp cleanup.

    def test_raw_nonfinite_rejected_despite_finite_processed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            _,plan,_,streams=fixture(root)
            for modality,sid in (("eeg",1),("fnirs",2)):
                for value in (np.nan,np.inf):
                    raw=copy.deepcopy(streams)
                    raw[sid]["time_series"][0,0]=value
                    # The raw MAT even agrees with the damaged source; finite
                    # processed data must not turn it into observed coverage.
                    savemat(plan["runs"][0][modality]["raw_mat"],dict(raw=raw[sid]["time_series"].T,blk=plan["runs"][0]["historical_blk"]))
                    with self.assertRaisesRegex(SourceRejected,"raw_xdf_or_mat_nonfinite"):
                        prepare_run(plan["runs"][0],raw,raw,{}, {})
                savemat(plan["runs"][0][modality]["raw_mat"],dict(raw=streams[sid]["time_series"].T,blk=plan["runs"][0]["historical_blk"]))

    def test_wrong_stream_header_or_shared_marker_id_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _,plan,_,streams=fixture(Path(temp).resolve())
            for sid in (1,2,3):
                for field,value in (("type",["wrong"]),("channel_count",["99"]),("nominal_srate",["999"])):
                    for side in ("raw","corrected"):
                        damaged=copy.deepcopy(streams)
                        damaged[sid]["info"][field]=value
                        raw,corrected=(damaged,streams) if side=="raw" else (streams,damaged)
                        with self.assertRaisesRegex(SourceRejected,"stream_header_layout_mismatch"):
                            prepare_run(plan["runs"][0],raw,corrected,{}, {})
            mapping=copy.deepcopy(plan["runs"][0])
            mapping["markers_stream_id"]=mapping["eeg"]["stream_id"]
            with self.assertRaisesRegex(SourceRejected,"stream_ids_not_distinct"):
                prepare_run(mapping,streams,streams,{}, {})

    def test_two_drifting_modalities_reconstruct_same_physical_time(self):
        grid = np.arange(.1,10,.04)
        for step, drift in ((.004,1.003),(.04,.997)):
            t = np.arange(0,11,step)*drift
            a, coverage, audit = interpolate_supported(t,np.stack([2*t+1]),grid,.1)
            self.assertTrue(coverage.all())
            np.testing.assert_allclose(a[0],2*grid+1,rtol=0,atol=2e-6)
            self.assertTrue(np.all(audit["left_timestamp"]<=grid))
            self.assertTrue(np.all(audit["right_timestamp"]>=grid))

    def test_gap_edges_and_direct_samples_no_hold_fill(self):
        t = np.array([0.,1.,4.,5.])
        grid = np.array([-1.,0.,.5,1.,2.,4.,4.5,5.,6.])
        a, covered, audit = interpolate_supported(t,np.array([[0,10,40,50]]),grid,1.1,source_start=8)
        np.testing.assert_array_equal(covered,[False,True,True,True,False,True,True,True,False])
        self.assertTrue(np.isnan(a[0,[0,4,8]]).all())
        np.testing.assert_allclose(a[0,covered],[0,5,10,40,45,50])
        np.testing.assert_array_equal(audit["direct_source"],[False,True,False,True,False,True,False,True,False])
        self.assertEqual(audit["left_source_index"][1],8)
        self.assertEqual(audit["right_source_index"][1],8)
        self.assertEqual(audit["right_weight"][1],0)

    def test_no_sorting_or_duplicate_repair(self):
        for t in ([0,1,1],[0,2,1],[0,float("nan"),1]):
            with self.assertRaises(SourceRejected):
                interpolate_supported(t,np.zeros((1,3)),np.array([.5]),1)

    def test_count_value_and_stream_identity_rejections(self):
        raw = {7:stream(7,[0,1],[[1],[2]])}
        exact_sample_identity(raw,copy.deepcopy(raw),{"7":2})
        for corrected in ({7:stream(7,[0],[[1]])},{7:stream(7,[0,1],[[1],[3]])},{}):
            with self.assertRaises(SourceRejected):
                exact_sample_identity(raw,corrected,{"7":2})
        self.assertEqual(set(stream_map([raw[7]])),{7})
        listed = copy.deepcopy(raw[7])
        listed["info"]["stream_id"]=[7]
        self.assertEqual(set(stream_map([listed])),{7})
        string_raw = {8:stream(8,[0,1],[["start"],["stop"]])}
        exact_sample_identity(string_raw,copy.deepcopy(string_raw),{"8":2})

    def test_marker_end_and_source_gap_exclude_only_affected_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,plan,old,streams=fixture(root)
            corrected=copy.deepcopy(streams)
            # A clock-corrected 0.1s EEG break exceeds the EEG gap policy;
            # values/sample order stay unchanged and no sorting is performed.
            times=corrected[1]["time_stamps"]
            times[times>=2]+=0.1
            corrected[3]["time_stamps"][1]=15.1
            with patch("fatigue_training.time_axis.load_source",return_value=(streams,corrected,{})):
                report=execute(load_plan(path),root/"out")
            reasons={r["source_index"]:r["new_reasons"] for r in report["rejections"]}
            self.assertIn("unsupported_interpolation:eeg",reasons[0])
            self.assertIn("window_exceeds_marker_end",reasons[2])
            self.assertEqual(report["total_rows"],len(old))

    def test_bad_hash_before_loader_and_fixed_double_load_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            path,plan,_,streams=fixture(Path(temp).resolve())
            source=plan["sources"]["source"]
            with patch("pyxdf.load_xdf",return_value=(list(streams.values()),{})) as loader:
                a,b,details=load_source(source)
                self.assertEqual(loader.call_count,2)
                self.assertEqual(loader.call_args_list[0].kwargs,RAW_LOADER)
                self.assertEqual(loader.call_args_list[1].kwargs,CORRECTED_LOADER)
                checked_hash(source["path"],source["sha256"])
                source["sha256"]="0"*64
                with self.assertRaises(SourceRejected):
                    load_source(source)
                self.assertEqual(loader.call_count,2)

    def test_preview_preserves_supervision_invalidity_and_new_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,plan,old,streams=fixture(root)
            with patch("fatigue_training.time_axis.load_source",return_value=(streams,streams,{})):
                report=execute(load_plan(path),root/"out")
            self.assertEqual((report["before_eligible_count"],report["after_eligible_count"]),(3,2))
            with (root/"out/windows.csv").open() as f:
                new=list(csv.DictReader(f))
            for a,b in zip(old,new):
                for key in ("source_index","subject_id","run_id","block_info","window_in_run","label_2class","outer_test_fold"):
                    self.assertEqual(a[key],b[key])
            self.assertEqual(new[1]["paired_valid"],"0")
            self.assertIn("legacy_missing",new[1]["invalid_reasons"])
            self.assertEqual(new[3]["paired_valid"],"0")
            ds=WindowDataset(root/"out/manifest.json")
            receipt=read_json(root/"out/preview_receipt.json")
            self.assertEqual(receipt["data_fingerprint"],ds.fingerprint())
            self.assertEqual(receipt["labels"],"not_revalidated")
            with self.assertRaises(ValueError):
                ds.acceptance(root/"out/preview_receipt.json")
            run=ds.runs["r2"]
            self.assertTrue(np.isnan(np.load(root/"out"/run["eeg"]["array"])).all())
            self.assertFalse(np.load(root/"out"/run["eeg"]["coverage"]).any())
            with self.assertRaises(FileExistsError):
                execute(load_plan(path),root/"out")
            manifest=(root/"out/manifest.json").read_text()
            self.assertNotIn(str(root),manifest)

    def test_mat_mismatch_rejects_run_without_deleting_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,plan,old,streams=fixture(root)
            values=streams[1]["time_series"].T.copy()
            values[0,0]+=1
            savemat(plan["runs"][0]["eeg"]["raw_mat"],dict(raw=values,blk=plan["runs"][0]["historical_blk"]))
            with patch("fatigue_training.time_axis.load_source",return_value=(streams,streams,{})):
                report=execute(load_plan(path),root/"out")
            self.assertEqual(report["after_eligible_count"],0)
            self.assertEqual(report["total_rows"],len(old))
            self.assertIn("raw_mat_xdf_value_mismatch:eeg",report["run_diagnostics"][0]["rejection_reason"])
            self.assertEqual(report["window_dataset_interface"]["state"],"not_loadable_no_common_valid_windows")

    def test_plan_only_no_signal_reads_imports_or_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,_,_,_=fixture(root)
            with patch("fatigue_training.time_axis.load_source",side_effect=AssertionError("signal read")), \
                 patch("fatigue_training.time_axis.mat_array",side_effect=AssertionError("MAT read")):
                main(["--source-plan",str(path),"--output",str(root/"out")])
            self.assertFalse((root/"out").exists())

    def test_plan_requires_complete_scope_and_fixed_window_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path,plan,_,_=fixture(root)
            for change in (lambda p:p["runs"].pop(),lambda p:p["policy"].update(stride_seconds=10)):
                bad=copy.deepcopy(plan)
                change(bad)
                write_json(path,bad)
                with self.assertRaises(ValueError):
                    load_plan(path)


if __name__=="__main__":
    unittest.main()
