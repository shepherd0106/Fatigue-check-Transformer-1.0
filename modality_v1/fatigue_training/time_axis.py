"""Build an opt-in clock-corrected preview from audited legacy source mappings.

This module does not revalidate fatigue labels, optical conversion or physical
synchronization. Planning reads metadata only; scipy and pyxdf are lazy imports.
"""
import argparse
import csv
import io
import logging
import numbers
import re
from collections import Counter
from importlib import metadata
from pathlib import Path
import numpy as np
from .common import digest, file_hash, read_json, write_json
from .data import CSV_FIELDS, MODALITIES, STAGES, WindowDataset

PYXDF_VERSION = "1.17.5"
LABEL_STATUS = "legacy_stored_per_window_unverified_after_remap"
FIXED_POLICY = dict(window_seconds=10, stride_seconds=5,
                    target_rate_hz=dict(eeg=250, fnirs=25),
                    label_policy="preserve_legacy_window_labels")
RAW_LOADER = dict(synchronize_clocks=False, handle_clock_resets=False,
                  dejitter_timestamps=False)
CORRECTED_LOADER = dict(synchronize_clocks=True, handle_clock_resets=True,
                        dejitter_timestamps=True)
SOURCE_STREAM_LAYOUT = dict(eeg=("EEG",4,250), fnirs=("FNIRS",16,25),
                            markers=("Markers",1,0))
TASK_MARKERS = dict(NBack=(1,1),SART=(200,202),MIST=(300,302),GoNoGo=(400,402),REST=(4,4))


class SourceRejected(ValueError):
    """A failed input check excludes mapped runs without dropping CSV rows."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_file(value, name):
    path = Path(value)
    require(path.is_absolute() and path.is_file(), f"{name} must name an existing absolute file")
    return path.resolve()


def load_plan(path):
    """Validate mapping and legacy metadata without reading any signal file."""
    path = Path(path).resolve()
    plan_hash = file_hash(path)
    plan = read_json(path)
    require(plan.get("schema_version") == 1, "source plan schema_version=1 required")
    policy = plan.get("policy", {})
    for key, value in FIXED_POLICY.items():
        require(policy.get(key) == value, f"Unsupported policy: {key}")
    gaps = policy.get("max_interpolation_gap_seconds", {})
    require(set(gaps) == set(MODALITIES), "Gap policy must specify both modalities")
    for value in gaps.values():
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value > 0, "Interpolation gaps must be positive finite seconds")
    manifest_path = absolute_file(plan["legacy_manifest"], "legacy_manifest")
    csv_path = absolute_file(plan["legacy_window_csv"], "legacy_window_csv")
    manifest_hash, csv_hash = file_hash(manifest_path), file_hash(csv_path)
    legacy = read_json(manifest_path)
    require(legacy.get("schema_version") == 1 and legacy.get("kind") == "real", "Legacy real schema 1 required")
    scope = plan.get("scope_subject_ids", [])
    require(isinstance(scope, list) and scope and all(isinstance(x, str) and x for x in scope)
            and len(scope) == len(set(scope)), "Unique nonempty string scope_subject_ids required")
    manifest_runs = {str(r["run_id"]): r for r in legacy["runs"] if str(r["subject_id"]) in scope}
    require(len(manifest_runs) == sum(str(r["subject_id"]) in scope for r in legacy["runs"]), "Duplicate legacy run IDs")
    require({str(r["subject_id"]) for r in manifest_runs.values()} == set(scope), "Scope subjects missing from legacy manifest")
    csv_bytes = csv_path.read_bytes()
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    require(set(CSV_FIELDS).issubset(fields or []), "Legacy CSV missing required fields")
    rows = [r for r in reader if r["subject_id"] in scope]
    require(rows, "Empty scoped CSV")
    seen, positions = set(), set()
    for r in rows:
        require(r["paired_valid"] in ("0", "1"), "Invalid legacy paired_valid")
        run = manifest_runs.get(r["run_id"])
        require(run is not None and str(run["subject_id"]) == r["subject_id"]
                and run["block_info"] == r["block_info"], "Legacy CSV/run mismatch")
        require(r["block_info"] in STAGES, "Invalid legacy stage")
        index, position = int(r["source_index"]), int(r["window_in_run"])
        require(index >= 0 and index not in seen and position >= 0
                and (r["run_id"], position) not in positions, "Duplicate/negative source or window index")
        seen.add(index)
        positions.add((r["run_id"], position))
        require(int(r["label_2class"]) in (0, 1) and int(r["outer_test_fold"]) in range(1, 6), "Invalid legacy label/fold")
        require(r["paired_valid"] == "0" or not r["invalid_reasons"], "Legacy valid row has exclusion reasons")
        for modality, (_, n) in MODALITIES.items():
            require(int(r[modality+"_start"]) >= 0 and int(r[modality+"_end"])-int(r[modality+"_start"]) == n,
                    "Invalid legacy interval")
    require({r["run_id"] for r in rows} == set(manifest_runs), "Every scoped legacy run must have CSV rows")
    run_plans = plan.get("runs", [])
    require(isinstance(run_plans, list), "Plan runs must be a list")
    mappings = {str(r["run_id"]): r for r in run_plans}
    require(len(mappings) == len(run_plans) and set(mappings) == set(manifest_runs), "Plan must map or reject every scoped run exactly once")
    sources = plan.get("sources", {})
    require(isinstance(sources, dict), "Plan sources must be a dictionary")
    used_sources = set()
    for rid, r in mappings.items():
        old = manifest_runs[rid]
        require(str(r["subject_id"]) == str(old["subject_id"]) and r["block_info"] == old["block_info"], "Plan run metadata differs from legacy")
        if r.get("rejection_reason"):
            require(isinstance(r["rejection_reason"], str), "Rejection reason must be a string")
            continue
        source_id = r.get("source_id")
        require(source_id in sources, "Run source_id not defined")
        used_sources.add(source_id)
        for key in ("markers_stream_id", "start_marker_index", "end_marker_index"):
            require(isinstance(r.get(key), int) and not isinstance(r[key], bool) and r[key] >= 0, f"Invalid {key}")
        require(r["end_marker_index"] > r["start_marker_index"], "Marker end index must follow start")
        for key in ("expected_marker_start_value","expected_marker_end_value"):
            require(isinstance(r.get(key),int) and not isinstance(r[key],bool),f"Integer {key} required")
        tolerance=r.get("marker_raw_time_tolerance_seconds")
        require(isinstance(tolerance,(int,float)) and not isinstance(tolerance,bool)
                and np.isfinite(tolerance) and 0<tolerance<=.01,"Marker raw tolerance must be >0 and <=.01 seconds")
        require(isinstance(r.get("historical_blk"),dict) and r["historical_blk"],"Historical blk metadata required")
        for modality, (channels, _) in MODALITIES.items():
            item = r[modality]
            for key in ("stream_id", "start_index", "stop_index_exclusive"):
                require(isinstance(item.get(key), int) and not isinstance(item[key], bool) and item[key] >= 0, f"Invalid {modality} {key}")
            require(item["stop_index_exclusive"] > item["start_index"], "Empty source slice")
            for prefix in ("raw", "processed"):
                absolute_file(item[prefix+"_mat"], modality+" "+prefix+"_mat")
                require(isinstance(item.get(prefix+"_variable"), str) and item[prefix+"_variable"], "MAT variable required")
            if item.get("processed_reference_mat"):
                absolute_file(item["processed_reference_mat"], "processed_reference_mat")
                require(bool(item.get("processed_reference_variable")), "Processed reference variable required")
    require(set(sources) == used_sources, "Unused plan sources are not permitted")
    for source in sources.values():
        absolute_file(source["path"], "XDF source")
        sha = source.get("sha256", "")
        require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Lowercase SHA256 required")
        counts = source.get("expected_stream_samples", {})
        require(counts and all(str(k).isdigit() and isinstance(v, int) and v >= 0 for k, v in counts.items()), "Expected stream sample counts required")
    require(file_hash(path) == plan_hash and file_hash(manifest_path) == manifest_hash
            and file_hash(csv_path) == csv_hash, "Metadata changed during plan validation")
    return dict(path=path, plan=plan, legacy=legacy, fields=fields, rows=rows, mappings=mappings,
                input_hashes=dict(source_plan_sha256=plan_hash,legacy_manifest_sha256=manifest_hash,legacy_csv_sha256=csv_hash),
                manifest_path=manifest_path, csv_path=csv_path)


def stream_map(streams):
    result = {}
    for stream in streams:
        sid = stream["info"]["stream_id"]
        # pyxdf 1.17.5 sets an integer; tolerate the one-item metadata convention.
        if isinstance(sid, (list, tuple)):
            require(len(sid) == 1, "Ambiguous stream ID")
            sid = sid[0]
        if isinstance(sid,bool) or not ((isinstance(sid,numbers.Integral)) or
                (isinstance(sid,str) and re.fullmatch(r"[0-9]+",sid))):
            raise SourceRejected("invalid_noninteger_stream_id")
        sid = int(sid)
        require(sid not in result, "Duplicate loaded stream ID")
        result[sid] = stream
    return result


def equal_values(a, b):
    # Marker streams may contain strings; equal_nan cannot be applied to them.
    numeric_float = np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact)
    return np.array_equal(a, b, equal_nan=True) if numeric_float else np.array_equal(a, b)


def exact_sample_identity(raw, corrected, expected):
    """Fail if importer recovery trims or changes values in either load."""
    wanted = {int(k): v for k, v in expected.items()}
    if set(raw) != set(wanted) or set(corrected) != set(wanted):
        raise SourceRejected("stream_set_mismatch")
    for sid, n in wanted.items():
        a, b = np.asarray(raw[sid]["time_series"]), np.asarray(corrected[sid]["time_series"])
        ta, tb = np.asarray(raw[sid]["time_stamps"]), np.asarray(corrected[sid]["time_stamps"])
        if len(a) != n or len(b) != n or len(ta) != n or len(tb) != n:
            raise SourceRejected(f"sample_count_mismatch:stream={sid}:expected={n}:raw={len(a)}:corrected={len(b)}")
        if a.shape != b.shape or a.dtype != b.dtype or not equal_values(a, b):
            raise SourceRejected(f"sample_value_identity_mismatch:stream={sid}")


def checked_hash(path, expected):
    actual = file_hash(path)
    if actual != expected:
        raise SourceRejected("source_sha256_mismatch")
    return actual


def json_value(value):
    """Convert loader/MAT metadata to portable JSON without signal arrays."""
    if isinstance(value,np.ndarray):
        if value.dtype.names:
            if value.size != 1:
                raise SourceRejected("ambiguous_mat_blk_structure")
            return {name:json_value(value[name].item()) for name in value.dtype.names}
        if value.size==1:
            return json_value(value.item())
        return [json_value(x) for x in value.tolist()]
    if isinstance(value,np.void) and value.dtype.names:
        return {name:json_value(value[name]) for name in value.dtype.names}
    if isinstance(value,np.generic):
        return json_value(value.item())
    if isinstance(value,dict):
        return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_value(x) for x in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def loader_stream_summary(stream):
    t=np.asarray(stream["time_stamps"],dtype=float)
    diff=np.diff(t)
    info=stream["info"]
    rate=info.get("nominal_srate",[0])
    rate=float(rate[0] if isinstance(rate,(list,tuple)) else rate)
    gap_threshold=max(.1,5/rate) if rate>0 else .1
    gaps=np.flatnonzero(diff>gap_threshold)
    largest=gaps[np.argsort(diff[gaps])[-20:]] if len(gaps) else gaps
    clocks=np.asarray(stream.get("clock_times",[]),dtype=float)
    offsets=np.asarray(stream.get("clock_values",[]),dtype=float)
    bounds=lambda a: dict(count=len(a),first=float(a[0]) if len(a) else None,
        last=float(a[-1]) if len(a) else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)
    return json_value(dict(info={k:info.get(k) for k in ("stream_id","name","type","channel_count","nominal_srate",
        "effective_srate","segments","clock_segments")},sample_count=len(t),
        timestamps=dict(first=float(t[0]) if len(t) else None,last=float(t[-1]) if len(t) else None,
            min=float(t.min()) if len(t) else None,max=float(t.max()) if len(t) else None,
            nonfinite_count=int(np.count_nonzero(~np.isfinite(t))),backward_count=int(np.count_nonzero(diff<0)),repeat_count=int(np.count_nonzero(diff==0))),
        gaps=dict(threshold_seconds=gap_threshold,count=len(gaps),largest=[dict(before_index=int(i),gap_seconds=float(diff[i])) for i in largest]),
        clock_times=bounds(clocks),clock_values=bounds(offsets)))


def captured_load(pyxdf,path,config):
    warnings=[]
    class Handler(logging.Handler):
        def emit(self,record):
            if record.name=="pyxdf" or record.name.startswith("pyxdf."):
                warnings.append(dict(logger=record.name,level=record.levelname,message=record.getMessage()))
                # Adding a handler otherwise disables logging's lastResort
                # console fallback. Preserve that fallback when it was the
                # only existing warning output, without duplicate propagation.
                current=logging.getLogger(record.name)
                existing=False
                while current is not None:
                    existing |= any(h is not self for h in current.handlers)
                    if not current.propagate:
                        break
                    current=current.parent
                if not existing and logging.lastResort is not None and record.levelno>=logging.lastResort.level:
                    logging.lastResort.handle(record)
    logger=logging.getLogger("pyxdf")
    handler=Handler(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        streams,header=pyxdf.load_xdf(str(path),**config)
    except Exception as error:
        error.captured_pyxdf_warnings=warnings
        raise
    finally:
        logger.removeHandler(handler)
    return streams,header,warnings


def load_source(source):
    import pyxdf
    if metadata.version("pyxdf") != PYXDF_VERSION:
        raise RuntimeError(f"pyxdf=={PYXDF_VERSION} required")
    path = Path(source["path"])
    before = checked_hash(path, source["sha256"])
    details=dict(raw_warnings=[],corrected_warnings=[],
        warning_decision="retained_for_nonformal_preview_manual_review_required",loader_version=PYXDF_VERSION)
    try:
        raw, raw_header, raw_warnings = captured_load(pyxdf,path,RAW_LOADER)
    except (ValueError,OSError) as error:
        details["raw_warnings"]=getattr(error,"captured_pyxdf_warnings",[])
        error.loader_details=details
        raise
    details.update(raw_warnings=raw_warnings,raw_file_header=json_value(raw_header))
    checked_hash(path, before)
    try:
        corrected, corrected_header, corrected_warnings = captured_load(pyxdf,path,CORRECTED_LOADER)
    except (ValueError,OSError) as error:
        details["corrected_warnings"]=getattr(error,"captured_pyxdf_warnings",[])
        error.loader_details=details
        raise
    details.update(corrected_warnings=corrected_warnings,corrected_file_header=json_value(corrected_header))
    checked_hash(path, before)
    try:
        raw, corrected = stream_map(raw), stream_map(corrected)
        details.update(raw_stream_summaries={str(k):loader_stream_summary(v) for k,v in raw.items()},
            corrected_stream_summaries={str(k):loader_stream_summary(v) for k,v in corrected.items()})
        exact_sample_identity(raw, corrected, source["expected_stream_samples"])
    except SourceRejected as error:
        error.loader_details=details
        raise
    return raw, corrected, details


def increasing_timestamps(values, label):
    t = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise SourceRejected(f"corrected_timestamps_not_strictly_increasing:{label}")
    return t


def check_stream_header(stream, modality):
    """Use declared acquisition layout, separate from processed model layout."""
    info = stream.get("info", {})
    def scalar(key):
        value = info.get(key)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise SourceRejected(f"ambiguous_stream_header:{modality}:{key}")
            value = value[0]
        return value
    expected_type, channels, rate = SOURCE_STREAM_LAYOUT[modality]
    try:
        valid = (scalar("type") == expected_type and int(scalar("channel_count")) == channels
                 and float(scalar("nominal_srate")) == rate)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise SourceRejected(f"stream_header_layout_mismatch:{modality}")
    values = np.asarray(stream.get("time_series"))
    if values.ndim != 2 or values.shape[1] != channels:
        raise SourceRejected(f"stream_value_channel_count_mismatch:{modality}")
    return channels


def interpolate_supported(t, values, grid, max_gap, source_start=0):
    """Linear interpolation with audited brackets, no extrapolation/hold fill.

    Exact source times are direct samples, with left=right and weight=0.
    Other times require two neighbors within the maximum gap. Invalid positions
    contain NaN values, -1 source indices, NaN weights and False coverage.
    """
    t = increasing_timestamps(t, "interpolation")
    values, grid = np.asarray(values), np.asarray(grid, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == len(t), "Interpolation requires channels x samples")
    require(grid.ndim == 1 and np.isfinite(grid).all() and np.all(np.diff(grid) > 0), "Invalid target grid")
    pos = np.searchsorted(t, grid, side="left")
    safe = np.minimum(pos, len(t)-1)
    direct = (pos < len(t)) & (t[safe] == grid)
    bracket = (pos > 0) & (pos < len(t)) & ~direct
    left, right = np.maximum(pos-1, 0), safe.copy()
    left[direct] = safe[direct]
    gap = t[right]-t[left]
    valid = direct | (bracket & (gap <= max_gap))
    weight = np.full(len(grid), np.nan, dtype=np.float64)
    weight[direct] = 0
    weight[bracket] = (grid[bracket]-t[left[bracket]])/gap[bracket]
    out = np.full((values.shape[0], len(grid)), np.nan, dtype=np.float32)
    if np.any(valid):
        out[:,valid] = (values[:,left[valid]].astype(np.float64)*(1-weight[valid]) +
                       values[:,right[valid]].astype(np.float64)*weight[valid]).astype(np.float32)
        valid &= np.isfinite(out).all(axis=0)
    out[:,~valid] = np.nan
    audit = dict(grid_timestamps=grid, coverage=valid,
                 direct_source=direct & valid, interpolation_mask=valid & ~direct,
                 left_source_index=np.where(valid,left+source_start,-1).astype(np.int64),
                 right_source_index=np.where(valid,right+source_start,-1).astype(np.int64),
                 right_weight=np.where(valid,weight,np.nan),
                 left_timestamp=np.where(valid,t[left],np.nan),
                 right_timestamp=np.where(valid,t[right],np.nan))
    return out, valid, audit


def mat_array(item, prefix, hashes, cache):
    from scipy.io import loadmat
    path = Path(item[prefix+"_mat"]).resolve()
    variable = item[prefix+"_variable"]
    if path not in cache:
        before = file_hash(path)
        if str(path) in hashes and hashes[str(path)] != before:
            raise SourceRejected("mat_changed_before_load")
        loaded = loadmat(path)
        if file_hash(path) != before:
            raise SourceRejected("mat_changed_during_load")
        hashes[str(path)] = before
        cache[path] = loaded
    try:
        a = np.asarray(cache[path][variable])
    except KeyError as error:
        raise SourceRejected(f"mat_variable_missing:{prefix}:{variable}") from error
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise SourceRejected(f"invalid_mat_channels_by_samples:{prefix}")
    return a


def prepare_run(mapping, raw, corrected, hashes, mat_cache):
    markers_id = mapping["markers_stream_id"]
    if markers_id not in corrected or markers_id not in raw:
        raise SourceRejected("marker_stream_missing")
    if len({markers_id,mapping["eeg"]["stream_id"],mapping["fnirs"]["stream_id"]}) != 3:
        raise SourceRejected("modality_marker_stream_ids_not_distinct")
    check_stream_header(raw[markers_id],"markers")
    check_stream_header(corrected[markers_id],"markers")
    marker_t = increasing_timestamps(corrected[markers_id]["time_stamps"], "markers")
    start_index, end_index = mapping["start_marker_index"], mapping["end_marker_index"]
    if end_index >= len(marker_t):
        raise SourceRejected("marker_index_out_of_bounds")
    # The EEG raw MAT is the source of historical task boundaries. A plan is a
    # mapping proposal and must agree with the actual stored blk structure.
    mat_array(mapping["eeg"],"raw",hashes,mat_cache)
    eeg_mat=mat_cache[Path(mapping["eeg"]["raw_mat"]).resolve()]
    blk=json_value(eeg_mat.get("blk"))
    if not isinstance(blk,dict) or not blk:
        raise SourceRejected("raw_eeg_mat_blk_missing")
    historical=json_value(mapping["historical_blk"])
    if blk != historical:
        raise SourceRejected("historical_blk_source_metadata_mismatch")
    stage=mapping["block_info"]
    task="REST" if stage.startswith("rest") else ("NBack","SART","MIST","GoNoGo")[(int(stage[5:])-1)//3]
    expected=(mapping["expected_marker_start_value"],mapping["expected_marker_end_value"])
    if blk.get("Task") != task or expected != TASK_MARKERS[task]:
        raise SourceRejected("stage_task_marker_code_mismatch")
    if str(blk.get("source_xdf")) != str(mapping["source_id"]):
        raise SourceRejected("blk_source_xdf_mismatch")
    for stream in (raw[markers_id],corrected[markers_id]):
        marker_values=np.asarray(stream["time_series"])
        if marker_values[start_index,0] != expected[0] or marker_values[end_index,0] != expected[1]:
            raise SourceRejected("marker_sample_value_mismatch")
    raw_marker_t=np.asarray(raw[markers_id]["time_stamps"],dtype=float)
    if not np.isfinite(raw_marker_t).all():
        raise SourceRejected("raw_marker_timestamps_nonfinite")
    residuals=[]
    for index,key in ((start_index,"start_time"),(end_index,"end_time")):
        try:
            historical_time=float(blk[key])
        except (TypeError,ValueError,KeyError) as error:
            raise SourceRejected("invalid_blk_boundary_time") from error
        if not np.isfinite(historical_time):
            raise SourceRejected("invalid_blk_boundary_time")
        distances=np.abs(raw_marker_t-historical_time)
        if int(np.argmin(distances)) != index or np.count_nonzero(distances==distances[index]) != 1:
            raise SourceRejected("marker_index_not_unique_nearest_to_blk")
        residual=float(raw_marker_t[index]-historical_time)
        if abs(residual)>mapping["marker_raw_time_tolerance_seconds"]:
            raise SourceRejected("marker_raw_time_blk_residual_exceeds_tolerance")
        residuals.append(residual)
    start_time, end_time = float(marker_t[start_index]), float(marker_t[end_index])
    modalities, diagnostics = {}, {}
    for modality, (channels, _) in MODALITIES.items():
        item = mapping[modality]
        sid, first, stop = item["stream_id"], item["start_index"], item["stop_index_exclusive"]
        if sid not in raw or sid not in corrected:
            raise SourceRejected(f"stream_missing:{modality}")
        raw_channels = check_stream_header(raw[sid],modality)
        check_stream_header(corrected[sid],modality)
        full_t = increasing_timestamps(corrected[sid]["time_stamps"], modality)
        if stop > len(full_t):
            raise SourceRejected(f"source_slice_out_of_bounds:{modality}")
        raw_values = np.asarray(raw[sid]["time_series"])[first:stop].T
        original = mat_array(item, "raw", hashes, mat_cache)
        processed = mat_array(item, "processed", hashes, mat_cache)
        raw_shape, processed_shape = (raw_channels,stop-first), (channels,stop-first)
        if raw_values.shape != raw_shape or original.shape != raw_shape or processed.shape != processed_shape:
            raise SourceRejected(f"raw_processed_mat_shape_mismatch:{modality}")
        if not np.isfinite(raw_values).all() or not np.isfinite(original).all():
            raise SourceRejected(f"raw_xdf_or_mat_nonfinite:{modality}")
        if not np.array_equal(original, raw_values, equal_nan=True):
            raise SourceRejected(f"raw_mat_xdf_value_mismatch:{modality}")
        if not np.isfinite(processed).all():
            raise SourceRejected(f"processed_mat_nonfinite:{modality}")
        if item.get("processed_reference_mat"):
            reference = mat_array(item, "processed_reference", hashes, mat_cache)
            if reference.shape != processed_shape or not np.array_equal(reference, processed, equal_nan=True):
                raise SourceRejected(f"processed_reference_value_mismatch:{modality}")
        t = full_t[first:stop]
        modalities[modality] = (t, processed, first)
        raw_t = np.asarray(raw[sid]["time_stamps"])[first:stop]
        diagnostics[modality] = dict(source_stream_id=sid, source_start_index=first, source_stop_index_exclusive=stop,
            raw_channel_count=raw_channels,processed_channel_count=channels,
            source_sample_count=stop-first, raw_first_timestamp=float(raw_t[0]), raw_last_timestamp=float(raw_t[-1]),
            corrected_first_timestamp=float(t[0]), corrected_last_timestamp=float(t[-1]),
            corrected_first_minus_marker_start_seconds=float(t[0]-start_time),
            corrected_last_minus_marker_end_seconds=float(t[-1]-end_time),
            corrected_max_adjacent_gap_seconds=float(np.max(np.diff(t))) if len(t)>1 else None)
    diagnostics.update(marker_start_seconds=start_time, marker_end_seconds=end_time,
        marker_duration_seconds=end_time-start_time,actual_historical_blk=blk,
        global_stage=stage,historical_block_id=blk.get("BlockID"),historical_block_id_remapped=blk.get("BlockID")!=stage,
        verified_task=task,expected_marker_values=list(expected),
        marker_mapping_raw_residuals_seconds=residuals,
        planned_marker_mapping_raw_residuals_seconds=mapping.get("marker_mapping_raw_residuals_seconds"),
        marker_nearest_indices_verified=True)
    return start_time, end_time, modalities, diagnostics


def excluded_row(row, reasons):
    result = dict(row)
    old = row["invalid_reasons"]
    additions = list(dict.fromkeys(reasons))
    if old:
        additions.insert(0, old)
    result["paired_valid"] = "0" if row["paired_valid"] == "0" or additions else "1"
    result["invalid_reasons"] = ";".join(additions)
    return result


def bundle_fingerprint(manifest, csv_path, output):
    """Same algorithm as WindowDataset.fingerprint, also for all-invalid previews."""
    payload = dict(manifest)
    payload.pop("data_root", None)
    files = {}
    for run in manifest["runs"]:
        for modality in MODALITIES:
            for key in ("array", "coverage"):
                relative = run[modality][key]
                files[relative] = file_hash(output/relative)
    return digest(dict(manifest=payload, csv=file_hash(csv_path), files=files))


def execute(context, output):
    """Generate a new preview package. Existing directories are never reused."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preview output already exists; choose a new directory")
    if metadata.version("pyxdf") != PYXDF_VERSION:
        raise RuntimeError(f"pyxdf=={PYXDF_VERSION} required")
    p = context["plan"]
    bound_inputs = context["input_hashes"]
    for name, path in (("source_plan_sha256",context["path"]), ("legacy_manifest_sha256",context["manifest_path"]), ("legacy_csv_sha256",context["csv_path"])):
        require(file_hash(path) == bound_inputs[name], "Metadata changed after plan validation")
    code_hashes = {path.name:file_hash(path) for path in
                   (Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("common.py"))}
    config = dict(policy=p["policy"], pyxdf_version=PYXDF_VERSION, raw_loader=RAW_LOADER, corrected_loader=CORRECTED_LOADER,
                  interpolation="linear_two_sided_or_exact_source_no_extrapolation", coverage_semantics="finite_supported_brackets_not_raw_direct")
    versions = {name:metadata.version(name) for name in ("numpy", "scipy", "pyxdf")}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", dict(state="running", purpose="nonformal_clock_corrected_preview"))
    source_results, source_checks, source_rejections, mat_hashes, mat_cache = {}, {}, {}, {}, {}
    # Bind all mapped MAT inputs, including references in subsequently rejected
    # runs. This is signal-file I/O and therefore only occurs under --execute.
    for mapping in context["mappings"].values():
        for modality in MODALITIES:
            for key, value in mapping.get(modality, {}).items():
                if key.endswith("_mat") and value:
                    path = Path(value).resolve()
                    mat_hashes[str(path)] = file_hash(path)
    for sid, source in p["sources"].items():
        actual_hash = file_hash(source["path"])
        try:
            source_results[sid] = load_source(source)
            loader_details=source_results[sid][2]
            source_checks[sid] = dict(expected_sha256=source["sha256"], actual_sha256=file_hash(source["path"]),
                expected_stream_samples=source["expected_stream_samples"], raw_corrected_sample_identity="verified",**loader_details)
        except (SourceRejected, ValueError, OSError) as error:
            source_rejections[sid] = str(error)
            source_checks[sid] = dict(expected_sha256=source["sha256"], actual_sha256=actual_hash,
                expected_stream_samples=source["expected_stream_samples"],rejection_reason=str(error),
                **getattr(error,"loader_details",{}))
    runs, rows, diagnostics, rejections = [], [], [], []
    grouped = {rid:[] for rid in context["mappings"]}
    for row in context["rows"]:
        grouped[row["run_id"]].append(row)
    row_results = {}
    for rid, mapping in context["mappings"].items():
        run_rows = grouped[rid]
        reason = mapping.get("rejection_reason")
        source_id = mapping.get("source_id")
        prepared = None
        if not reason:
            reason = source_rejections.get(source_id)
            if not reason:
                try:
                    raw, corrected = source_results[source_id][:2]
                    prepared = prepare_run(mapping, raw, corrected, mat_hashes, mat_cache)
                except (SourceRejected, ValueError, OSError) as error:
                    reason = str(error)
        directory = "runs/"+digest(dict(run_id=rid))[:20]
        (output/directory).mkdir(parents=True)
        run = dict(run_id=rid, subject_id=str(mapping["subject_id"]), block_info=mapping["block_info"])
        details = dict(run_id=rid, source_id=source_id, source_indices=[int(x["source_index"]) for x in run_rows], rejection_reason=reason)
        masks = {}
        for modality, (channels, length) in MODALITIES.items():
            rate = p["policy"]["target_rate_hz"][modality]
            n = max(max(int(r[modality+"_end"]) for r in run_rows),
                    max(int(r["window_in_run"])*5*rate+length for r in run_rows))
            if prepared is None:
                arr, mask = np.full((channels,n),np.nan,dtype=np.float32), np.zeros(n,dtype=bool)
                audit = dict(grid_timestamps=np.full(n,np.nan), coverage=mask,
                    direct_source=mask.copy(), interpolation_mask=mask.copy(),
                    left_source_index=np.full(n,-1,dtype=np.int64), right_source_index=np.full(n,-1,dtype=np.int64),
                    right_weight=np.full(n,np.nan), left_timestamp=np.full(n,np.nan), right_timestamp=np.full(n,np.nan))
            else:
                start_time, end_time, modalities, run_diag = prepared
                grid = start_time+np.arange(n,dtype=np.float64)/rate
                t, values, source_start = modalities[modality]
                arr, mask, audit = interpolate_supported(t,values,grid,p["policy"]["max_interpolation_gap_seconds"][modality],source_start)
                # Points outside the marked interval are never eligible support.
                mask &= grid < end_time
                arr[:,~mask] = np.nan
                for flag in ("coverage", "direct_source", "interpolation_mask"):
                    audit[flag] &= mask
                for index in ("left_source_index", "right_source_index"):
                    audit[index][~mask] = -1
                for value in ("right_weight", "left_timestamp", "right_timestamp"):
                    audit[value][~mask] = np.nan
                details[modality] = dict(**run_diag[modality], grid_samples=n, target_rate_hz=rate,
                    grid_first_timestamp=float(grid[0]), grid_last_timestamp=float(grid[-1]),
                    supported_sample_count=int(mask.sum()), direct_source_sample_count=int(audit["direct_source"].sum()),
                    interpolated_sample_count=int(audit["interpolation_mask"].sum()))
            masks[modality] = mask
            array_path, mask_path, audit_path = f"{directory}/{modality}.npy", f"{directory}/{modality}_coverage.npy", f"{directory}/{modality}_interpolation.npz"
            np.save(output/array_path,arr,allow_pickle=False)
            np.save(output/mask_path,mask,allow_pickle=False)
            np.savez_compressed(output/audit_path,**audit)
            run[modality] = dict(array=array_path,coverage=mask_path,interpolation_audit=audit_path)
        if prepared:
            details.update({k:v for k,v in prepared[3].items() if k not in MODALITIES})
            details["modality_grid_start_difference_seconds"] = 0.0
        for row in run_rows:
            reasons = []
            if reason:
                reasons.append("time_axis_source_rejected:"+reason)
            elif prepared:
                offset = int(row["window_in_run"])*5
                if prepared[0]+offset+10 > prepared[1]:
                    reasons.append("window_exceeds_marker_end")
                for modality, (_,length) in MODALITIES.items():
                    start = int(row["window_in_run"])*5*p["policy"]["target_rate_hz"][modality]
                    if not np.all(masks[modality][start:start+length]):
                        reasons.append("unsupported_interpolation:"+modality)
            revised = excluded_row(row,reasons)
            for modality, (_,length) in MODALITIES.items():
                start = int(row["window_in_run"])*5*p["policy"]["target_rate_hz"][modality]
                revised[modality+"_start"], revised[modality+"_end"] = str(start), str(start+length)
            row_results[row["source_index"]] = revised
            if revised["paired_valid"] == "0":
                rejections.append(dict(source_index=int(row["source_index"]),run_id=rid,source_id=source_id,
                    legacy_paired_valid=row["paired_valid"]=="1",legacy_invalid_reasons=row["invalid_reasons"],
                    new_reasons=reasons,invalid_reasons=revised["invalid_reasons"]))
        runs.append(run)
        diagnostics.append(details)
    rows = [row_results[r["source_index"]] for r in context["rows"]]
    # Detect mutation even for cached inputs and metadata before freezing output.
    for name, path in (("source_plan_sha256",context["path"]), ("legacy_manifest_sha256",context["manifest_path"]), ("legacy_csv_sha256",context["csv_path"])):
        if file_hash(path) != bound_inputs[name]:
            raise RuntimeError("Input metadata changed during execution")
    for path, sha in mat_hashes.items():
        if file_hash(path) != sha:
            raise RuntimeError("MAT input changed during execution")
    for sid, source in p["sources"].items():
        if file_hash(source["path"]) != source_checks[sid]["actual_sha256"]:
            raise RuntimeError("XDF input changed during execution")
    require({path.name:file_hash(path) for path in
             (Path(__file__),Path(__file__).with_name("data.py"),Path(__file__).with_name("common.py"))} == code_hashes,
            "Code changed during execution")
    version_payload = dict(**bound_inputs,policy=p["policy"],source_checks=source_checks,mat_sha256=mat_hashes,
                           loader_configuration=config,code_sha256=code_hashes,dependency_versions=versions)
    data_version = "time-axis-v2-preview-"+digest(version_payload)[:24]
    manifest = dict(schema_version=1,kind="real",data_version=data_version,data_root=".",window_index="windows.csv",
        source=dict(time_axis="clock_corrected_preview",preprocessing="legacy_processed_signals_resampled",
                    fnirs_conversion="legacy_unverified",labels=LABEL_STATUS,
                    source_plan_sha256=bound_inputs["source_plan_sha256"],policy_sha256=digest(p["policy"]),
                    preview_receipt="preview_receipt.json"),runs=runs)
    write_json(output/"manifest.json",manifest)
    with (output/"windows.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=context["fields"])
        writer.writeheader()
        writer.writerows(rows)
    fingerprint = bundle_fingerprint(manifest,output/"windows.csv",output)
    before = sum(r["paired_valid"]=="1" for r in context["rows"])
    after = sum(r["paired_valid"]=="1" for r in rows)
    check = dict(state="not_loadable_no_common_valid_windows", fingerprint_algorithm="WindowDataset.fingerprint")
    if after:
        dataset = WindowDataset(output/"manifest.json")
        require(dataset.fingerprint() == fingerprint,"Preview fingerprint mismatch")
        for w in dataset.windows:
            if w.paired_valid:
                for modality in MODALITIES:
                    dataset.slice(w,modality)
        check = dict(state="verified",eligible_windows_checked=after,fingerprint_algorithm="WindowDataset.fingerprint")
    invariants = ("source_index","subject_id","run_id","block_info","window_in_run","label_2class","outer_test_fold")
    require(all(all(old[k]==new[k] for k in invariants) and not (old["paired_valid"]=="0" and new["paired_valid"]=="1")
                for old,new in zip(context["rows"],rows)),"Legacy supervision or invalidity changed")
    report = dict(before_eligible_count=before,after_eligible_count=after,total_rows=len(rows),
        newly_excluded_count=before-after,legacy_invalid_count=len(rows)-before,
        exclusion_reason_counts=dict(Counter(reason for r in rejections for reason in r["new_reasons"])),
        source_checks=source_checks,source_rejections=source_rejections,run_diagnostics=diagnostics,rejections=rejections,
        labels=LABEL_STATUS,physical_synchronization="not_established",window_dataset_interface=check)
    write_json(output/"report.json",report)
    sidecars = {str(path.relative_to(output)).replace("\\","/"):file_hash(path)
                for path in output.rglob("*_interpolation.npz")}
    receipt = dict(kind="nonformal_preview_receipt",data_fingerprint=fingerprint,data_version=data_version,
        time_axis="pending",fnirs_conversion="pending",labels="not_revalidated",label_policy=LABEL_STATUS,
        physical_synchronization="not_established",formal_acceptance="pending",
        checks=dict(raw_corrected_identity_by_source=source_checks,window_dataset_interface=check,
                    supervision_and_original_invalidity_preserved=True),
        bindings=version_payload,source_plan=p,input_paths=dict(source_plan=str(context["path"]),
                    legacy_manifest=str(context["manifest_path"]),legacy_window_csv=str(context["csv_path"])),
        interpolation_audit_sha256=sidecars,report_sha256=file_hash(output/"report.json"),all_rejections=rejections)
    write_json(output/"preview_receipt.json",receipt)
    write_json(output/"status.json",dict(state="completed",purpose="nonformal_clock_corrected_preview",
        data_fingerprint=fingerprint,data_version=data_version,after_eligible_count=after))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    context = load_plan(args.source_plan)
    if not args.execute:
        import json
        print(json.dumps(dict(state="plan_validated_metadata_only",execution_required=True,
            scoped_subject_ids=context["plan"]["scope_subject_ids"],runs=len(context["mappings"]),
            windows=len(context["rows"]),legacy_eligible_count=sum(r["paired_valid"]=="1" for r in context["rows"]),
            sources=len(context["plan"]["sources"]),output=str(args.output.resolve())),indent=2))
        return
    report = execute(context,args.output)
    print(f"Preview complete: {report['after_eligible_count']}/{report['before_eligible_count']} eligible; formal pending")


if __name__ == "__main__":
    main()
