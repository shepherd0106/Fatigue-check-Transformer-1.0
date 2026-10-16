"""Fixed, inference-only EEG diagnostic. Imports only verified original server sources."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

DATA_CHECKSUMS_SHA256 = "e226148b0fbf36e80a733a3d39d49d0410e547687d8dd75caa144d2f09adde8f"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(path):
    p = Path(path).absolute()
    for q in [p, *p.parents]:
        if q.is_symlink() or q.exists() and getattr(q.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse paths forbidden: " + str(q))
    return p.resolve()


def child(root, name):
    root = safe(root)
    p = safe(root / name)
    if root not in p.parents:
        raise ValueError("Artifact path escapes root")
    return p


def validate_output(output, protected):
    root = safe(output)
    for value in protected:
        p = safe(value)
        if root == p or root in p.parents or p in root.parents:
            raise ValueError("Output overlaps input, data, code or history: " + str(p))
    if root.exists():
        raise ValueError("Output already exists; never overwrite or resume")
    return root


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def executed_counts(stream, steps, allowed_indices):
    """Only rows [0, executed_steps) count; saved research suffix is unexecuted."""
    if type(steps) is not int or not 0 < steps <= len(stream):
        raise ValueError("Invalid executed prefix")
    allowed = set(allowed_indices)
    counts = Counter()
    for row in stream[:steps]:
        if len(row) != 64:
            raise ValueError("Sampling batch must contain 64 indices")
        for v in row:
            i = int(v)
            if i != v or i not in allowed:
                raise ValueError("Sampling index outside guarded training universe")
            counts[i] += 1
    return counts


def describe(values):
    values = list(values)
    if not values:
        return dict(count=0, min=None, median=None, max=None, zero=0)
    return dict(count=len(values), min=min(values), median=statistics.median(values), max=max(values), zero=sum(v == 0 for v in values))


def coverage(counts, universe, probe, subject_of):
    universe = list(universe)
    people = defaultdict(list)
    for i in universe:
        people[subject_of(i)].append(i)
    def one(ids):
        seen = sum(counts[i] > 0 for i in ids)
        eligible = set(ids)
        selected = [i for i in probe if i in eligible]
        return dict(eligible_windows=len(ids), unique_sampled=seen,
                    unique_window_coverage=seen / len(ids) if ids else None,
                    draws=sum(counts[i] for i in ids), probe_exposure=describe(counts[i] for i in selected))
    return dict(total=one(universe), per_subject={s: one(ids) for s, ids in sorted(people.items(), key=lambda x: int(x[0]))})


def compare_rows(rows, csv_path, expected_step, rtol=1e-4, atol=1e-5):
    """Strict metadata/order and finite old-logit comparison; reference is the tolerance RHS."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        old = list(csv.DictReader(stream))
    if len(rows) != len(old):
        raise ValueError("Saved prediction length differs")
    maximum = 0.0
    for a, b in zip(rows, old):
        for k in ("source_index", "subject_id", "run_id", "window_in_run", "block_info", "label"):
            if str(a[k]) != b[k]:
                raise ValueError("Saved prediction metadata/order differs: " + k)
        if int(b["step"]) != expected_step or b["model"] != "C_E" or b["fold"] != "2" or b["seed"] != "17":
            raise ValueError("Saved prediction spec/step differs")
        for value, name in zip(a["logits"], ("logits0", "logits1")):
            reference = float(b[name])
            if not math.isfinite(value) or not math.isfinite(reference) or abs(value - reference) > atol + rtol * abs(reference):
                raise ValueError("Standard eval logits failed fixed tolerance")
            maximum = max(maximum, abs(value - reference))
    return dict(passed=True, rows=len(rows), max_absolute_error=maximum, rtol=rtol, atol=atol, reference=str(csv_path))


def bootstrap(code):
    """Hash code BEFORE import. Never import source_snapshot or extracted review ZIP code."""
    code = safe(code)
    expected = read(Path(__file__).with_name("trusted_sources.json"))
    actual = {name: sha(child(code, name)) for name in expected}
    if actual != expected:
        raise ValueError("Original code does not match reviewed fixed source inventory")
    package = {p.relative_to(code).as_posix() for p in (code / "fatigue_training").rglob("*")
               if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    if package != {name for name in expected if name.startswith("fatigue_training/")}:
        raise ValueError("Unexpected original package sources")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    import modality_experiment as m
    if safe(m.__file__) != code / "modality_experiment.py":
        raise ValueError("Imported module outside original code root")
    m.source_binding()
    return m, expected


def readonly_context(args, m, inventory):
    root = safe(args.results)
    if (root / "writer.lock").exists():
        raise ValueError("Original result writer active")
    prepared = read(root / "prepared.json")
    binding = prepared["binding"]
    for name, key in (("manifest", "manifest_path"), ("baseline", "baseline_root"), ("preflight", "preflight_path")):
        bound = safe(binding[key])
        given = getattr(args, name)
        if given is not None and safe(given) != bound:
            raise ValueError("Only original bound " + name + " path is permitted")
        setattr(args, name, str(bound))
    # Snapshot is evidence only: never added to sys.path or executed.
    snapshot = root / "source_snapshot"
    expected_snapshot = {k: v for k, v in inventory.items() if k != "test_eeg_readonly.py"}
    actual_snapshot = {p.relative_to(snapshot).as_posix(): sha(p) for p in snapshot.rglob("*")
                       if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    required = {"fatigue_diagnostics.py", "diagnostics_protocol_v1.json", "frozen_source_manifest.json",
                "modality_models.py", "modality_experiment.py", "modality_protocol_v1.json", "test_modality_experiment.py"}
    required.update("fatigue_training/" + n for n in read(safe(args.code) / "frozen_source_manifest.json")["files"])
    if actual_snapshot != {k: expected_snapshot[k] for k in required}:
        raise ValueError("Original source snapshot missing/changed")
    dataset, norms, lists, selection, refs, recomputed = m.metadata_context(args)
    if recomputed != binding or read(root / "selections.json") != selection or read(root / "normalization.json") != {str(k): v for k, v in norms.items()} or read(root / "references.json") != refs:
        raise ValueError("Prepared metadata/bindings changed")
    m.anchors(args, binding)
    ledger = read(root / "ledger.json")
    budget = object.__new__(m.Budget)  # validate ONLY: constructor can write a ledger.
    budget.path = root / "ledger.json"
    budget.state = ledger
    budget.validate()
    if ledger["binding"] != binding or ledger["runs"] != 12 or any(s["state"] != "completed" for s in ledger["slots"].values()):
        raise ValueError("Expected all twelve original runs completed")
    results = {}
    for spec in m.specs():
        slot = ledger["slots"][spec["key"]]
        if slot["spec"] != spec:
            raise ValueError("Ledger spec differs")
        results[spec["key"]] = m.verify_completed(root, spec, binding, slot["steps"])
    return dataset, norms, lists, binding, results, ledger


def metadata_snapshot(args, m, dataset, binding, inventory):
    """Pin status files themselves, so rehashing their inventories cannot accept replacement."""
    root = safe(args.results)
    paths = [root / name for name in ("prepared.json", "ledger.json", "normalization.json", "selections.json", "references.json")]
    paths += [root / "runs" / spec["key"] / "status.json" for spec in m.specs()]
    paths += [safe(args.code) / name for name in inventory]
    paths += [root / "source_snapshot" / name for name in inventory]
    paths += [Path(args.manifest), dataset.index_path, Path(args.preflight)]
    paths += [safe(args.baseline) / name for name in binding["baseline"]["files"]]
    for fold in (2, 3, 4, 5):
        folder = m.old_c_folder(args.baseline, fold)
        status = read(folder / "status.json")
        paths += [folder / "status.json", folder / "result.json", folder / f"attempt{status['attempt']}/initialization.json",
                  Path(binding["folds"][str(fold)]["normalization"]["path"])]
    return {str(safe(path)): sha(safe(path)) for path in paths}


def data_snapshot(dataset, indices, allowed):
    """Hash only selected training EEG runs and masks, against the frozen uploaded list."""
    from fatigue_training.development import SubjectGuard
    guard = SubjectGuard(dataset, allowed)
    indices = guard._check(indices)
    checksums = child(dataset.root, "checksums.json")
    if sha(checksums) != DATA_CHECKSUMS_SHA256:
        raise ValueError("Frozen uploaded checksums.json changed/missing")
    entries = read(checksums)
    listed = {row["path"]: row for row in entries}
    if len(listed) != len(entries):
        raise ValueError("Duplicate data checksum entries")
    names = set()
    for i in indices:
        w = dataset.windows[int(i)]
        info = dataset.runs[w.run_id]["eeg"]
        names.update(info[k] for k in ("array", "coverage") if info.get(k))
    report = {str(checksums): DATA_CHECKSUMS_SHA256}
    for name in sorted(names):
        path = child(dataset.root, name)
        row = listed.get(name)
        if row is None or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Training EEG/coverage file differs from uploaded data checksum: " + name)
        report[str(path)] = row["sha256"]
    return report


def export_audit(output):
    paths = sorted(p for p in output.iterdir() if p.is_file() and p.name not in ("audit_index.json", "eeg_readonly_audit.zip"))
    write_new(output / "audit_index.json", {p.name: sha(p) for p in paths})
    with zipfile.ZipFile(output / "eeg_readonly_audit.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [output / "audit_index.json"]:
            archive.write(path, path.name)
    print("Upload audit: " + str(output / "eeg_readonly_audit.zip"), flush=True)


def sampling_report(args, m, dataset, lists, binding, results):
    import numpy as np
    from fatigue_training.development import SubjectGuard
    reports = {}
    specs = [s for s in m.specs() if s["phase"] in ("tiny", "research")]
    for spec in specs:
        folder = safe(args.results) / "runs" / spec["key"]
        status = read(folder / "status.json")
        path = folder / "sampling_indices.npy"
        if sha(path) != status["artifacts"]["sampling_indices.npy"]:
            raise ValueError("Sampling SHA mismatch before mmap")
        stream = np.load(path, mmap_mode="r", allow_pickle=False)
        if stream.dtype != np.dtype("int64") or stream.shape != (spec["max_steps"], 64) or stream.flags.writeable:
            raise ValueError("Invalid read-only sampling array")
        fold = spec["fold"]
        allowed = lists[fold]["tiny_subjects"] if spec["phase"] == "tiny" else binding["folds"][str(fold)]["train"]
        guard = SubjectGuard(dataset, allowed)
        universe = lists[fold]["tiny"] if spec["phase"] == "tiny" else guard.indices(allowed)
        steps = results[spec["key"]]["executed_steps"]
        counts = executed_counts(stream, steps, universe)
        guard._check(list(counts))
        summary = coverage(counts, universe, lists[fold]["probe"], lambda i: dataset.windows[int(i)].subject_id)
        summary.update(executed_steps=steps, saved_steps=len(stream), unexecuted_suffix_steps=len(stream) - steps,
                       sampling_sha256=sha(path), label_draws={str(label): sum(v for i, v in counts.items() if dataset.windows[i].label_2class == label) for label in (0, 1)})
        # Tiny denominator above is the intended tiny training universe; expose its
        # limited coverage of all eighteen training people separately.
        if spec["phase"] == "tiny":
            train = binding["folds"][str(fold)]["train"]
            summary["all_18_training_people"] = coverage(counts, dataset.indices(train), lists[fold]["probe"], lambda i: dataset.windows[int(i)].subject_id)
        reports[spec["key"]] = summary
    return reports


def input_report(dataset, indices, stats, allowed):
    """Window/channel summaries. Repeated overlap samples count once PER selected window."""
    import numpy as np
    from fatigue_training.development import SubjectGuard
    guard = SubjectGuard(dataset, allowed)
    indices = guard._check(indices)
    st = stats["modalities"]["eeg"]
    mean = np.asarray(st["mean"], dtype=np.float64)[:, None]
    scale = np.asarray(st["std"], dtype=np.float64)[:, None]
    if mean.shape != (4, 1) or scale.shape != (4, 1) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid original eighteen-person normalization")
    rows = []
    for i in indices:
        w = dataset.windows[int(i)]
        # Guard checked metadata BEFORE loading arrays. arrays() uses mmap='r'.
        arr, mask = dataset.arrays(w.run_id, "eeg")
        if arr.flags.writeable or w.eeg_end > arr.shape[1] or mask is not None and not np.all(mask[w.eeg_start:w.eeg_end]):
            raise ValueError("Nonreadonly, out-of-bounds or uncovered EEG window")
        raw = np.asarray(arr[:, w.eeg_start:w.eeg_end], dtype=np.float64)
        for space, values in (("raw", raw), ("normalized", ((raw - mean) / scale).astype(np.float32).astype(np.float64))):
            for c, channel in enumerate(values):
                finite = channel[np.isfinite(channel)]
                nbad = int(channel.size - finite.size)
                rows.append(dict(index=int(i), source_index=w.source_index, subject_id=w.subject_id,
                    label=w.label_2class, space=space, channel=c, samples=int(channel.size), nonfinite=nbad,
                    mean=float(finite.mean()) if finite.size else None,
                    std=float(finite.std(ddof=0)) if finite.size else None,
                    rms=float(np.sqrt(np.mean(finite * finite))) if finite.size else None,
                    maxabs=float(np.max(np.abs(finite))) if finite.size else None,
                    flat=bool(finite.size == channel.size and np.ptp(finite) == 0)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["subject_id"], row["space"], row["channel"])].append(row)
    summaries = []
    for (person, space, channel), group in grouped.items():
        summaries.append(dict(subject_id=person, space=space, channel=channel, windows=len(group),
            nonfinite=sum(r["nonfinite"] for r in group), flat_windows=sum(r["flat"] for r in group),
            **{k: describe(r[k] for r in group if r[k] is not None) for k in ("mean", "std", "rms", "maxabs")}))
    return dict(per_window_channel=rows, per_subject_channel=summaries)


def bn_summary(model):
    result = {}
    for name, value in model.named_buffers():
        if any(s in name for s in ("running_mean", "running_var", "num_batches_tracked")):
            a = value.detach().cpu().double()
            result[name] = dict(shape=list(a.shape), min=float(a.min()), max=float(a.max()), mean=float(a.mean()))
    return result


def replay(args, m, dataset, norms, lists, binding, results, output):
    import torch
    from fatigue_training.development import SubjectGuard
    from modality_models import ModalityModel
    allowed = binding["folds"]["2"]["train"]
    tiny = lists[2]["tiny"]
    probe = lists[2]["probe"]
    if len(tiny) != 128 or len(probe) != 576 or len(allowed) != 18 or lists[2]["tiny_subjects"] != ["1", "3", "4", "6"]:
        raise ValueError("Fixed fold2 tiny/probe recipe differs")
    SubjectGuard(dataset, allowed)._check(tiny + probe)
    report = {}
    for spec in [s for s in m.specs() if s["model"] == "C_E" and s["fold"] == 2 and s["phase"] in ("tiny", "research")]:
        folder = safe(args.results) / "runs" / spec["key"]
        result = results[spec["key"]]
        for checkpoint in ("best", "last"):
            step = result["selection"]["best_step"] if checkpoint == "best" else result["executed_steps"]
            path = folder / (checkpoint + ".pt")
            if sha(path) != read(folder / "status.json")["artifacts"][checkpoint + ".pt"]:
                raise ValueError("Checkpoint SHA mismatch immediately before load")
            receipt = torch.load(path, map_location="cpu", weights_only=True)
            if receipt["binding"] != binding or receipt["spec"] != spec or receipt["step"] != step:
                raise ValueError("Checkpoint binding/spec/step mismatch")
            model = ModalityModel("C_E", 17, 2).to(args.device)
            model.load_state_dict(receipt["model"], strict=True)
            model.eval()
            key = spec["phase"] + "_" + checkpoint
            entry = dict(step=step, checkpoint_sha256=sha(path), bn=bn_summary(model), splits={})
            for split, ids in (("tiny128", tiny), ("train_probe576", probe)):
                metrics, rows = m.evaluate_fixed(model, dataset, ids, norms[2], allowed, args.device)
                m.write_predictions(output / (key + "_" + split + ".csv"), rows, spec, step)
                entry["splits"][split] = metrics
                if split == "train_probe576":
                    people = set(lists[2]["tiny_subjects"])
                    entry["splits"]["probe_4tiny_people"] = m.old.classification_metrics([r for r in rows if r["subject_id"] in people])
                    entry["splits"]["probe_14_other_training_people"] = m.old.classification_metrics([r for r in rows if r["subject_id"] not in people])
                reference = None
                if spec["phase"] == "tiny" and split == "tiny128":
                    reference = folder / ("evaluations/best_reloaded.csv" if checkpoint == "best" else f"evaluations/step{step:04d}_probe.csv")
                if spec["phase"] == "research" and split == "train_probe576":
                    reference = folder / f"evaluations/step{step:04d}_probe.csv"
                if reference is not None:
                    entry["standard_eval_reproduction"] = compare_rows(rows, reference, step)
            report[key] = entry
            write_new(output / (key + ".json"), entry)
            print("Completed read-only replay: " + key, flush=True)
            del model, receipt
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="/root/fatigue-modality-v1-code")
    p.add_argument("--results", default="/root/fatigue-modality-v1")
    p.add_argument("--output", default="/root/fatigue-eeg-readonly-v1")
    for name in ("manifest", "baseline", "preflight"):
        p.add_argument("--" + name, help="Optional; must equal original prepared binding path")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    m, inventory = bootstrap(args.code)
    dataset, norms, lists, binding, results, ledger = readonly_context(args, m, inventory)
    protected = [args.code, args.results, args.baseline, Path(args.preflight).parent,
                 dataset.root, dataset.manifest_path.parent, Path(__file__).parent,
                 "/root/fatigue-diagnostics-v1", "/root/fatigue-diagnostics-v1-code", "/root/Fatigue-check-Mamba-v2"]
    output = validate_output(args.output, protected)
    import torch
    from fatigue_training.development_preflight import configure_precision
    configure_precision()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA GPU required; no CPU fallback and no GPU success claimed")
    torch.cuda.get_device_properties(args.device)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", dict(started_at=datetime.now(timezone.utc).isoformat(),
        actual_new_fits=0, actual_optimizer_steps=0, binding=binding, source_inventory=inventory,
        scope="Training-only internal diagnostic; four tiny people and fourteen other training people are not independent generalization evidence"))
    print("Source, bindings, completed ledger and all status artifacts verified", flush=True)
    try:
        metadata_before = metadata_snapshot(args, m, dataset, binding, inventory)
        write_new(output / "input_metadata_sha256.json", metadata_before)
        signal_before = data_snapshot(dataset, lists[2]["tiny"] + lists[2]["probe"], binding["folds"]["2"]["train"])
        write_new(output / "accessed_signal_files_sha256.json", signal_before)
        print("Selected training EEG/coverage bytes verified against frozen uploaded checksums", flush=True)
        sample = sampling_report(args, m, dataset, lists, binding, results)
        write_new(output / "sampling.json", sample)
        inputs = {name: input_report(dataset, lists[2][name], norms[2], binding["folds"]["2"]["train"]) for name in ("tiny", "probe")}
        write_new(output / "inputs.json", inputs)
        if any(row["nonfinite"] for group in inputs.values() for row in group["per_window_channel"]):
            raise ValueError("Nonfinite EEG found; input summaries preserved, no evaluation attempted")
        print("Sampling coverage and guarded EEG summaries saved", flush=True)
        evaluations = replay(args, m, dataset, norms, lists, binding, results, output)
        dataset.check_unchanged()
        m.anchors(args, binding)
        # Rehash all historical completed artifact inventories and sources after inference.
        if (safe(args.results) / "writer.lock").exists() or m.source_binding() != binding["source"] or read(safe(args.results) / "ledger.json") != ledger:
            raise ValueError("Original source/ledger changed during diagnostic")
        for spec in m.specs():
            m.verify_completed(safe(args.results), spec, binding, ledger["slots"][spec["key"]]["steps"])
        if metadata_snapshot(args, m, dataset, binding, inventory) != metadata_before:
            raise ValueError("Original metadata/status/source bytes changed during diagnostic")
        if data_snapshot(dataset, lists[2]["tiny"] + lists[2]["probe"], binding["folds"]["2"]["train"]) != signal_before:
            raise ValueError("Accessed training EEG bytes changed during diagnostic")
        write_new(output / "report.json", dict(state="completed", finished_at=datetime.now(timezone.utc).isoformat(),
            actual_new_fits=0, actual_optimizer_steps=0, sampling=sample, evaluations=evaluations,
            gpu=dict(device=args.device, name=torch.cuda.get_device_name(args.device), torch=torch.__version__),
            boundary="No validation/outer predictions or signals; no normalization refit or BN recalibration"))
        text = ["EEG read-only diagnostic completed. New fits=0; optimizer steps=0.",
                "Training-only probe. Neither tiny people nor the other fourteen training people constitute independent generalization evidence."]
        for key, value in evaluations.items():
            text.append(f"{key}: step={value['step']}, tiny128 accuracy={value['splits']['tiny128']['accuracy']:.6f}, train probe576 accuracy={value['splits']['train_probe576']['accuracy']:.6f}; saved logits reproduced.")
        (output / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        export_audit(output)
    except Exception as error:
        write_new(output / "failure.json", dict(state="failed", error=repr(error), actual_new_fits=0, actual_optimizer_steps=0))
        if not (output / "eeg_readonly_audit.zip").exists():
            export_audit(output)
        raise


if __name__ == "__main__":
    main()
