"""Apply the reviewed source payload and migrate this one observed E0 ledger.

No model forward/backward/optimizer step; no data access or budget reset.
Run from the extracted repair delivery, not from inside the target package.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

from fatigue_training.common import digest, file_hash, read_json, write_json as _write_json
from fatigue_training.development import configuration, source_hash, PROTOCOL
from fatigue_training.development_preflight import ExclusiveRun

MIGRATION = "bn1_zero_frozen_v1"
OLD_CODE_HASH = "fb01431dbea3e99dd55b87f23d924d0754d0854aa3d52fac294828edce24c900"
SLOTS = ["E0_C_EF", "E0_M0_EF"]
OLD_E0 = dict(steps=8, iterations=0, runs=2)


def reject_linked_path(path):
    """Do not follow symlinks, Windows junctions, or other reparse points."""
    path=Path(path).absolute()
    for part in [path]+list(path.parents):
        if part.is_symlink():raise ValueError("Linked path is not an approved repair target")
        if part.exists() and getattr(part.stat(follow_symlinks=False),"st_file_attributes",0)&0x400:
            raise ValueError("Reparse-point path is not an approved repair target")
    return path


def contained_path(root,name):
    root=reject_linked_path(root).resolve()
    path=reject_linked_path(root/name).resolve()
    if root not in path.parents:raise ValueError("Artifact/source path escapes its root")
    return path


def write_json(path,value):
    path=reject_linked_path(path)
    reject_linked_path(path.with_name(path.name+".tmp"))
    _write_json(path,value)


def tree_code_hash(root):
    package=Path(root)/"fatigue_training"
    return digest({p.relative_to(package).as_posix():file_hash(p)
        for p in sorted(package.rglob("*")) if p.is_file()
        and p.suffix in (".py",".json") and "__pycache__" not in p.parts})


def validate_review(review,old_config_hash,new_config_hash,new_code_hash):
    if not review.get("approved_by") or not review.get("approved_at") or not review.get("reason"):
        raise ValueError("Missing authorized scoped repair review")
    if review.get("old_code_hash")!=OLD_CODE_HASH or review.get("new_code_hash")!=new_code_hash or new_code_hash==OLD_CODE_HASH:
        raise ValueError("Repair source review mismatch")
    if review.get("old_config_hash")!=old_config_hash or review.get("new_config_hash")!=new_config_hash:
        raise ValueError("Repair configuration review mismatch")
    if review.get("slots")!=SLOTS or review.get("reviewed_unaffected_e0_models")!=[]:
        raise ValueError("All seven E0 models require current-source rechecks")


def validate_original_ledger(ledger,environment,old_config_hash):
    if ledger.get("binding")!=dict(protocol_id=PROTOCOL,config_hash=old_config_hash,environment=environment):
        raise ValueError("Original config/protocol/environment mismatch")
    if ledger.get("phases",{}).get("E0")!=OLD_E0:
        raise ValueError("Expected observed E0 8 steps/2 runs exactly")
    if set(ledger.get("phases",{}))!={"E0","D0","D1","D2"} or any(
        ledger["phases"][phase]!=dict(steps=0,iterations=0,runs=0) for phase in ("D0","D1","D2")):
        raise ValueError("Research/development fits have already started")
    if set(ledger.get("slots",{}))!=set(SLOTS):
        raise ValueError("Unexpected E0 slots")
    for key,state in zip(SLOTS,("completed","failed")):
        slot=ledger["slots"][key]
        attempts=slot.get("attempts",[])
        if slot.get("phase")!="E0" or slot.get("state")!=state or len(attempts)!=1:
            raise ValueError("Unexpected original slot state")
        attempt=attempts[0]
        if attempt.get("code_hash")!=OLD_CODE_HASH or attempt.get("state")!=state or attempt.get("steps")!=4 or attempt.get("iterations")!=0:
            raise ValueError("Unexpected original attempt provenance/budget")
    if "Repeated complete short training differs beyond fixed tolerance" not in str(ledger["slots"]["E0_M0_EF"]["attempts"][0].get("error")):
        raise ValueError("Failure is not the reviewed repeat-comparison failure")


def evidence_paths(root):
    root=Path(root)
    required=["ledger.json","C_EF/status.json","C_EF/result.json","failure.json",
        "M0_EF/attempt1/failure.json"]
    for model in ("C_EF","M0_EF"):
        reject_linked_path(root/model/"attempt1")
        required += [f"{model}/attempt1/repeat{i}.pt" for i in (0,1)]
        required += [p.relative_to(root).as_posix() for p in (root/model/"attempt1").rglob("*") if p.is_file()]
    if any(not contained_path(root,name).is_file() for name in required):
        raise ValueError("Original mandatory checkpoint/failure evidence missing")
    status=read_json(root/"C_EF/status.json")
    if status.get("state")!="completed" or status.get("code_hash")!=OLD_CODE_HASH:
        raise ValueError("Original completed C status mismatch")
    for name,h in status.get("artifacts",{}).items():
        path=contained_path(root/"C_EF",name)
        if (root/"C_EF").resolve() not in path.parents or not path.is_file() or file_hash(path)!=h:
            raise ValueError("Original completed C evidence changed")
    return sorted(set(required))


def archive_original(root,review,environment,old_config_hash,new_config_hash,new_code_hash):
    """Prepare immutable evidence before the first source or ledger replacement."""
    root=reject_linked_path(root);history=contained_path(root,"history/"+MIGRATION)
    record_path=contained_path(history,"migration.json")
    validate_review(review,old_config_hash,new_config_hash,new_code_hash)
    if record_path.exists():
        record=read_json(record_path)
        if record.get("review")!=review or record.get("environment")!=environment:
            raise ValueError("Existing migration belongs to different review/environment")
        expected=dict(migration_id=MIGRATION,old_config_hash=old_config_hash,new_config_hash=new_config_hash,
            old_code_hash=OLD_CODE_HASH,new_code_hash=new_code_hash)
        if any(record.get(k)!=v for k,v in expected.items()) or record.get("state") not in ("prepared","committed"):
            raise ValueError("Existing migration differs from frozen code/config/revision")
        for name,h in record["original_artifacts"].items():
            archived=contained_path(history/"original",name)
            if not archived.is_file() or file_hash(archived)!=h:
                raise ValueError("Archived original evidence changed")
            if "/attempt1/" in name and file_hash(contained_path(root,name))!=h:
                raise ValueError("Original attempt evidence changed")
        original=read_json(history/"original/ledger.json")
        validate_original_ledger(original,environment,old_config_hash)
        if record.get("original_phases")!=original["phases"]:raise ValueError("Archived budget provenance changed")
        return history,record,original
    original=read_json(contained_path(root,"ledger.json"))
    validate_original_ledger(original,environment,old_config_hash)
    names=evidence_paths(root)
    hashes={name:file_hash(root/name) for name in names}
    # A pre-ledger crash may leave partial identical copies; reuse only exact bytes.
    for name,h in hashes.items():
        target=contained_path(history/"original",name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_hash(target)!=h:raise ValueError("Partial archive differs")
        if not target.exists():shutil.copy2(root/name,target)
    record=dict(migration_id=MIGRATION,state="prepared",review=review,environment=environment,
        old_config_hash=old_config_hash,new_config_hash=new_config_hash,
        old_code_hash=OLD_CODE_HASH,new_code_hash=new_code_hash,
        original_artifacts=hashes,original_phases=copy.deepcopy(original["phases"]))
    write_json(record_path,record)
    return history,record,original


def migrate_ledger(root,history,record,original):
    root=Path(root);history=Path(history)
    migration={k:record[k] for k in ("migration_id","old_config_hash","new_config_hash","old_code_hash","new_code_hash","original_artifacts","original_phases")}
    migration.update(review_sha256=digest(record["review"]),history=f"history/{MIGRATION}")
    ledger_path=contained_path(root,"ledger.json")
    current=read_json(ledger_path)
    updated=copy.deepcopy(original)
    updated["binding"]["config_hash"]=record["new_config_hash"]
    for key in SLOTS:
        updated["slots"][key]["state"]="superseded"
        updated["slots"][key]["superseded_by"]=MIGRATION
    updated.setdefault("migrations",[]).append(migration)
    if current==original:
        write_json(ledger_path,updated)
    elif current!=updated:
        # Idempotent after normal E0 progress: preserve its newer reservations.
        if current.get("binding")!=updated["binding"] or migration not in current.get("migrations",[]):
            raise ValueError("Ledger changed outside this migration")
        for phase,values in original["phases"].items():
            if any(current["phases"][phase][field]<v for field,v in values.items()):
                raise ValueError("Budget decreased")
        for key in SLOTS:
            if current["slots"][key]["attempts"][0]!=original["slots"][key]["attempts"][0]:
                raise ValueError("Original attempt provenance changed")
        if any(current["phases"][phase]!=values for phase,values in original["phases"].items() if phase!="E0"):
            raise ValueError("Non-E0 fits started during migration")
        allowed={"E0_"+name for name in configuration()["models"]}|{"E0_scan_reference"}
        if not set(current["slots"])<=allowed:raise ValueError("Unexpected migrated E0 slot")
        attempts=[]
        for key,slot in current["slots"].items():
            if slot.get("phase")!="E0":raise ValueError("Unexpected migrated slot phase")
            new_attempts=slot["attempts"][1:] if key in SLOTS else slot["attempts"]
            if any(a.get("code_hash")!=record["new_code_hash"] for a in new_attempts):
                raise ValueError("Migrated attempt has unreviewed source")
            attempts+=slot["attempts"]
        e0=current["phases"]["E0"]
        if not 8<=e0["steps"]<=64 or not 2<=e0["runs"]<=12 or e0["iterations"]!=0:
            raise ValueError("Migrated E0 budget is invalid")
        if e0["steps"]!=sum(a["steps"] for a in attempts) or e0["runs"]!=len(attempts):
            raise ValueError("Migrated reservations disagree with attempt history")
    write_json(contained_path(root,"bn1_repair_review.json"),record["review"])
    record=dict(record,state="committed")
    write_json(history/"migration.json",record)
    return record


def apply_sources(project,payload,history,original_manifest,payload_manifest):
    project=Path(project);payload=Path(payload);history=Path(history)
    old={row["path"]:row["sha256"] for row in original_manifest["files"]}
    new={row["path"]:row["sha256"] for row in payload_manifest["files"]}
    if len(old)!=32 or old.keys()!=new.keys():raise ValueError("Expected the reviewed 32 delivery files")
    for name,h in new.items():
        incoming=contained_path(payload,name);current=contained_path(project,name)
        if file_hash(incoming)!=h:raise ValueError("Repair payload checksum mismatch")
        existing=file_hash(current)
        if existing not in (old[name],h):raise ValueError("Target source is neither original nor reviewed payload")
        if old[name]!=h:
            backup=contained_path(history/"source",name)
            if backup.exists():
                if file_hash(backup)!=old[name]:raise ValueError("Original source backup changed")
            elif existing==old[name]:
                backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(current,backup)
            else:raise ValueError("New source lacks original backup")
    for name,h in new.items():
        if file_hash(contained_path(project,name))!=h:
            target=contained_path(project,name);temporary=reject_linked_path(target.with_name(target.name+".bn1-repair.tmp"))
            shutil.copy2(payload/name,temporary);os.replace(temporary,target)
    if tree_code_hash(project)!=payload_manifest["new_code_hash"]:
        raise ValueError("Target package contains unreviewed code or configuration")


def verify_no_development_fits(path):
    path=reject_linked_path(path)
    if not path.exists():return
    ledger_path=contained_path(path,"ledger.json")
    if not ledger_path.exists():
        if any(path.iterdir()):raise ValueError("Nonempty development root lacks auditable ledger")
        return
    ledger=read_json(ledger_path)
    if ledger.get("slots") or any(any(values.values()) for values in ledger.get("phases",{}).values()):
        raise ValueError("Development/research fits already exist; scoped E0 migration refused")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",required=True)
    parser.add_argument("--e0-root",required=True)
    parser.add_argument("--source-root",required=True,help="Frozen official Mamba checkout")
    parser.add_argument("--development-root",required=True,help="Existing planned dev-v2 artifact root; must contain no fits")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--review",required=True)
    args=parser.parse_args(argv)
    payload=reject_linked_path(Path(__file__).absolute().parent).resolve()
    project=reject_linked_path(args.project_root).resolve();root=reject_linked_path(args.e0_root).resolve()
    if project==payload or project in payload.parents or payload in project.parents:
        raise ValueError("Extract repair separately from the target source tree")
    review=read_json(args.review)
    old_manifest=read_json(payload/"original_delivery_manifest.json")
    new_manifest=read_json(payload/"repair_payload_manifest.json")
    if old_manifest.get("old_code_hash")!=OLD_CODE_HASH or new_manifest.get("old_code_hash")!=OLD_CODE_HASH:
        raise ValueError("Original code provenance mismatch")
    if new_manifest.get("old_config_hash")!=digest(read_json(payload/"original_configuration.json")):
        raise ValueError("Original configuration provenance mismatch")
    if review.get("payload_manifest_sha256")!=file_hash(payload/"repair_payload_manifest.json"):
        raise ValueError("Repair payload lacks exact manifest review")
    if new_manifest.get("new_code_hash")!=source_hash() or new_manifest.get("new_config_hash")!=digest(configuration()):
        raise ValueError("Running repair code/config is unreviewed")
    auxiliary=new_manifest.get("auxiliary_files",[])
    if {row["path"] for row in auxiliary}!={"apply_bn1_repair.py","tests/test_bn1_repair.py","original_delivery_manifest.json","original_configuration.json"}:
        raise ValueError("Missing reviewed repair tool/test/provenance checksums")
    for row in auxiliary:
        if file_hash(contained_path(payload,row["path"]))!=row["sha256"]:raise ValueError("Repair tool/test/provenance checksum mismatch")
    from fatigue_training.development_preflight import environment_lock
    environment=environment_lock(args.device,args.source_root)
    with ExclusiveRun(contained_path(root,"writer.lock")):
        verify_no_development_fits(args.development_root)
        if not (root/"history"/MIGRATION/"migration.json").exists():
            if tree_code_hash(project)!=OLD_CODE_HASH:raise ValueError("Original target code hash mismatch")
            for row in old_manifest["files"]:
                if file_hash(project/row["path"])!=row["sha256"]:raise ValueError("Original 32-file delivery checksum mismatch")
        history,record,original=archive_original(root,review,environment,
            new_manifest["old_config_hash"],new_manifest["new_config_hash"],new_manifest["new_code_hash"])
        apply_sources(project,payload,history,old_manifest,new_manifest)
        migrate_ledger(root,history,record,original)
    print(json.dumps(dict(state="migration_completed",optimizer_steps=0,
        repair=str(root/"bn1_repair_review.json"),e0_original_steps=8,e0_original_runs=2,
        next="Run all seven E0 models and scan reference with --repair; no research fits started"),ensure_ascii=False))


if __name__=="__main__":main()
