"""Metadata-only tests for the one reviewed E0 migration; no torch import."""
import copy
import tempfile
from pathlib import Path
import unittest

from fatigue_training.common import digest,file_hash,read_json,write_json
from fatigue_training.development import Ledger,configuration,COUNTS,LEGACY_COUNTS
from apply_bn1_repair import (OLD_CODE_HASH,OLD_E0,SLOTS,MIGRATION,
    validate_original_ledger,archive_original,migrate_ledger,verify_no_development_fits,
    apply_sources,tree_code_hash)

TEST_TMP=Path(__file__).parent/".tmp"
TEST_TMP.mkdir(exist_ok=True)


class MigrationTests(unittest.TestCase):
    environment={"device":"cuda:0","GPU":"fixture only; not a GPU result"}
    old_config="old-config"
    new_config="new-config"
    new_code="new-code"

    def fixture(self,root):
        review=dict(approved_by="test fixture",approved_at="2026-10-14",reason="reviewed redundant-zero bias repair",
            old_code_hash=OLD_CODE_HASH,new_code_hash=self.new_code,old_config_hash=self.old_config,
            new_config_hash=self.new_config,slots=SLOTS,reviewed_unaffected_e0_models=[])
        phases={phase:dict(steps=0,iterations=0,runs=0) for phase in ("E0","D0","D1","D2")}
        phases["E0"]=dict(OLD_E0)
        ledger=dict(binding=dict(protocol_id=configuration()["protocol_id"],config_hash=self.old_config,environment=self.environment),
            phases=phases,slots={})
        for model,state in (("C_EF","completed"),("M0_EF","failed")):
            attempt=dict(code_hash=OLD_CODE_HASH,state=state,steps=4,iterations=0,
                error=None if state=="completed" else "ValueError('Repeated complete short training differs beyond fixed tolerance')")
            ledger["slots"]["E0_"+model]=dict(phase="E0",state=state,attempts=[attempt])
            folder=root/model/"attempt1";folder.mkdir(parents=True)
            for index in (0,1):(folder/f"repeat{index}.pt").write_bytes(b"fixture-checkpoint-no-model")
            if state=="failed":write_json(folder/"failure.json",{"error":attempt["error"]})
        write_json(root/"C_EF/result.json",{"state":"completed","parameters":LEGACY_COUNTS["C_EF"]})
        names=["result.json","attempt1/repeat0.pt","attempt1/repeat1.pt"]
        write_json(root/"C_EF/status.json",dict(state="completed",code_hash=OLD_CODE_HASH,
            artifacts={name:file_hash(root/"C_EF"/name) for name in names}))
        write_json(root/"failure.json",{"state":"blocked","error":"repeat mismatch"})
        write_json(root/"ledger.json",ledger)
        return ledger,review

    def prepare(self,root,review):
        return archive_original(root,review,self.environment,self.old_config,self.new_config,self.new_code)

    def test_counts_are_actual_trainable_counts_and_policy_provenance(self):
        self.assertEqual(COUNTS,dict(C_EF=3434,M0_EF=43370,T_EF=37610,Q_EF=51626,A_EF=46538,F_EF=54794,K_EF=54786))
        self.assertTrue(all(LEGACY_COUNTS[name]-count==16 for name,count in COUNTS.items()))
        cfg=configuration();self.assertEqual(cfg["frontend_bias_policy"]["original_parameter_counts"],LEGACY_COUNTS)
        self.assertEqual((cfg["training"]["eps"],cfg["training"]["precision"]),(1e-8,"float32"))

    def test_archive_migration_and_idempotent_progress_keep_budget_history(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);original,review=self.fixture(root);before=file_hash(root/"ledger.json")
            history,record,saved=self.prepare(root,review)
            self.assertEqual(file_hash(history/"original/ledger.json"),before)
            migrate_ledger(root,history,record,saved)
            migrated=read_json(root/"ledger.json")
            self.assertEqual(migrated["phases"],original["phases"])
            self.assertEqual(migrated["binding"]["environment"],self.environment)
            for key in SLOTS:
                self.assertEqual(migrated["slots"][key]["state"],"superseded")
                self.assertEqual(migrated["slots"][key]["attempts"],original["slots"][key]["attempts"])
            ledger=Ledger(root/"ledger.json",migrated["binding"])
            with self.assertRaises(ValueError):ledger.begin("E0",SLOTS[0],self.new_code)
            self.assertEqual(ledger.begin("E0",SLOTS[0],self.new_code,review),2)
            ledger.reserve(SLOTS[0],steps=2);ledger.finish(SLOTS[0],"failed","fixture interruption")
            current_hash=file_hash(root/"ledger.json")
            history,record,saved=self.prepare(root,review)
            migrate_ledger(root,history,record,saved)
            self.assertEqual(file_hash(root/"ledger.json"),current_hash)
            self.assertEqual(read_json(root/"ledger.json")["phases"]["E0"],dict(steps=10,iterations=0,runs=3))

    def test_interrupted_after_ledger_replace_is_recoverable(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);original,review=self.fixture(root)
            history,record,saved=self.prepare(root,review)
            migrate_ledger(root,history,record,saved)
            # Simulate a crash before the migration marker's final state write.
            record["state"]="prepared";write_json(history/"migration.json",record)
            history,record,saved=self.prepare(root,review)
            migrate_ledger(root,history,record,saved)
            self.assertEqual(read_json(history/"migration.json")["state"],"committed")
            self.assertEqual(read_json(root/"ledger.json")["phases"],original["phases"])

    def test_prepared_marker_with_valid_new_attempt_progress_preserves_reservations(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);original,review=self.fixture(root)
            history,record,saved=self.prepare(root,review)
            migrate_ledger(root,history,record,saved)
            ledger=Ledger(root/"ledger.json",read_json(root/"ledger.json")["binding"])
            ledger.begin("E0",SLOTS[0],self.new_code,review);ledger.reserve(SLOTS[0],steps=1)
            current_hash=file_hash(root/"ledger.json")
            record["state"]="prepared";write_json(history/"migration.json",record)
            history,record,saved=self.prepare(root,review)
            migrate_ledger(root,history,record,saved)
            self.assertEqual(file_hash(root/"ledger.json"),current_hash)
            self.assertEqual(read_json(history/"migration.json")["state"],"committed")

    def test_reject_wrong_budget_environment_slots_attempt_source_or_research(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);original,review=self.fixture(root)
            variations=[]
            bad=copy.deepcopy(original);bad["phases"]["E0"]["steps"]=7;variations.append(bad)
            bad=copy.deepcopy(original);bad["binding"]["environment"]={};variations.append(bad)
            bad=copy.deepcopy(original);bad["slots"][SLOTS[0]]["state"]="failed";variations.append(bad)
            bad=copy.deepcopy(original);bad["slots"][SLOTS[1]]["attempts"][0]["code_hash"]="unreviewed";variations.append(bad)
            bad=copy.deepcopy(original);bad["phases"]["D1"]["runs"]=1;variations.append(bad)
            for bad in variations:
                with self.subTest(bad=digest(bad)),self.assertRaises(ValueError):validate_original_ledger(bad,self.environment,self.old_config)

    def test_reject_original_artifact_mutation_and_new_review(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);original,review=self.fixture(root)
            history,record,saved=self.prepare(root,review)
            with self.assertRaises(ValueError):self.prepare(root,dict(review,approved_by="someone else"))
            (root/"M0_EF/attempt1/repeat0.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):self.prepare(root,review)

    def test_reject_corrupted_migration_code_config_budget_and_path(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);original,review=self.fixture(root)
            history,record,saved=self.prepare(root,review)
            for key,value in (("migration_id","other"),("new_config_hash","other"),
                ("new_code_hash","other"),("original_phases",{})):
                bad=dict(record);bad[key]=value;write_json(history/"migration.json",bad)
                with self.subTest(key=key),self.assertRaises(ValueError):self.prepare(root,review)
            bad=dict(record);bad["original_artifacts"]={"../outside": "fake"}
            write_json(history/"migration.json",bad)
            with self.assertRaises(ValueError):self.prepare(root,review)

    def test_no_development_fits_guard(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);verify_no_development_fits(root)
            (root/"unknown.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):verify_no_development_fits(root)

    def test_source_apply_backup_and_idempotence_without_unreviewed_replacement(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root=Path(tmp);project=root/"project";payload=root/"payload";history=root/"history"
            old=[];new=[]
            for index in range(32):
                name=f"fatigue_training/source{index}.py"
                for parent in (project,payload):(parent/name).parent.mkdir(parents=True,exist_ok=True)
                (project/name).write_bytes(b"original")
                (payload/name).write_bytes(b"reviewed" if index==0 else b"original")
                old.append(dict(path=name,sha256=file_hash(project/name)))
                new.append(dict(path=name,sha256=file_hash(payload/name)))
            reviewed=dict(files=new,new_code_hash=tree_code_hash(payload))
            (project/old[-1]["path"]).write_bytes(b"unexpected")
            with self.assertRaises(ValueError):apply_sources(project,payload,history,dict(files=old),reviewed)
            self.assertEqual((project/old[0]["path"]).read_bytes(),b"original")
            (project/old[-1]["path"]).write_bytes(b"original")
            apply_sources(project,payload,history,dict(files=old),reviewed)
            self.assertEqual((history/"source"/old[0]["path"]).read_bytes(),b"original")
            apply_sources(project,payload,history,dict(files=old),reviewed)
            self.assertEqual(tree_code_hash(project),reviewed["new_code_hash"])
            write_json(root/"ledger.json",dict(slots={},phases={"D1":dict(steps=0,runs=0,iterations=0)}))
            verify_no_development_fits(root)
            write_json(root/"ledger.json",dict(slots={"one":{}},phases={}))
            with self.assertRaises(ValueError):verify_no_development_fits(root)


if __name__=="__main__":unittest.main()
