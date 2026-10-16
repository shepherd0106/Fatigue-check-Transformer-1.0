import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import backup_before_diagnostics as backup
from backup_before_diagnostics import create_backup,verify_archive,check_references,sha,write


class BackupTests(unittest.TestCase):
    def test_binary_included_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            weight=source/'best.pt';weight.write_bytes(b'checkpoint\x00\xff')
            sampling=source/'sampling.npy';sampling.write_bytes(b'numpy bytes\x00')
            inputs={'fits/best.pt':weight,'fits/sampling.npy':sampling}
            result=create_backup(inputs,root/'backups',{'code':'same'},[source])
            self.assertEqual(result['state'],'verified');self.assertFalse(result['reused'])
            verify_archive(result['archive_path'],result['input_hashes'],result['bindings'])
            with zipfile.ZipFile(result['archive_path']) as z:
                self.assertEqual(z.read('fits/best.pt'),weight.read_bytes())
            reused=create_backup(inputs,root/'backups',{'code':'same'},[source])
            self.assertTrue(reused['reused']);self.assertEqual(result['archive_sha256'],reused['archive_sha256'])

    def test_changes_produce_new_backup_and_keep_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();p=source/'state';p.write_bytes(b'old')
            first=create_backup({'state':p},root/'backups',{},[source])
            p.write_bytes(b'new')
            second=create_backup({'state':p},root/'backups',{},[source])
            self.assertNotEqual(first['archive_sha256'],second['archive_sha256'])
            self.assertTrue(Path(first['archive_path']).exists())
            with zipfile.ZipFile(first['archive_path']) as z:self.assertEqual(z.read('state'),b'old')

    def test_nested_output_refused_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'source';p.mkdir();f=p/'file';f.write_bytes(b'a')
            with self.assertRaises(ValueError):create_backup({'file':f},p/'nested',{},[p])
            self.assertFalse((p/'nested').exists())

    def test_traversal_and_changed_archive_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'file';f.write_bytes(b'a')
            with self.assertRaises(ValueError):create_backup({'../file':f},root/'b',{})
            result=create_backup({'file':f},root/'b',{})
            Path(result['archive_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):create_backup({'file':f},root/'b',{})

    def test_referenced_missing_checkpoint_or_changed_array_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'sampling.npy';p.write_bytes(b'original')
            refs={'sampling.npy':sha(p)}
            check_references(root,refs)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):check_references(root,refs)
            with self.assertRaises(ValueError):check_references(root,{'missing.pt':'0'*64})
            with self.assertRaises(ValueError):check_references(root,{'../outside':'0'*64})

    def test_final_validation_failure_does_not_advance_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'file';f.write_bytes(b'old')
            first=create_backup({'file':f},root/'b',{})
            previous=(root/'b/latest.json').read_bytes()
            f.write_bytes(b'new')
            def fail(hashes):raise ValueError('Membership changed')
            with self.assertRaises(ValueError):create_backup({'file':f},root/'b',{},final_validator=fail)
            self.assertEqual(previous,(root/'b/latest.json').read_bytes())
            receipts=[json.loads(p.read_text()) for p in (root/'b').glob('development-*.json')]
            self.assertEqual(sorted(r['state'] for r in receipts),['failed','verified'])
            with self.assertRaises(ValueError):create_backup({'file':f},root/'b',{},final_validator=fail)
            self.assertTrue(Path(first['archive_path']).exists())

    def test_index_bindings_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'file';f.write_bytes(b'old')
            result=create_backup({'file':f},root/'b',{'binding':'real'})
            with self.assertRaises(ValueError):verify_archive(result['archive_path'],result['input_hashes'],{'binding':'wrong'})

    def test_noncanonical_member_names_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'file';f.write_bytes(b'x')
            for name in ('a//b','./a','a/','.', ''):
                with self.assertRaises(ValueError):create_backup({name:f},root/'b',{})

    def test_legacy_fixed_tmp_path_cannot_redirect_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);victim=root/'evidence';victim.write_bytes(b'keep')
            # A pre-existing legacy temp file must never be opened or truncated.
            legacy=root/'latest.json.tmp';legacy.write_bytes(b'keep-temp')
            write(root/'latest.json',{'new':True})
            self.assertEqual(legacy.read_bytes(),b'keep-temp')
            self.assertEqual(victim.read_bytes(),b'keep')

    def test_latest_write_failure_keeps_verified_receipt_and_old_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'file';f.write_bytes(b'old')
            create_backup({'file':f},root/'b',{})
            old=(root/'b/latest.json').read_bytes();f.write_bytes(b'new')
            real_write=backup.write
            def fail_latest(path,value):
                if Path(path).name=='latest.json':raise OSError('simulated pointer failure')
                return real_write(path,value)
            with patch.object(backup,'write',side_effect=fail_latest),self.assertRaises(OSError):
                create_backup({'file':f},root/'b',{})
            self.assertEqual(old,(root/'b/latest.json').read_bytes())
            receipts=[json.loads(p.read_text()) for p in (root/'b').glob('development-*.json')]
            self.assertTrue(all(r['state']=='verified' for r in receipts))
            orphan=next(r for r in receipts if r.get('latest_pointer_committed') is False)
            verify_archive(root/'b'/orphan['archive'],orphan['input_hashes'],orphan['bindings'])


if __name__=='__main__':unittest.main()
