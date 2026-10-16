import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import eeg_fit576 as e

class Tests(unittest.TestCase):
    def test_durable_failed_reservation_no_refund_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.json';b=e.Budget(path,{'protocol':'test'});b.begin('D03');b.reserve('D03');b.finish('D03','failed','simulated step failure')
            saved=e.read(path);self.assertEqual((saved['runs'],saved['reserved_steps'],saved['confirmed_steps']),(1,1,0))
            with self.assertRaises(ValueError):b.begin('D03')
            with self.assertRaises(ValueError):e.Budget(path,{})
            with self.assertRaises(ValueError):b.reserve('D03')

    def test_cap_and_confirmed_endpoint(self):
        with tempfile.TemporaryDirectory() as d,patch.object(e,'STEPS',2):
            b=e.Budget(Path(d)/'ledger.json',{})
            for arm in e.ARMS:
                b.begin(arm)
                for _ in range(2):b.reserve(arm);b.confirm(arm)
                with self.assertRaises(ValueError):b.reserve(arm)
                b.finish(arm,'completed')
            self.assertEqual(b.state['confirmed_steps'],4)
            with self.assertRaises(ValueError):b.begin('third')

    def test_incomplete_cannot_complete_and_no_double_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            b=e.Budget(Path(d)/'ledger.json',{});b.begin('D03')
            with self.assertRaises(ValueError):b.confirm('D03')
            b.reserve('D03')
            with self.assertRaises(ValueError):b.reserve('D03')
            original=e.sha(b.path)
            with self.assertRaises(ValueError):b.finish('D03','completed')
            self.assertEqual(b.state['slots']['D03']['state'],'running');self.assertEqual(e.sha(b.path),original)
            b.confirm('D03');self.assertEqual(b.state['slots']['D03']['confirmed_steps'],1)

    def test_prefix_exposure_excludes_unexecuted_suffix(self):
        stream=[[1]*64,[2]*64]+[[1]*64 for _ in range(2302)]
        windows={1:SimpleNamespace(source_index=1,subject_id='1',run_id='a',block_info='block1',label_2class=0),2:SimpleNamespace(source_index=2,subject_id='3',run_id='b',block_info='rest4',label_2class=1)}
        # Production fixed denominator576 is intentional; exact counts test executed prefix.
        r=e.sampling_report(stream,1,[1,2],SimpleNamespace(windows=windows))
        self.assertEqual(r['draws'],64);self.assertEqual([r['draws'] for r in r['per_window']],[64,0])
        with self.assertRaises(ValueError):e.sampling_report(stream,-1,[1,2],SimpleNamespace(windows=windows))

    def test_selection_exact_metadata_and_tamper(self):
        candidate=e.read(Path(e.__file__).with_name('selection.json'))
        self.assertEqual(e.digest(candidate['selection']),e.SELECTION_SHA)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'windows.csv';p.write_text('metadata fixture',encoding='utf-8')
            windows={r['index']:SimpleNamespace(source_index=r['source_index'],subject_id=r['subject_id'],label_2class=r['label'],run_id=r['run_id'],block_info=r['block_info'],window_in_run=r['window_in_run']) for r in candidate['selection']}
            ds=SimpleNamespace(index_path=p,windows=windows,indices=lambda allowed:list(windows))
            fixture=copy.deepcopy(candidate);fixture['source_windows_sha256']=e.sha(p)
            self.assertEqual(len(e.verify_selection(fixture,ds,candidate['subjects'])),576)
            windows[candidate['selection'][0]['index']].label_2class=1-candidate['selection'][0]['label']
            with self.assertRaises(ValueError):e.verify_selection(fixture,ds,candidate['subjects'])
            fixture['selection'][0]['source_index']+=1
            with self.assertRaises(ValueError):e.verify_selection(fixture,ds,candidate['subjects'])

    def test_paths_and_source_pin(self):
        local=Path(__file__).resolve().parents[2]/'EEG完整训练评估_20261016/code/eeg_fulltrain.py'
        if not local.exists():local=Path('/root/fatigue-eeg-fulltrain-v1-code/eeg_fulltrain.py')
        f=e.checked_import(local,'test_full_helper',e.FULL_HELPER_SHA)
        readonly=Path('/root/fatigue-eeg-readonly-v1-code')
        if not readonly.exists():readonly=Path(__file__).resolve().parents[2]/'EEG只读诊断_20261016/code'
        v1=f.load_previous(readonly)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);old=d/'old';old.mkdir()
            for root in (old,old/'nested',d):
                with self.assertRaises(ValueError):v1.validate_output(root,[old])
            p=d/'changed.py';p.write_text('raise Exception("must not execute")')
            with self.assertRaises(ValueError):e.checked_import(p,'bad',e.FULL_HELPER_SHA)

    def test_export_keeps_arm_paths_and_excludes_binaries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for arm in e.ARMS:
                p=root/'runs'/arm;p.mkdir(parents=True);(p/'endpoint.json').write_text('{}');(p/'last.pt').write_bytes(b'weight')
            (root/'sampling_indices.npy').write_bytes(b'stream')
            with contextlib.redirect_stdout(io.StringIO()):e.export(None,root,True)
            import zipfile
            with zipfile.ZipFile(root/'eeg_fit576_partial_audit.zip') as z:
                self.assertIn('runs/D03/endpoint.json',z.namelist());self.assertIn('runs/D00/endpoint.json',z.namelist())
                self.assertFalse(any(n.endswith(('.pt','.npy')) for n in z.namelist()))

    def test_progress_is_short_read_only(self):
        import check_fit
        with tempfile.TemporaryDirectory() as d:
            b=e.Budget(Path(d)/'ledger.json',{});before=e.sha(b.path)
            with contextlib.redirect_stdout(io.StringIO()) as buf:check_fit.main(['--output',d])
            self.assertLess(len(buf.getvalue().splitlines()),6);self.assertEqual(before,e.sha(b.path))
            e.write(Path(d)/'report.json',{'state':'completed'});e.write(Path(d)/'failure.json',{'error':'export failed'})
            with contextlib.redirect_stdout(io.StringIO()) as buf:check_fit.main(['--output',d])
            self.assertIn('finalization failed',buf.getvalue());self.assertNotIn('upload',buf.getvalue())
            (Path(d)/'failure.json').unlink()
            with contextlib.redirect_stdout(io.StringIO()) as buf:check_fit.main(['--output',d])
            self.assertNotIn('upload',buf.getvalue());self.assertIn('not confirmed',buf.getvalue())
            (Path(d)/'eeg_fit576_audit.zip').write_bytes(b'fixture')
            with contextlib.redirect_stdout(io.StringIO()) as buf:check_fit.main(['--output',d])
            self.assertIn('upload',buf.getvalue())

    def test_training_output_fixed_protocol_path(self):
        self.assertEqual(e.parser().parse_args([]).output,'/root/fatigue-eeg-fit576-v1')
        for output in ('/root/new-budget','/root/fatigue-eeg-fit576-v1/../new-budget'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):e.parser().parse_args(['--output',output])

    def test_actual_dropout_only_p_changes_optional_torch(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable locally: no model/GPU validation claimed')
        root=Path('/root/fatigue-modality-v1-code')
        if not root.exists():self.skipTest('Deployed original source unavailable: server runtime test required')
        full=e.checked_import(Path('/root/fatigue-eeg-fulltrain-v1-code/eeg_fulltrain.py'),'runtime_full',e.FULL_HELPER_SHA)
        v1=full.load_previous('/root/fatigue-eeg-readonly-v1-code');v1.bootstrap(root)
        from modality_models import ModalityModel
        model=ModalityModel('C_E',17,2);before=e.state_hashes(model);cfg=e.set_dropout(model,0.)
        self.assertEqual(before,e.state_hashes(model));self.assertEqual(set(cfg),set(e.SITES))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),1802)

if __name__=='__main__':unittest.main()
