import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import eeg_fulltrain as e

def row(i,s='1',block='block1',label=0,logits=(1.,0.)):
    return dict(source_index=i,subject_id=s,run_id='r',window_in_run=i,block_info=block,label=label,logits=list(logits))

class Tests(unittest.TestCase):
    def test_partitions_disjoint_cover(self):
        rr=[row(10),row(11),row(12,'7'),row(13,'7')]
        g=e.partition_rows(rr,[10],[10,12],['1'])
        self.assertEqual([len(g[k]) for k in ('tiny128','tiny4_other_windows','other14_people','probe576','nonprobe11111')],[1,1,2,2,2])
        self.assertEqual(set(id(r) for r in g['probe576']) | set(id(r) for r in g['nonprobe11111']),set(map(id,rr)))

    def test_partition_rejects_bad_universe(self):
        for rr,tt,pp in (([row(1),row(1)],[1],[1]),([row(1)],[2],[1]),([row(1)],[1],[1,2])):
            with self.assertRaises(ValueError):e.partition_rows(rr,tt,pp,['1'])

    def test_single_class_empty_and_stable_ce(self):
        m=e.metrics([row(1,logits=(1000.,-1000.)),row(2,logits=(1000.,-1000.))])
        self.assertEqual(m['recall'],[1.,None]);self.assertEqual(m['primary'],1.)
        self.assertEqual(m['window_equal_ce'],0.)
        self.assertIsNone(e.metrics([])['primary'])
        with self.assertRaises(ValueError):e.metrics([row(1,logits=(float('nan'),0))])

    def test_equal_subject_primary(self):
        rr=[row(1),row(2),row(3),row(4,'7',label=1)]
        self.assertEqual(e.metrics(rr)['accuracy'],.75)
        self.assertEqual(e.metrics(rr)['primary'],.5)
        self.assertEqual(e.metrics(rr)['recall'],[1.,0.])

    def test_all_16_blocks_and_na_csv(self):
        report=e.group_report({'full_train':[row(1),row(2,'7','rest4',1,(0.,1.))]},['1','7'])
        self.assertEqual(len(report['full_train']['per_block']),16)
        self.assertEqual(report['full_train']['per_block']['rest4']['n'],1)
        self.assertEqual(report['full_train']['per_block']['block2']['n'],0)
        self.assertIsNone(report['full_train']['per_block']['block1']['recall'][1])
        json.dumps(report,allow_nan=False)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'groups.csv';e.write_groups(p,report)
            self.assertIn('NA',p.read_text())
        with self.assertRaises(ValueError):e.group_report({'bad':[row(1,block='block17')]},['1'])

    def test_reference_reorder_missing_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ref.csv'
            def write(ids):
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['source_index']);w.writeheader();w.writerows({'source_index':i} for i in ids)
            rr=[row(2),row(1)];write([1,2])
            self.assertEqual([r['source_index'] for r in e.aligned_reference(rr,p)],[1,2])
            for ids in ([1,1],[3]):
                write(ids)
                with self.assertRaises(ValueError):e.aligned_reference(rr,p)

    def test_pinned_helper_and_old_paths(self):
        # Execute only the hash-pinned stdlib helper, never original training code.
        base=Path(e.parser().parse_args([]).previous_code)
        if not base.exists():base=Path(__file__).resolve().parents[2]/'EEG只读诊断_20261016'/'code'
        v1=e.load_previous(base)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);old=d/'old';old.mkdir()
            for output in (old,old/'new',d):
                with self.assertRaises(ValueError):v1.validate_output(output,[old])
            self.assertEqual(v1.validate_output(d/'fresh',[old]),(d/'fresh').resolve())
            with self.assertRaises(ValueError):e.load_previous(d)
        # Compare wrong corresponding metadata even when row count/logits agree.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.csv';r=row(1)
            with p.open('w',newline='') as f:
                fields=['source_index','subject_id','run_id','window_in_run','block_info','label','step','model','fold','seed','logits0','logits1']
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                w.writerow(dict(source_index=1,subject_id='7',run_id='r',window_in_run=1,block_info='block1',label=0,step=512,model='C_E',fold=2,seed=17,logits0=1,logits1=0))
            with self.assertRaises(ValueError):v1.compare_rows(e.aligned_reference([r],p),p,512)

    def test_zero_training_ast_and_cuda_fixed(self):
        tree=ast.parse(Path(e.__file__).read_text(encoding='utf-8'))
        forbidden={'backward','step','zero_grad','save','train','fit'}
        self.assertFalse([n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden])
        self.assertEqual(e.CHECKPOINTS,(('tiny','best'),('research','best'),('research','last')))
        original=e.metrics([row(1),row(2,'7',label=1)])
        e.compare_original_metrics(original,original)
        with self.assertRaises(ValueError):e.compare_original_metrics(original,dict(original,primary=.9))
        self.assertEqual(e.parser().parse_args([]).device,'cuda:0')
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):e.parser().parse_args(['--device','cpu'])

if __name__=='__main__':unittest.main()
