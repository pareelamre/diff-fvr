"""A full batch, not the presence of a few JSON files, authorizes a summary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.journal_protocol import protocol_hash
from experiments.summarize_journal_replication import read_complete_batch


class SummaryTests(unittest.TestCase):
    def batch(self, root):
        protocol={'methods':['a','b'],'tiers':[['bounded',0,0,0,0]],'inits_per_tier':1}
        identity={'git_commit':'fixture','git_status':'','code_sha256':{'fixture':'abc'}}
        manifest={'protocol':protocol,'sha256':protocol_hash(protocol),'code_identity':identity}
        (root/'protocol.json').write_text(json.dumps(manifest))
        for s in ['subject01','subject02']:
            for m in ['a','b']:
                row=dict(subject=s,method=m,tier='bounded',init_idx=1,pair='A+B',status='ok',
                         initial_delta=[[0]*6],candidate_sha256='same',final_mtre_mm=1,
                         reg_time_s=1,setup_s=2,warmup_s=3,
                         memory={'peak_allocated_MiB':4,'peak_reserved_MiB':5})
                result=dict(protocol=protocol,protocol_sha256=manifest['sha256'],
                            code_identity=identity,source_sha256={'CT':'abc'},gpu='fixture-GPU',rows=[row])
                (root/f'{s}_{m}.json').write_text(json.dumps(result))

    def test_requires_entire_expected_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.batch(root)
            _,rows,_=read_complete_batch(root,subjects=['subject01','subject02'])
            self.assertEqual(len(rows),4)
            for m in ['a','b']: (root/f'subject02_{m}.json').unlink()
            with self.assertRaisesRegex(ValueError,'Missing'):
                read_complete_batch(root,subjects=['subject01','subject02'])

    def test_rejects_duplicate_and_mismatched_initial_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.batch(root)
            p=root/'subject01_b.json'; data=json.loads(p.read_text())
            data['rows'][0]['initial_delta']=[[1]*6]; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'initial'):
                read_complete_batch(root,subjects=['subject01','subject02'])
            data['rows']*=2; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'roster'):
                read_complete_batch(root,subjects=['subject01','subject02'])


if __name__=='__main__': unittest.main()
