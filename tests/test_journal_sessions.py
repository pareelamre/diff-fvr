"""Device strata retain accuracy while refusing crossed-device cost claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.journal_protocol import protocol_hash
from experiments.summarize_journal_replication import read_complete_batch
from experiments.summarize_journal_sessions import summarize_sessions


class SessionTests(unittest.TestCase):
    def batch(self, root):
        protocol={'id':'fixture','methods':['a','b'],'tiers':[['bounded',0,0,0,0]],
                  'inits_per_tier':1,'analysis':{'reference':'a'},
                  'resources':{'required':True,'interval_s':.02,'max_gap_s':.5,
                               'success_threshold_mm':2.,'amortization_registrations':[1,10]}}
        manifest={'protocol':protocol,'sha256':protocol_hash(protocol),'code_identity':{'fixture':'immutable'}}
        (root/'protocol.json').write_text(json.dumps(manifest))
        for subject in ['s1','s2','s3']:
            for method in ['a','b']:
                row=dict(subject=subject,method=method,tier='bounded',init_idx=1,pair='A+B',
                         status='ok',initial_delta=[[0]*6],candidate_sha256='same',
                         final_mtre_mm=1. if method=='a' else 1.5,reg_time_s=10. if method=='a' else 2.,
                         setup_s=2.,warmup_s=3.,memory={'peak_allocated_MiB':4.,'peak_reserved_MiB':5.})
                data=dict(protocol=protocol,protocol_sha256=manifest['sha256'],code_identity=manifest['code_identity'],
                          source_sha256={'CT':subject},gpu='fixture-GPU',rows=[row])
                path=root/f'{subject}_{method}.json';path.write_text(json.dumps(data))
                uuid='A' if subject=='s1' or (subject=='s2' and method=='a') else 'B'
                profile=dict(schema='DFVR-RESOURCE-1',status='ok',returncode=0,gpu_required=True,
                             protocol_sha256=manifest['sha256'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             peak_process_gpu_bytes=100,peak_root_rss_bytes=200,peak_tree_rss_bytes=300,
                             samples=100,worker_wall_s=50. if method=='a' else 10.,requested_interval_s=.02,
                             max_sample_gap_s=.03,sampling_overhead_s=.1,gpu_identity={'name':'fixture-GPU','uuid':uuid})
                path.with_suffix('.resources.json').write_text(json.dumps(profile))

    def test_explicit_strata_keep_accuracy_and_exclude_cross_device_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.batch(root)
            with self.assertRaisesRegex(ValueError,'Mixed GPU devices'):
                read_complete_batch(root,subjects=['s1','s2','s3'])
            report=summarize_sessions(root,subjects=['s1','s2','s3'])
            self.assertEqual([r['n_trials'] for r in report['accuracy']], [3,3])
            self.assertEqual([s['matched_subjects'] for s in report['device_strata']], [['s1'],['s3']])
            self.assertEqual(report['cross_device_specimens'],['s2'])
            for stratum in report['device_strata']:
                comparison=stratum['comparisons'][0]
                self.assertEqual(comparison['n_joint_success'],1)
                self.assertEqual(comparison['median_warm_speed_ratio_joint_success'],5.)
                self.assertEqual(comparison['median_worker_wall_ratio'],5.)
                self.assertEqual(len(stratum['methods']),2)

    def test_corruption_missing_trials_and_mixed_models_still_fail(self):
        for defect in ['hash','missing','model','zero_cost']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);self.batch(root)
                path=root/'s1_b.resources.json';profile=json.loads(path.read_text())
                if defect=='hash':profile['result_sha256']='wrong';path.write_text(json.dumps(profile))
                elif defect=='missing':(root/'s1_b.json').unlink()
                elif defect=='model':
                    path=root/'s1_b.json';result=json.loads(path.read_text());result['gpu']='another-model';path.write_text(json.dumps(result))
                    profile['gpu_identity']['name']='another-model';profile['result_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                    path.with_suffix('.resources.json').write_text(json.dumps(profile))
                else:
                    path=root/'s1_b.json';result=json.loads(path.read_text());result['rows'][0]['reg_time_s']=0.;path.write_text(json.dumps(result))
                    profile['result_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.resources.json').write_text(json.dumps(profile))
                with self.assertRaises(ValueError):summarize_sessions(root,subjects=['s1','s2','s3'])

    def test_all_worker_memory_retains_the_cross_device_specimen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.batch(root)
            path=root/'s2_a.resources.json';p=json.loads(path.read_text());p['peak_tree_rss_bytes']=3000;path.write_text(json.dumps(p))
            report=summarize_sessions(root,subjects=['s1','s2','s3'])
            self.assertEqual(report['all_worker_memory'][0]['max_process_tree_rss_MiB'],3000/2**20)
            self.assertEqual(report['device_strata'][0]['methods'][0]['max_process_tree_rss_MiB'],300/2**20)


if __name__=='__main__':unittest.main()
