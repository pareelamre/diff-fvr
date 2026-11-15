"""Protocol and specimen-level inference for DFVR-J34 (no renderer dependency)."""
import hashlib
import json
import math
from collections import defaultdict
from itertools import product

import numpy as np


PROTOCOL = {
    'id': 'DFVR-J34-r1', 'height': 256, 'candidates': 64, 'survivors': 4,
    'steps': {'single': 40, 'spatial': 50, 'pou': 50, 'screen': 35, 'polish': 20},
    'nano_samples': 500, 'diffdrr_microbatch': 2, 'seed': 34000,
    'tiers': [['bounded', .10, 8., 10., 12.], ['wide', .20, 18., 16., 22.]],
    'inits_per_tier': 2, 'pair_rule': 'maximum optical separation (legacy development pair)',
    'split': 'development replication; no independent test anatomy',
    'calibration': 'fixed synthetic canonical poses; no evaluation pose or target pixels',
    'canonical_poses': [[0., 0., 0., 0., 850., 0.], [0., 0., .6, 0., 850., 0.]],
    'methods': ['DiffDRR-SS-FP32', 'DiffDRR-C64-FP32', 'nanodrr-C64-FP32',
                'PoU-FVR-G4x4x4-R64', 'PoU-FVR+nanodrr-K4-FP32', 'PoU-FVR+DiffDRR-K4-FP32'],
    'parameters': {'standalone_grid': [4,4,4], 'screen_grid': [2,2,2], 'block_res':64,
        'padding':2, 'screen_precision':'bfloat16', 'spatial_precision':'float32',
        'backend':'eager torch / auto Triton for nanodrr', 'calibration_min_ncc':.975,
        'lr_rotation':.015, 'lr_translation':1.5, 'polish_lr_rotation':.008,
        'polish_lr_translation':.8, 'adam_betas':[.8,.99], 'gradient_clamps':[.25,2.5],
        'cosine_floor':.1, 'seed_formula':'seed+subject_index*100+tier_index*20+init_index; candidates add 3000',
        'tier_units':['name','initial_rotation_rad','initial_translation_mm','candidate_rotation_deg','candidate_translation_mm']},
    'analysis': {'primary':'per-specimen paired effects and failure counts; estimation first',
        'reference':'DiffDRR-C64-FP32', 'comparisons':'all five remaining arms vs reference, pooled tiers',
        'test':'two-sided exact sign flip of mean specimen effects; descriptive Holm family',
        'alpha':.05, 'minimum_two_sided_p_n6':.03125,
        'warning':'Six specimens cannot achieve Holm-adjusted p<=.05 across five comparisons; no equivalence inference',
        'interval':'exploratory percentile cluster bootstrap; small-n coverage not guaranteed',
        'failures':'report per specimen; refuse paired error inference if any expected row fails or is absent'},
}


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.] * len(pvalues)
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * pvalues[i]))
        adjusted[i] = previous
    return adjusted


def paired_specimen_effect(rows, method, reference, *, expected_keys, bootstrap=10000, seed=34000):
    """Equal-weight specimen medians, exploratory bootstrap and exact sign flip.

    Negative effect favors method. Reject incomplete, duplicate or failed pairs;
    failures require an explicit failure-rate analysis, not silent deletion.
    """
    selected = defaultdict(dict)
    for row in rows:
        if row['method'] not in (method, reference):
            continue
        key = (row['subject'], row['tier'], row['init_idx'], row['pair'])
        if row['method'] in selected[key]:
            raise ValueError('Duplicate paired trial key')
        if row.get('status', 'ok') != 'ok' or row.get('final_mtre_mm') is None:
            raise ValueError('Failed/nonfinite trial requires explicit failure analysis')
        value = float(row['final_mtre_mm'])
        if not math.isfinite(value):
            raise ValueError('Failed/nonfinite trial requires explicit failure analysis')
        selected[key][row['method']] = value
    if set(selected) != set(expected_keys):
        raise ValueError('Incomplete expected trial roster')
    clustered = defaultdict(list)
    for key, pair in selected.items():
        if set(pair) != {method, reference}:
            raise ValueError('Incomplete paired trial key')
        clustered[key[0]].append(pair[method] - pair[reference])
    effects = np.asarray([np.median(clustered[s]) for s in sorted(clustered)])
    if len(effects) < 2:
        raise ValueError('At least two independent specimens required')
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(effects, (bootstrap, len(effects))), axis=1)
    if len(effects) > 16:
        raise ValueError('Exact sign-flip enumeration limited to 16 specimens')
    null = np.abs(np.asarray(list(product((-1,1),repeat=len(effects)))) @ effects / len(effects))
    p = float(np.mean(null >= abs(float(np.mean(effects))) - 1e-12))
    return {'n_specimens': len(effects), 'n_trials': len(selected),
            'median_paired_effect_mm': float(np.median(effects)),
            'exploratory_cluster_percentile95_mm': np.quantile(draws, [.025, .975]).tolist(),
            'p_unadjusted': p, 'specimen_effects_mm': effects.tolist(),
            'test':'exact sign flip of mean specimen effects; assumes symmetry/exchangeability',
            'n_favors_method':int(np.sum(effects<0))}
