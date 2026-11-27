from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import warnings
import zipfile
from functools import lru_cache
sys.dont_write_bytecode = True
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser
from scipy.optimize import linear_sum_assignment
CORE_ACTIONS = {'delete_below_atom_action', 'delete_around_atom_action', 'insert_between_atoms_action', 'move_towards_atom_action', 'rotate_around_atom_action'}
CORE_MODELS = {'qwen3-vl-plus', 'qwen3.5-plus', 'qwen3.7-plus', 'qwen3.8-max', 'deepseek-v4-flash', 'deepseek-v4-flash-0731'}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_model(name):
    name = str(name or '').lower()
    return 'deepseek-v4-flash-0731' if name == 'deepseek-v4-flash' else name

def view_label(d):
    if isinstance(d.get('use_view'), bool):
        return 'ON' if d['use_view'] else 'OFF'
    return 'UNKNOWN'

def iter_logs(source, manifest=None):
    allowed = None
    if manifest:
        obj = json.loads(Path(manifest).read_text(encoding='utf-8'))
        allowed = obj.get('members') if isinstance(obj, dict) else obj
        if not isinstance(allowed, list) or not all((isinstance(x, str) for x in allowed)):
            raise ValueError('Manifest must be a JSON list of member paths or {members: [...]}')
        if len(allowed) != len(set(allowed)):
            raise ValueError('Manifest repeats a member path')
        allowed = set(allowed)
    found = set()
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            infos = [i for i in z.infolist() if i.filename.endswith('/run.json') and (not i.filename.startswith('__MACOSX/'))]
            if len(infos) != len({i.filename for i in infos}):
                raise ValueError('ZIP has duplicate member names; resolve before audit')
            for i in sorted(infos, key=lambda i: i.filename):
                if allowed is not None and i.filename not in allowed:
                    continue
                found.add(i.filename)
                yield (i.filename, z.read(i))
    elif source.is_dir():
        for p in sorted(source.rglob('run.json')):
            name = p.relative_to(source).as_posix()
            if '__MACOSX' in p.parts or (allowed is not None and name not in allowed):
                continue
            found.add(name)
            yield (name, p.read_bytes())
    else:
        raise ValueError('Source must be a ZIP archive or directory')
    if allowed is not None and allowed - found:
        raise ValueError(f'Manifest members not found: {sorted(allowed - found)}')

@lru_cache(maxsize=512)
def parse_cif(text):
    messages = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Missing or empty CIF')
            parser = CifParser.from_str(text)
            structures = parser.parse_structures(primitive=False, check_occu=False)
            if not structures:
                raise ValueError('No structure in CIF')
            result, error = (structures[0], None)
        except Exception as e:
            result, error = (None, f'{type(e).__name__}: {e}')
        messages = sorted({str(w.message) for w in captured})
    return (result, messages, error)

def matcher(stol, scale=True):
    return StructureMatcher(ltol=0.2, stol=stol, angle_tol=5, primitive_cell=False, scale=scale, attempt_supercell=False, allow_subset=False)

@lru_cache(maxsize=8192)
def match_texts(a, b, stol, scale):
    sa, _, ea = parse_cif(a)
    sb, _, eb = parse_cif(b)
    if ea or eb:
        return (None, ea or eb)
    try:
        return (bool(matcher(stol, scale).fit(sa, sb)), None)
    except Exception as e:
        return (None, f'{type(e).__name__}: {e}')

def check_lattice_same(initial, output, rtol=1e-06, atol=1e-06):
    if any((not math.isfinite(x) or x < 0 for x in (rtol, atol))):
        raise ValueError('Lattice rtol/atol must be finite and non-negative')
    if initial is None or output is None:
        return {'status': 'UNRESOLVED', 'lattice_same': None, 'error': 'Input or output CIF is missing/unparseable'}
    try:
        a = np.asarray(initial.lattice.matrix, dtype=float)
        b = np.asarray(output.lattice.matrix, dtype=float)
        if a.shape != (3, 3) or b.shape != (3, 3):
            raise ValueError('Lattice matrix must be 3x3')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Lattice matrix contains NaN or infinity')
        volume_a, volume_b = (abs(float(np.linalg.det(a))), abs(float(np.linalg.det(b))))
        if not (math.isfinite(volume_a) and math.isfinite(volume_b) and (volume_a > 0) and (volume_b > 0)):
            raise ValueError('Lattice volume must be finite and positive')
        delta = np.abs(b - a)
        allowed = atol + rtol * np.abs(a)
        same = bool(np.all(delta <= allowed))
        return {'status': 'MATCH' if same else 'LatticeMismatch', 'lattice_same': same, 'error': None, 'rtol': rtol, 'atol_A': atol, 'input_matrix_A': a.tolist(), 'output_matrix_A': b.tolist(), 'max_abs_matrix_diff_A': float(np.max(delta)), 'max_length_diff_A': float(np.max(np.abs(np.asarray(output.lattice.abc) - np.asarray(initial.lattice.abc)))), 'max_angle_diff_deg': float(np.max(np.abs(np.asarray(output.lattice.angles) - np.asarray(initial.lattice.angles)))), 'input_volume_A3': volume_a, 'output_volume_A3': volume_b, 'volume_ratio': volume_b / volume_a}
    except Exception as e:
        return {'status': 'UNRESOLVED', 'lattice_same': None, 'error': f'{type(e).__name__}: {e}'}

def filter_decision(old_pass, lattice_same, near_initial):
    if old_pass is False:
        return (False, 'OLD_FAIL')
    if old_pass is not True:
        return (None, 'UNRESOLVED')
    if lattice_same is False:
        return (False, 'REVOKED_LATTICE_MISMATCH')
    if lattice_same is not True or near_initial is None:
        return (None, 'UNRESOLVED')
    if near_initial is True:
        return (False, 'REVOKED_NEAR_INITIAL')
    return (True, 'RETAINED_PASS')

def fixed_frame_metrics(a, b):
    if a is None or b is None or len(a) != len(b) or (not len(a)):
        return {'status': 'unavailable_atom_count_or_parse'}
    if not np.allclose(a.lattice.matrix, b.lattice.matrix, rtol=1e-06, atol=1e-06):
        return {'status': 'unavailable_lattice_differs'}
    species_a = [s.species_string for s in a]
    species_b = [s.species_string for s in b]
    if collections.Counter(species_a) != collections.Counter(species_b):
        return {'status': 'unavailable_composition_differs'}
    ds = []
    for species in sorted(set(species_a)):
        ia = [i for i, s in enumerate(species_a) if s == species]
        ib = [i for i, s in enumerate(species_b) if s == species]
        cost = a.lattice.get_all_distances(a.frac_coords[ia], b.frac_coords[ib])
        rows, cols = linear_sum_assignment(cost ** 2)
        ds.extend(cost[rows, cols].tolist())
    out = {'status': 'ok', 'rmsd_A': float(np.sqrt(np.mean(np.square(ds)))), 'max_A': float(max(ds))}
    if species_a == species_b:
        diagonal = np.diag(a.lattice.get_all_distances(a.frac_coords, b.frac_coords))
        out['indexed_max_A'] = float(np.max(diagonal))
    return out

def audit_record(name, raw, stols, primary, scale, lattice_rtol=1e-06, lattice_atol=1e-06):
    r = {'member': name, 'run_sha256': hashlib.sha256(raw).hexdigest()}
    try:
        d = json.loads(raw)
        if not isinstance(d, dict):
            raise ValueError('run.json must contain an object')
    except Exception as e:
        return {**r, 'record_status': 'invalid_json', 'record_error': f'{type(e).__name__}: {e}', 'old_pass': None, 'lattice_same': None, 'lattice_check': {'status': 'UNRESOLVED', 'lattice_same': None, 'error': 'Invalid run.json'}, 'by_stol': {str(t): {'status': 'UNRESOLVED', 'new_pass': None, 'lattice_same': None, 'new_wrong_type': None} for t in stols}}
    score = d.get('score')
    flag = score.get('is_error') if isinstance(score, dict) else None
    old_pass = not flag if isinstance(flag, bool) else None
    model, action = (d.get('model'), d.get('action'))
    r.update(model=model, model_canonical=canonical_model(model), action=action, sample=d.get('sample'), repeat=d.get('repeat'), view=view_label(d), use_view_raw=d.get('use_view'), allow_imaging_raw=d.get('allow_imaging'), repair_attempts=d.get('repair_attempts'), old_pass=old_pass, old_wrong_type=score.get('wrong_type') if isinstance(score, dict) else None, old_rmsd=score.get('rmsd') if isinstance(score, dict) else None, old_max_diff=score.get('max_diff') if isinstance(score, dict) else None, paper_like_configuration=action in CORE_ACTIONS and str(model or '').lower() in CORE_MODELS and (view_label(d) != 'UNKNOWN'), record_status='ok', by_stol={})
    parsed = {}
    for key in ['input_cif', 'exported_cif', 'target_cif']:
        text = d.get(key)
        s, ws, err = parse_cif(text)
        parsed[key] = s
        r[key + '_sha256'] = hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
        r[key + '_warnings'] = ws
        r[key + '_error'] = err
        r[key + '_atoms'] = len(s) if s is not None else None
    inp, output, target = (parsed[k] for k in ['input_cif', 'exported_cif', 'target_cif'])
    lattice_check = check_lattice_same(inp, output, lattice_rtol, lattice_atol)
    lattice_same = lattice_check['lattice_same']
    r['lattice_same'] = lattice_same
    r['lattice_check'] = lattice_check
    for t in stols:
        near, err = match_texts(d.get('input_cif'), d.get('exported_cif'), t, scale)
        risk, target_err = match_texts(d.get('input_cif'), d.get('target_cif'), t, scale)
        new, status = filter_decision(old_pass, lattice_same, near)
        if status == 'REVOKED_LATTICE_MISMATCH':
            new_wrong_type = 'LatticeMismatch'
        elif status == 'REVOKED_NEAR_INITIAL':
            new_wrong_type = 'NearInitialStructure'
        elif status == 'OLD_FAIL':
            new_wrong_type = r['old_wrong_type']
        else:
            new_wrong_type = None
        r['by_stol'][str(t)] = {'near_initial': near, 'lattice_same': lattice_same, 'new_pass': new, 'status': status, 'new_wrong_type': new_wrong_type, 'match_error': err, 'lattice_check_error': lattice_check.get('error'), 'target_near_initial': risk, 'target_check_error': target_err}
    primary_result = r['by_stol'][str(primary)]
    if old_pass is True:
        r['output_vs_input_fixed_frame'] = fixed_frame_metrics(inp, output)
        r['output_vs_target_fixed_frame'] = fixed_frame_metrics(target, output)
    if old_pass is True and primary_result['near_initial'] is True and (inp is not None) and (output is not None):
        try:
            dist = matcher(primary, scale).get_rms_dist(inp, output)
            r['primary_matcher_rmsd_normalized'] = float(dist[0]) if dist is not None else None
            r['primary_matcher_max_normalized'] = float(dist[1]) if dist is not None else None
        except Exception as e:
            r['primary_distance_error'] = str(e)
    return r

def summarize(records, stols):
    summaries = []
    for t in stols:
        for scope in ['all_archive_records', 'paper_like_all_versions']:
            rs = records if scope == 'all_archive_records' else [r for r in records if r.get('paper_like_configuration')]
            by_group = collections.defaultdict(list)
            by_group['ALL', 'ALL', 'ALL'] = rs
            for r in rs:
                by_group[r.get('model_canonical', 'UNKNOWN'), r.get('view', 'UNKNOWN'), r.get('action', 'UNKNOWN')].append(r)
            for (m, v, a), group in sorted(by_group.items(), key=lambda x: str(x[0])):
                old = sum((r['old_pass'] is True for r in group))
                revoked_noop = sum((r['by_stol'][str(t)]['status'] == 'REVOKED_NEAR_INITIAL' for r in group))
                revoked_lattice = sum((r['by_stol'][str(t)]['status'] == 'REVOKED_LATTICE_MISMATCH' for r in group))
                revoked = revoked_noop + revoked_lattice
                lattice_mismatch = sum((r.get('lattice_same') is False for r in group))
                lattice_unresolved = sum((r.get('lattice_same') is None for r in group))
                both = sum((r['old_pass'] is True and r.get('lattice_same') is False and (r['by_stol'][str(t)].get('near_initial') is True) for r in group))
                unresolved = sum((r['by_stol'][str(t)]['status'] == 'UNRESOLVED' for r in group))
                kept = sum((r['by_stol'][str(t)]['new_pass'] is True for r in group))
                risk = sum((r['by_stol'][str(t)].get('target_near_initial') is True for r in group))
                conflict = sum((r['by_stol'][str(t)]['status'] == 'REVOKED_NEAR_INITIAL' and r['by_stol'][str(t)].get('target_near_initial') is True for r in group))
                summaries.append(dict(scope=scope, stol=t, model=m, view=v, action=a, records=len(group), old_pass=old, revoked=revoked, retained_pass=kept, unresolved=unresolved, revoked_near_initial=revoked_noop, revoked_lattice_mismatch=revoked_lattice, lattice_mismatch_records=lattice_mismatch, lattice_unresolved_records=lattice_unresolved, old_pass_both_lattice_and_near_initial=both, target_near_initial_records=risk, revoked_target_risk_records=conflict, old_rate=old / len(group) if group else None, new_rate=kept / len(group) if group and (not unresolved) else None))
    return summaries

def duplicate_records(records):
    groups = collections.defaultdict(list)
    for r in records:
        if r.get('record_status') != 'ok':
            continue
        key = (r['model_canonical'], r['view'], r['action'], r['sample'], r['repeat'])
        groups[key].append(r)
    return [{**r, 'duplicate_group_size': len(g)} for g in groups.values() if len(g) > 1 for r in g]

def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path, help='Saved logs ZIP or directory (read only)')
    ap.add_argument('--out-dir', required=True, type=Path, help='A NEW directory; existing paths are rejected')
    ap.add_argument('--stols', default='0.01,0.05,0.1,0.2')
    ap.add_argument('--primary-stol', type=float, default=0.05, help='Exploratory, not a validated scientific threshold')
    ap.add_argument('--scale', choices=['true', 'false'], default='true', help='Scale flag for near-initial matcher only; independent lattice check is always enforced')
    ap.add_argument('--lattice-rtol', type=float, default=1e-06, help='Relative tolerance for each lattice matrix entry (dimensionless)')
    ap.add_argument('--lattice-atol', type=float, default=1e-06, help='Absolute tolerance for each lattice matrix entry, in angstrom')
    ap.add_argument('--manifest', help='Optional confirmed JSON member list; never auto-deduplicate')
    ap.add_argument('--limit', type=int, help='Smoke-test first N records only, clearly marked in metadata')
    args = ap.parse_args()
    stols = sorted(set((float(x) for x in args.stols.split(','))))
    if args.primary_stol not in stols or any((not math.isfinite(x) or x <= 0 or x >= 0.5 for x in stols)):
        ap.error('stols must be finite, >0, <0.5 and contain primary-stol')
    if any((not math.isfinite(x) or x < 0 for x in (args.lattice_rtol, args.lattice_atol))):
        ap.error('lattice-rtol and lattice-atol must be finite and non-negative')
    if args.limit is not None and args.limit <= 0:
        ap.error('limit must be positive')
    if args.out_dir.exists():
        ap.error('out-dir already exists: refusing to overwrite any files')
    if not args.out_dir.parent.is_dir():
        ap.error('out-dir parent does not exist')
    source = args.source.resolve(strict=True)
    if source.is_dir() and args.out_dir.resolve().is_relative_to(source):
        ap.error('out-dir cannot be inside the source directory')
    scale = args.scale == 'true'
    records = []
    for i, (name, raw) in enumerate(iter_logs(source, args.manifest), 1):
        if args.limit is not None and i > args.limit:
            break
        records.append(audit_record(name, raw, stols, args.primary_stol, scale, lattice_rtol=args.lattice_rtol, lattice_atol=args.lattice_atol))
        if i % 100 == 0:
            print(f'Checked {i} saved runs', flush=True)
    if not records:
        ap.error('No run.json records found')
    summaries = summarize(records, stols)
    dupes = duplicate_records(records)
    meta = dict(source=str(source), source_sha256=sha256_file(source) if source.is_file() else None, script_sha256=sha256_file(Path(__file__)), records=len(records), stols=stols, primary_stol=args.primary_stol, scale=scale, ltol=0.2, angle_tol=5, primitive_cell=False, attempt_supercell=False, allow_subset=False, filter_version='lattice_and_noop_v1', lattice_rtol=args.lattice_rtol, lattice_atol_A=args.lattice_atol, lattice_rule='Every input/output matrix entry: abs(output-input) <= atol_A + rtol*abs(input); no scaling or basis alignment; independent of atom count', parser='CifParser defaults; primitive=False; check_occu=False', scope='Every archived log, including duplicates; not the confirmed paper manifest', limit=args.limit, manifest=args.manifest, versions={p: importlib.metadata.version(p) for p in ['pymatgen', 'numpy', 'scipy']}, formula='new_pass = old_pass AND lattice_same AND NOT near_initial; unresolved remains null', warnings=['Post-hoc saved final submissions only; cannot recover counterfactual repair trajectories', 'Near-initial match is equivalence under StructureMatcher, not proof of indexed immobility', 'Lattice preservation is checked in the parsed input basis/frame; equivalent basis changes are not accepted', 'Default lattice tolerances are numerical settings, not validated scientific thresholds', 'No action-specific correctness or physical-stability validation added', 'Target/input matches are recorded as risks; no automatic exclusion or denominator change'])
    args.out_dir.mkdir()
    for name, obj in [('metadata.json', meta), ('summary.json', summaries), ('duplicates.json', dupes)]:
        write_json(args.out_dir / name, obj)
    with open(args.out_dir / 'records.jsonl', 'x', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    with open(args.out_dir / 'summary.csv', 'x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    selected = [r for r in records if r['by_stol'][str(args.primary_stol)]['status'] in ('REVOKED_NEAR_INITIAL', 'REVOKED_LATTICE_MISMATCH')]
    write_json(args.out_dir / 'revoked_primary.json', selected)
    near_selected = [r for r in selected if r['by_stol'][str(args.primary_stol)]['status'] == 'REVOKED_NEAR_INITIAL']
    lattice_selected = [r for r in selected if r['by_stol'][str(args.primary_stol)]['status'] == 'REVOKED_LATTICE_MISMATCH']
    write_json(args.out_dir / 'revoked_near_initial_primary.json', near_selected)
    write_json(args.out_dir / 'revoked_lattice_primary.json', lattice_selected)
    print(json.dumps({'out_dir': str(args.out_dir), 'records': len(records), 'primary_revoked': len(selected), 'primary_revoked_near_initial': len(near_selected), 'primary_revoked_lattice_mismatch': len(lattice_selected), 'duplicate_records': len(dupes), 'summary': [s for s in summaries if s['model'] == 'ALL']}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
