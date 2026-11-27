#!/usr/bin/env python3
"""Offline target-site-tolerance scan of fixed archived final structures.
Only target StructureMatcher stol varies. No model or archived agent code runs.
"""
from pathlib import Path
import argparse,json,hashlib,time,warnings,collections,concurrent.futures,importlib.metadata,inspect
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser
import fixed_guard as guard
ROOT=Path(__file__).resolve().parent
TOLERANCES=[0.5,0.2,0.1,0.05]
TASKS=['delete_below_atom_action','delete_around_atom_action','insert_between_atoms_action','move_towards_atom_action','rotate_around_atom_action','RemoveMoleculeAction']
NAMES=dict(zip(TASKS,['Delete Below','Delete Around','Insert Between','Move Towards','Rotate Around','Remove Molecule']))
PARAMS={'target_matcher':{'ltol':0.2,'angle_tol':5,'primitive_cell':False,'scale':True,'attempt_supercell':False,'allow_subset':False,'supercell_size':'num_sites','ignored_species':[]},'noop_stol':0.05,'noop_scale':True,'lattice_rtol':1e-6,'lattice_atol_A':1e-6,'height_tolerance_A':0.1}
def match(stol):return StructureMatcher(stol=stol,**PARAMS['target_matcher'])
def evaluate(item):
 uid,data=item
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  structures=[]
  for key in ['input_cif','target_cif','exported_cif']:
   parser=CifParser.from_str(data[key]); ss=parser.parse_structures(primitive=False,check_occu=False)
   if not ss:raise ValueError(f'{uid}: unparseable {key}')
   structures.append(ss[0])
  initial,target,out=structures
  composition_same=len(target)==len(out) and target.composition.as_dict()==out.composition.as_dict()
  lattice=guard.check_lattice_same(initial,out,1e-6,1e-6)
  if lattice['lattice_same'] is None:raise ValueError(f'{uid}: lattice unresolved')
  # Match the production short-circuit: guards run only after reference success.
  results={};near=None
  for stol in TOLERANCES:
   target_fit=bool(match(stol).fit(target,out)) if composition_same else False
   if target_fit and near is None:
    near=bool(guard.matcher(0.05,True).fit(initial,out))
   verdict='AtomCountMismatch' if not composition_same else ('StructureMismatch' if not target_fit else ('LatticeMismatch' if not lattice['lattice_same'] else ('NearInitialStructure' if near else 'PASS')))
   results[str(stol)]={'reference_match':target_fit,'pass':verdict=='PASS','verdict':verdict}
  warn=sorted(set(str(x.message) for x in caught))
 return uid,{'input_atoms':len(initial),'reference_atoms':len(target),'output_atoms':len(out),'composition_same':composition_same,'lattice_same':lattice['lattice_same'],'lattice_max_abs_diff_A':lattice.get('max_abs_matrix_diff_A'),'near_initial':near,'by_stol':results,'parser_warnings':warn}
def summary(rows,task=None,view=None,model=None,variant=None):
 subset=[x for x in rows if (task is None or x['action']==task) and (view is None or x['view']==view) and (model is None or x['model']==model) and (variant is None or x['variant']==variant)]
 n=len(subset)
 return {'action':task,'task':NAMES.get(task,'All six tasks'),'view':view or 'ON+OFF','model':model,'variant':variant,'n':n,'by_stol':{str(t):{'passes_min':sum(x['by_stol'][str(t)]['pass_min'] for x in subset),'passes_max':sum(x['by_stol'][str(t)]['pass_max'] for x in subset),'rate_min_pct':sum(x['by_stol'][str(t)]['pass_min'] for x in subset)/n*100,'rate_max_pct':sum(x['by_stol'][str(t)]['pass_max'] for x in subset)/n*100} for t in TOLERANCES}}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=2);p.add_argument('--reuse',action='store_true');p.add_argument('--input',type=Path,default=ROOT/'evaluation_inputs.json');p.add_argument('--output-dir',type=Path,default=ROOT)
 args=p.parse_args();out=args.output_dir;out.mkdir(exist_ok=True)
 payload=json.loads(args.input.read_text());digest=hashlib.sha256(args.input.read_bytes()).hexdigest();cache=out/'triple_scores.json';start=time.monotonic()
 if args.reuse and cache.exists():
  data=json.loads(cache.read_text());assert data['input_sha256']==digest and data['params']==PARAMS;scores=data['scores']
 else:
  scores={}
  checkpoint=out/'scan_checkpoint.json'
  if checkpoint.exists():
   previous=json.loads(checkpoint.read_text());assert previous['input_sha256']==digest and previous['params']==PARAMS
   scores=previous['scores']
  items=[x for x in payload['structure_triples'].items() if x[0] not in scores]
  print(f'Resuming with {len(scores)} completed and {len(items)} remaining triples',flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
   futures=[pool.submit(evaluate,item) for item in items]
   for i,future in enumerate(concurrent.futures.as_completed(futures),1):
    uid,result=future.result();scores[uid]=result
    if i%25==0:
     print(f'Scanned {i}/{len(items)} unique final triples; {time.monotonic()-start:.1f}s',flush=True)
     (out/'scan_checkpoint.json').write_text(json.dumps({'input_sha256':digest,'params':PARAMS,'scores':scores}))
  data={'input_sha256':digest,'params':PARAMS,'tolerances':TOLERANCES,'scores':scores};cache.write_text(json.dumps(data,indent=2))
 rows=[];mismatches=[];ambiguous=[]
 for c in payload['configurations']:
  rs=[scores[x['triple_id']] for x in c['candidates']]
  by={}
  for t in TOLERANCES:
   passed=[x['by_stol'][str(t)]['pass'] for x in rs];verdicts=sorted(set(x['by_stol'][str(t)]['verdict'] for x in rs))
   by[str(t)]={'pass_min':int(all(passed)),'pass_max':int(any(passed)),'verdicts':verdicts}
   if len(set(passed))>1:ambiguous.append({'model':c['model'],'view':c['view'],'action':c['action'],'sample':c['sample'],'stol':t,'candidate_verdicts':verdicts})
  if by['0.5']['pass_min']!=by['0.5']['pass_max'] or by['0.5']['pass_min']!=int(c['pass']):mismatches.append({'model':c['model'],'view':c['view'],'action':c['action'],'sample':c['sample'],'expected_pass':c['pass'],'scan':by['0.5']})
  rows.append({k:c[k] for k in ['model','model_display','view','action','sample','pass','bucket','failure_type','variant']}|{'candidates':c['candidates'],'by_stol':by})
 if mismatches:
  (out/'baseline_mismatches.json').write_text(json.dumps(mismatches,indent=2));raise RuntimeError(f'{len(mismatches)} baseline mismatches; do not publish before resolving')
 pooled=[summary(rows,task=task) for task in TASKS]+[summary(rows)]
 views=[summary(rows,task=task,view=v) for task in TASKS+[None] for v in ['ON','OFF']]
 models=[summary(rows,model=m,view=v) for m in sorted(set(x['model'] for x in rows)) for v in ['ON','OFF']]
 variants=[summary(rows,task=TASKS[0],variant=variant,view=v) for variant in ['Including','Excluding'] for v in ['ON','OFF',None]]
 # Strictness cannot increase success in these nested matcher settings.
 nonmonotone=[]
 for c in rows:
  for k in ['pass_min','pass_max']:
   seq=[c['by_stol'][str(t)][k] for t in TOLERANCES]
   if seq!=sorted(seq,reverse=True):nonmonotone.append((c['model'],c['view'],c['action'],c['sample'],seq))
 assert not nonmonotone,nonmonotone
 warning_counts=collections.Counter(w for score in scores.values() for w in score['parser_warnings'])
 result={'input_sha256':digest,'tolerances':TOLERANCES,'parameters':PARAMS,'configuration_count':len(rows),'candidate_count':sum(len(x['candidates']) for x in rows),'unique_structure_triples':len(scores),'baseline_matches_current_manuscript':True,'candidate_disagreements':ambiguous,'pooled_tasks':pooled,'task_view':views,'model_view':models,'delete_below_variants':variants,'elapsed_seconds':time.monotonic()-start,'versions':{x:importlib.metadata.version(x) for x in ['numpy','pymatgen','scipy']},'parser_warnings':dict(warning_counts)}
 (out/'configuration_scores.json').write_text(json.dumps(rows,indent=2));(out/'tolerance_results.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'n':len(rows),'ambiguous':ambiguous,'pooled':pooled,'versions':result['versions'],'warnings':dict(warning_counts)},indent=2),flush=True)
if __name__=='__main__':main()
