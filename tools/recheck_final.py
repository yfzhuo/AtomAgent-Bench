#!/usr/bin/env python3
"""Audit final submissions with the SAME current evaluator used by the live runner.
Never executes archived code or calls a model. All candidates retained, each
configuration counted once. A required unresolved score fails this audit.
"""
from pathlib import Path
import argparse, collections, hashlib, json, sys, warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agentv2.run_agent_demo import LatticeNoopEvaluator, LatticeNoopConfig, score_with_evaluator
from agentv2 import offline_noop_filter as guard

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--archive',type=Path,default=ROOT/'artifacts')
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--limit',type=int)
 args=ap.parse_args()
 if args.out.exists():ap.error('Refusing to overwrite an existing audit')
 records=json.loads((ROOT/'paper_data/instance_manifest.json').read_text())
 if args.limit:
  if args.limit<=0:ap.error('limit must be positive')
  records=records[:args.limit]
 cache={};result=[]
 for r in records:
  checks=[]
  for c in r['candidates']:
   p=args.archive/c['packaged_run']
   raw=p.read_bytes()
   assert hashlib.sha256(raw).hexdigest()==c['packaged_run_sha256']
   d=json.loads(raw)
   values=[d.get(k) for k in ['input_cif','target_cif','task_prompt','exported_cif']]
   key=hashlib.sha256(json.dumps([r['action'],values]).encode()).hexdigest()
   if key not in cache:
    e=object.__new__(LatticeNoopEvaluator);e.action_name=r['action'];e._use_exact_match_metrics=False
    e.guard_config=LatticeNoopConfig();e._guard=guard
    e.scorer_config={'name':'lattice-noop','noop_stol':.05,'noop_scale':True,'lattice_rtol':1e-6,'lattice_atol':1e-6}
    row=dict(input_cif=values[0],output_cif=values[1],action_prompt=values[2])
    with warnings.catch_warnings(record=True) as ws:
     warnings.simplefilter('always')
     try:
      score=score_with_evaluator(e,row,values[3])
      compact={k:score.get(k) for k in ['is_error','wrong_type','rmsd','max_diff','base_score','lattice_same','near_initial','guard_status']}
      compact['pass']=not score['is_error']
     except Exception as ex:compact={'pass':None,'exception':type(ex).__name__+': '+str(ex)}
     compact['warnings']=sorted({str(w.message) for w in ws})
    cache[key]=compact
   checks.append({'candidate':c['candidate_id'],'score':cache[key]})
  states={x['score']['pass'] for x in checks}
  passed=next(iter(states)) if len(states)==1 else None
  result.append({'configuration_id':r['configuration_id'],'pass':passed,'agreement':passed==r['new_pass'],'candidates':checks})
  if len(result)%100==0:print('Audited',len(result),'configurations',flush=True)
 summary={'scope':'historical final submissions; no archived code/model execution','configuration_count':len(result),'candidate_count':sum(len(x['candidates']) for x in result),'scoring_calls':len(cache),'pass':sum(x['pass'] is True for x in result),'fail':sum(x['pass'] is False for x in result),'unresolved':sum(x['pass'] is None for x in result),'all_agree':all(x['agreement'] for x in result),'results':result}
 args.out.parent.mkdir(parents=True,exist_ok=True)
 with args.out.open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
 if summary['unresolved'] or not summary['all_agree']:sys.exit(1)
if __name__=='__main__':main()
