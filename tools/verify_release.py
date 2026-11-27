#!/usr/bin/env python3
"""Verify the current evaluated inputs, outputs, trajectories, images and figures.
No model API calls or execution of archived model-generated Python.
"""
from pathlib import Path
import argparse,collections,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,default=ROOT/'artifacts');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.exists():ap.error('Output exists; choose a new audit path')
 manifest=json.loads((ROOT/'paper_data/instance_manifest.json').read_text());tasks=json.loads((ROOT/'paper_data/task_instances.json').read_text());tm={t['instance_id']:t for t in tasks};scope=json.loads((ROOT/'paper_data/scope.json').read_text())
 assert len(manifest)==len({r['configuration_id'] for r in manifest})==1800 and len(tasks)==180
 bins=collections.defaultdict(lambda:[0]*7);images=set();candidates=0;failure=collections.Counter()
 for t in tasks:
  for k,h in [('initial','input_cif_sha256'),('target','target_cif_sha256')]:assert sha(ROOT/'paper_data'/t[k])==t[h]
  assert (ROOT/'paper_data'/t['prompt']).read_text()==t['task_prompt']
 for r in manifest:
  bins[r['model_display'],r['view'],r['action']][r['bucket']]+=1
  assert r['pass']==(r['bucket']<6)
  if not r['pass']:failure[r['failure_type']]+=1
  for c in r['candidates']:
   candidates+=1;p=(args.archive/c['packaged_run']).resolve();assert p.is_relative_to(args.archive.resolve())
   assert sha(p)==c['packaged_run_sha256'];d=json.loads(p.read_text());t=tm[r['instance_id']]
   assert d['sample']==r['sample'] and d['action']==r['action'] and d['use_view']==(r['view']=='ON')
   assert d['input_cif']==(ROOT/'paper_data'/t['initial']).read_text() and d['target_cif']==(ROOT/'paper_data'/t['target']).read_text() and d['task_prompt']==t['task_prompt']
   f=args.archive/c['final_cif'];assert sha(f)==c['final_cif_sha256'] and f.read_text()==d['exported_cif']
   assert json.loads((args.archive/c['model_config']).read_text())['recorded']['model_name']==d['model']
   for rec in d['trace']:
    for a in [rec,*rec.get('actions',[])]:
     if isinstance(a,dict) and a.get('image_file'):images.add((p.parent/a['image_file']).resolve(strict=True))
 assert candidates==scope['candidate_logs']==1809 and len(images)==scope['visual_observations']
 assert len(bins)==60 and all(sum(v)==30 for v in bins.values())
 totals=[sum(v[k] for v in bins.values()) for k in range(7)];assert totals==scope['total_seven']==[1155,279,89,39,27,18,193]
 assert dict(failure)=={'AtomCountMismatch':51,'StructureMismatch':137,'NearInitialStructure':5}
 from PIL import Image
 for p in images:
  with Image.open(p) as im:im.verify()
 assert json.loads((args.archive/'data/instance_manifest.json').read_text())==manifest
 checks=json.loads((args.archive/'audit/file_checksums.json').read_text())
 for rel,h in checks.items():assert sha(args.archive/rel)==h
 sys.path.insert(0,str(ROOT/'paper_figures'));import plot_data as plot
 for m in plot.MODELS:
  for v in ['ON','OFF']:
   for i,a in enumerate(plot.ACTIONS):assert plot.data[m,v][:,i].tolist()==bins[m,v,a]
 scanned=0
 for base in [ROOT/'src',ROOT/'paper_data',ROOT/'paper_figures',ROOT/'analysis',args.archive]:
  for p in base.rglob('*'):
   if any(x in p.relative_to(base).parts for x in ['runs','__pycache__']) and base!=args.archive:continue
   if p.is_file() and p.suffix in ['.py','.json','.jsonl','.md','.txt','.csv']:
    text=p.read_text(encoding='utf-8');scanned+=1
    assert not re.search(r'\bsk-[A-Za-z0-9_.-]{16,}',text),p
    assert '/Users/'+'bytedance/' not in text,p
 report={'configurations':1800,'task_instances':180,'candidate_logs':candidates,'multi_candidate_configurations':sum(len(r['candidates'])>1 for r in manifest),'images_verified':len(images),'total_seven':totals,'final_pass':1607,'final_fail':193,'failure_types':dict(failure),'all_saved_paths_and_hashes_verified':True,'figure_counts_match':True,'text_files_scanned':scanned,'archived_code_executed':False}
 args.out.parent.mkdir(parents=True,exist_ok=True)
 with args.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
