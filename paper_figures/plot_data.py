"""Read the specified Markdown tables; all plot counts derive from these tables.
P(k) is the historical-round cumulative retained count, not a new online rerun.
"""
from pathlib import Path
import json,re,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/论文三张表_1500复评加分子删除原结果.md'
MODELS=['Qwen3-VL-Plus','Qwen3.5-Plus','Qwen3.7-Plus','Qwen3.8-Max','DeepSeek-V4-Flash']
TASKS=['Delete Below','Delete Around','Insert Between','Move Towards','Rotate Around','Remove Molecule']
ACTIONS=['delete_below_atom_action','delete_around_atom_action','insert_between_atoms_action','move_towards_atom_action','rotate_around_atom_action','RemoveMoleculeAction']
text=SOURCE.read_text(encoding='utf-8')
sections=re.split(r'^##\s+',text,flags=re.M)
seven_section=next(s for s in sections if s.startswith('1.'))
data={}
for line in seven_section.splitlines():
    fields=[x.strip() for x in line.strip().strip('|').split('|')]
    if len(fields)==9 and fields[0] in MODELS:
        mat=np.array([list(map(int,re.split(r'\s*/\s*',s))) for s in fields[2:8]],dtype=int).T
        assert mat.shape==(7,6) and np.all(mat.sum(axis=0)==30)
        assert mat.sum()==180 and mat.sum(axis=1).tolist()==list(map(int,fields[8].split('/')))
        data[fields[0],fields[1]]=mat
assert len(data)==10
pooled={v:sum((data[m,v] for m in MODELS),start=np.zeros((7,6),dtype=int)) for v in ['ON','OFF']}
total=pooled['ON'].sum(axis=1)+pooled['OFF'].sum(axis=1)
assert total.tolist()==[1042,304,119,55,41,22,217]
# Check the complete task/view and triple tables in the same Markdown.
for line in next(s for s in sections if s.startswith('2.')).splitlines():
    f=[x.strip() for x in line.strip().strip('|').split('|')]
    if len(f)!=9 or f[0] not in TASKS+['全部任务']:continue
    p=sum((pooled[v] for v in (['ON','OFF'] if f[1] in ['ON+OFF','合计'] else [f[1]])),start=np.zeros((7,6),dtype=int))
    c=p.sum(axis=1) if f[0]=='全部任务' else p[:,TASKS.index(f[0])]
    assert list(map(int,f[2:]))==c.tolist()
for line in next(s for s in sections if s.startswith('3.')).splitlines():
    f=[x.strip() for x in line.strip().strip('|').split('|')]
    if len(f)==9 and f[0] in MODELS:
        for i,s in enumerate(f[2:8]):
            c=data[f[0],f[1]][:,i];assert list(map(int,re.split(r'\s*/\s*',s)))==[int(c[0]),int(c[1:6].sum()),int(c[6])]
# The table has no error-type breakdown: use separately supplied archived evidence.
evidence=json.loads((ROOT/'data/failure_evidence.json').read_text())
assert len(evidence)==1800
assert len({(r['model_display'],r['view'],r['action'],r['sample']) for r in evidence})==1800
failures=collections.Counter();model_failures=collections.Counter()
for r in evidence:
    a=r['action'];new=r['new_pass']
    if a=='RemoveMoleculeAction':assert new==r['old_pass']
    if not new:
        failures[r['view'],a,r['failure_type']]+=1
        model_failures[r['model_display'],r['view'],a]+=1
for m in MODELS:
    for v in ['ON','OFF']:
        for i,a in enumerate(ACTIONS):assert model_failures[m,v,a]==data[m,v][6,i]
for v in ['ON','OFF']:
    for i,a in enumerate(ACTIONS):assert sum(n for (view,act,typ),n in failures.items() if view==v and act==a)==pooled[v][6,i]

# Display categories split the existing Delete Below column; they are NOT new tasks.
VARIANT_SOURCE=ROOT/'data/delete_below_variants.json'
variant_payload=json.loads(VARIANT_SOURCE.read_text())
variants={}
for row in variant_payload['model_seven']:
    key=(row['model'],row['view'],row['variant'])
    assert key not in variants
    variants[key]=np.array(row['seven'],dtype=int)
assert len(variants)==20
DISPLAY_TASKS=['Delete Below (Including)','Delete Below (Excluding)']+TASKS[1:]
DISPLAY_N=np.array([11,19,30,30,30,30,30],dtype=int)
DISPLAY_POOLED_N=DISPLAY_N*len(MODELS)
display_data={}
for m in MODELS:
    for v in ['ON','OFF']:
        inc,exc=variants[m,v,'Including'],variants[m,v,'Excluding']
        assert inc.sum()==11 and exc.sum()==19
        np.testing.assert_array_equal(inc+exc,data[m,v][:,0])
        display_data[m,v]=np.column_stack([inc,exc,data[m,v][:,1:]])
        np.testing.assert_array_equal(display_data[m,v].sum(axis=1),data[m,v].sum(axis=1))
        np.testing.assert_array_equal(display_data[m,v].sum(axis=0),DISPLAY_N)
display_pooled={v:sum((display_data[m,v] for m in MODELS),start=np.zeros((7,7),dtype=int)) for v in ['ON','OFF']}
MODEL_ALIAS=dict(zip(['qwen3-vl-plus','qwen3.5-plus','qwen3.7-plus','qwen3.8-max','deepseek-v4-flash-0731'],MODELS))
variant_by_key={};record_counts=collections.Counter()
for r in variant_payload['records']:
    key=(MODEL_ALIAS[r['model']],r['view'],r['sample'])
    assert key not in variant_by_key and r['old_pass']==r['new_pass']
    assert re.findall(r'\b(Including|Excluding)\b',r['task_prompt'])==[r['variant']]
    variant_by_key[key]=r['variant'];record_counts[key[0],key[1],r['variant'],r['bucket']]+=1
assert len(variant_by_key)==300
for (m,v,t),seven in variants.items():
    assert seven.tolist()==[record_counts[m,v,t,k] for k in range(7)]
display_failures=collections.Counter()
for r in evidence:
    if r['new_pass']:continue
    if r['action']==ACTIONS[0]:
        label='Delete Below ('+variant_by_key[r['model_display'],r['view'],r['sample']]+')'
    else:label=TASKS[ACTIONS.index(r['action'])]
    display_failures[r['view'],label,r['failure_type']]+=1
for v in ['ON','OFF']:
    for i,t in enumerate(DISPLAY_TASKS):
        assert sum(n for (view,label,typ),n in display_failures.items() if view==v and label==t)==display_pooled[v][6,i]

def save_numeric(name,obj):
    with (ROOT/(name+'_data.json')).open('w',encoding='utf-8') as f:
        json.dump({'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'variant_source':VARIANT_SOURCE.name,'variant_source_sha256':hashlib.sha256(VARIANT_SOURCE.read_bytes()).hexdigest(),
        'display_groups':DISPLAY_TASKS,'display_group_n_per_model_view':DISPLAY_N.tolist(),
        'scope':'original five tasks post-hoc final audit + Remove Molecule original scores; Delete Below split by prompt without rescoring',
        'pk_definition':'cumulative retained successes by historical first-pass repair bucket; not an enhanced-online-protocol rerun',
        'data':obj},f,ensure_ascii=False,indent=2,allow_nan=False)
