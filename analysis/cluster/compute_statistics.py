#!/usr/bin/env python3
"""Material/input-cluster bootstrap and paired cluster sign-flip inference.
No model/scorer calls. Requires Python 3.10+ and NumPy.
"""
from pathlib import Path
import argparse, collections, hashlib, json, math
import numpy as np

MODELS=['qwen3-vl-plus','qwen3.5-plus','qwen3.7-plus','qwen3.8-max','deepseek-v4-flash-0731']
LABELS=['Qwen3-VL-Plus','Qwen3.5-Plus','Qwen3.7-Plus','Qwen3.8-Max','DeepSeek-V4-Flash']
TASKS=['delete_below_atom_action','delete_around_atom_action','insert_between_atoms_action','move_towards_atom_action','rotate_around_atom_action','RemoveMoleculeAction']
TASK_LABELS=['Delete Below','Delete Around','Insert Between','Move Towards','Rotate Around','Remove Molecule']


def exact_signflip(coefficients):
    # Swap ON/OFF for the whole input cluster, never individual task outcomes.
    # DP computes the exact two-sided conditional distribution of sum(D_cluster).
    values=[int(x) for x in coefficients if x]
    target=abs(sum(int(x) for x in coefficients))
    if not values: return {'p':1.,'nonzero_clusters':0,'observed_sum':0}
    dist={0:1}
    for value in values:
        new=collections.defaultdict(int)
        for total,count in dist.items():
            new[total+value]+=count; new[total-value]+=count
        dist=dict(new)
    numerator=sum(count for total,count in dist.items() if abs(total)>=target)
    denominator=2**len(values)
    return {'p':numerator/denominator,'nonzero_clusters':len(values),'observed_sum':sum(coefficients),
            'tail_numerator':str(numerator),'permutations_denominator':str(denominator)}


def holm(values):
    order=np.argsort(values,kind='stable');m=len(values);result=np.empty(m);last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(m-rank)*values[index]));result[index]=last
    return result.tolist()


def summarize(rows,scope,label,endpoint,rng,reps):
    selected=rows
    pairs=collections.defaultdict(dict)
    for row in selected:pairs[(row['model'],row['action'],row['sample'])][row['view']]=row
    assert all(set(p)=={'ON','OFF'} for p in pairs.values())
    cluster=collections.defaultdict(lambda:[0,0,0])
    strata={}
    both=on_only=off_only=neither=0
    for pair in pairs.values():
        on,off=pair['ON'],pair['OFF'];assert on['cluster']==off['cluster']
        key=on['cluster'];c=cluster[key];c[0]+=on[endpoint];c[1]+=off[endpoint];c[2]+=1;strata[key]=on['stratum']
        a,b=on[endpoint],off[endpoint]
        both+=int(a==1 and b==1);on_only+=int(a==1 and b==0);off_only+=int(a==0 and b==1);neither+=int(a==0 and b==0)
    arrays={}
    for stratum in sorted(set(strata.values())):
        keys=sorted(k for k in cluster if strata[k]==stratum)
        arrays[stratum]=np.array([cluster[k] for k in keys],dtype=np.int64)
        assert len(keys)==30
        assert len(set(arrays[stratum][:,2]))==1
    n=len(pairs)
    totals=np.sum(list(cluster.values()),axis=0);assert totals[2]==n
    draws=np.zeros((reps,2))
    for stratum,array in arrays.items():
        count=len(array)
        # Generate cluster multiplicities with multinomial sampling. The paired
        # outcomes and all tasks/models for each input travel together.
        mult=rng.multinomial(count,np.full(count,1/count),size=reps)
        draws += mult@array[:,:2]
    draws/=n
    values=np.column_stack([draws[:,0],draws[:,1],draws.mean(axis=1),draws[:,0]-draws[:,1]])*100
    ci=np.quantile(values,[.025,.975],axis=0,method='linear')
    point=np.array([totals[0]/n,totals[1]/n,(totals[0]+totals[1])/(2*n),(totals[0]-totals[1])/n])*100
    sign=exact_signflip([c[0]-c[1] for c in cluster.values()])
    return {'scope':scope,'label':label,'endpoint':endpoint,'paired_configurations':n,'configurations':2*n,
        'clusters':len(cluster),'clusters_by_stratum':{k:len(v) for k,v in arrays.items()},
        'ON_successes':int(totals[0]),'OFF_successes':int(totals[1]),
        'ON_pct':float(point[0]),'ON_CI95_pct':ci[:,0].tolist(),
        'OFF_pct':float(point[1]),'OFF_CI95_pct':ci[:,1].tolist(),
        'pooled_pct':float(point[2]),'pooled_CI95_pct':ci[:,2].tolist(),
        'ON_minus_OFF_pp':float(point[3]),'difference_CI95_pp':ci[:,3].tolist(),
        'paired_outcomes':{'both_success':both,'ON_only_success':on_only,'OFF_only_success':off_only,'both_failure':neither},
        'cluster_signflip':sign}


def run(records,reps,seed):
    keys=[(r['model'],r['view'],r['action'],r['sample']) for r in records]
    assert len(records)==len(set(keys))==1800
    assert sum(r['initial'] for r in records)==1155 and sum(r['final'] for r in records)==1607
    rng=np.random.Generator(np.random.PCG64(seed));results=[]
    for scope,label,rows in [
        ('original_five_tasks','Five atomic-edit tasks',[r for r in records if r['action']!='RemoveMoleculeAction']),
        ('all_six_tasks','All six tasks',records),
        ('molecule_task','Remove Molecule',[r for r in records if r['action']=='RemoveMoleculeAction'])]:
        for endpoint in ['initial','final']:results.append(summarize(rows,scope,label,endpoint,rng,reps))
    for m,label in zip(MODELS,LABELS):
        for endpoint in ['initial','final']:
            results.append(summarize([r for r in records if r['model']==m],'model_six_tasks',label,endpoint,rng,reps))
    for action,label in zip(TASKS,TASK_LABELS):
        for endpoint in ['initial','final']:
            if action=='RemoveMoleculeAction':
                # Identical sample/endpoint: reuse the same bootstrap draws,
                # avoiding Monte Carlo discrepancies between duplicate tables.
                copied=json.loads(json.dumps(next(x for x in results if x['scope']=='molecule_task' and x['endpoint']==endpoint)))
                copied['scope']='task';results.append(copied)
            else:
                results.append(summarize([r for r in records if r['action']==action],'task',label,endpoint,rng,reps))
    # Explicit multiplicity families, no selection based on observed p values.
    for scope in ['original_five_tasks','all_six_tasks','molecule_task','model_six_tasks','task']:
        subset=[x for x in results if x['scope']==scope]
        adjusted=holm([x['cluster_signflip']['p'] for x in subset])
        for result,value in zip(subset,adjusted):
            result['p_Holm_within_scope']=value;result['multiplicity_family_size']=len(subset)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).with_name('analysis_records.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('statistical_results.json'))
    parser.add_argument('--reps',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=20261126)
    args=parser.parse_args();assert args.reps>=1000
    records=json.loads(args.input.read_text())
    results=run(records,args.reps,args.seed)
    report={'method':{'bootstrap':'stratified input-cluster percentile bootstrap; complete ON/OFF pairs and all associated model/task outcomes resampled together',
        'bootstrap_replicates':args.reps,'seed':args.seed,'rng':'PCG64',
        'confidence_interval':'two-sided 95% percentile, NumPy linear quantiles',
        'test':'exact two-sided paired cluster sign-flip test for the sum of ON-OFF outcomes; dynamic programming enumerates the exact distribution',
        'test_assumption':'under the null, complete ON/OFF labels are exchangeable within each independent input cluster (conditional sign symmetry). This is not proof of randomized assignment or a causal visual effect.',
        'cluster_unit':'30 shared crystal inputs for the original five tasks; 30 separate molecule inputs for Remove Molecule. All-six analyses resample these two input groups separately, preserving the 5:1 task weight.',
        'material_interpretation':'60 unique input structures, not an assertion of 60 independent parent materials; parent genealogy for molecule inputs is not present in the adopted result records.',
        'multiplicity':'Holm correction within declared families: original-five pooled endpoints (2), all-six pooled endpoints (2), molecule endpoints (2), model-by-endpoint comparisons (10), task-by-endpoint comparisons (12).',
        'McNemar':'A pooled 900-pair McNemar test is not used because repeated tasks/models on the same input violate pair independence.',
        'scope':'Empirical uncertainty over the tested input structures, conditional on models, task templates and the recorded runs. Not repeated-model-sampling uncertainty or equivalence testing.'},
        'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'results':results}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for x in results:
        print(f"{x['scope']} {x['label']} {x['endpoint']}: ON={x['ON_pct']:.2f} OFF={x['OFF_pct']:.2f} delta={x['ON_minus_OFF_pp']:.2f} CI={x['difference_CI95_pp']} p={x['cluster_signflip']['p']:.6g} Holm={x['p_Holm_within_scope']:.6g}")

if __name__=='__main__':main()
