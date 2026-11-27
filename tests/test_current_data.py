"""Canonical evaluated data and deterministic Delete Below reference checks."""
from pathlib import Path
import json,re,sys,warnings
import numpy as np
from ase import Atoms
from pymatgen.core import Structure
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agentv2.run_agent_demo import resolve_data_dir
from atom_world.actions import DeleteBelowAtomAction

def test_canonical_dataset_and_prompt_reference_agreement():
 data_dir=Path(resolve_data_dir('delete_below_atom_action'))
 assert data_dir==ROOT/'paper_data/runtime'
 rows=json.loads((data_dir/'delete_below_atom_action.json').read_text())
 samples=[0,2,4,5,6,8,10,12,15,17,18,21,24,25,26,27,30,32,35,36,39,40,41,43,44,45,46,47,48,49]
 assert len(rows)==50
 assert [i for i,x in enumerate(rows) if x.get('input')]==samples
 variants={'Including':0,'Excluding':0}
 for i in samples:
  row=rows[i];prompt=row['action_prompt'];index=int(re.search(r'index (\d+)',prompt).group(1));variant=re.findall(r'\b(Including|Excluding)\b',prompt)[0];variants[variant]+=1
  with warnings.catch_warnings(record=True) as parsed_warnings:
   warnings.simplefilter('always')
   initial=Structure.from_str(row['input'],fmt='cif');target=Structure.from_str(row['output'],fmt='cif')
  # The production parser rounds nearly ideal fractional coordinates. Accept
  # only that audited warning; any new parser warning fails this check.
  assert all('fractional coordinates rounded to ideal values' in str(w.message) for w in parsed_warnings)
  atoms=Atoms(symbols=[str(x.specie) for x in initial],positions=initial.cart_coords,cell=initial.lattice.matrix,pbc=True)
  action=DeleteBelowAtomAction(atoms,index,variant=='Including',0.1)
  assert str(action)==prompt
  result=action.execute()
  assert len(result)==len(target)
  np.testing.assert_allclose(result.cell.array,target.lattice.matrix,atol=1e-6,rtol=1e-6)
  # Independent species/periodic-position association; no global alignment.
  available=list(range(len(target)))
  for symbol,position in zip(result.get_chemical_symbols(),result.positions):
   matched=[]
   for j in available:
    if str(target[j].specie)!=symbol:continue
    diff=target[j].frac_coords-initial.lattice.get_fractional_coords(position);diff-=np.round(diff)
    if np.linalg.norm(initial.lattice.get_cartesian_coords(diff))<1e-4:matched.append(j)
   assert matched,(i,symbol,position)
   available.remove(matched[0])
  assert not available
 assert variants=={'Including':10,'Excluding':20}

def test_all_task_defaults_resolve_to_current_runtime():
 actions=['delete_below_atom_action','delete_around_atom_action','insert_between_atoms_action','move_towards_atom_action','rotate_around_atom_action','RemoveMoleculeAction']
 for a in actions:
  p=Path(resolve_data_dir(a))/f'{a}.json';d=json.loads(p.read_text())
  assert len(d)==50 and sum(bool(x.get('input')) for x in d)==30
 assert resolve_data_dir('delete_below_atom_action','explicit-data')=='explicit-data'
