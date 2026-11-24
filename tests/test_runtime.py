"""Offline tests exercise the real runner, tools, guards, and OpenAI HTTP protocol.
The HTTP endpoint is deterministic and local, NOT a real model/provider test.
"""
from pathlib import Path
import base64, io, json, os, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
import numpy as np
from PIL import Image
from pymatgen.core import Structure, Lattice
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agentv2.run_agent_demo import build_evaluator, score_with_evaluator, compute_stats
from agentv2.structure_env import StructureEnvironment
from agentv2.vision_agent import VisionModelingAgent

@pytest.fixture
def data_dir(tmp_path):
 initial=Structure(Lattice.cubic(10),['Na','Cl'],[[.1,.1,.1],[.6,.6,.6]])
 target=initial.copy();target.translate_sites([0],[2.,0,0],frac_coords=False)
 d=tmp_path/'data';d.mkdir()
 (d/'move_towards_atom_action.json').write_text(json.dumps([{'input':initial.to(fmt='cif'),'output':target.to(fmt='cif'),'action_prompt':'Move atom 0 by 2 angstrom along Cartesian x.'}]))
 return d

def test_scorer_original_and_guards(data_dir,tmp_path,monkeypatch):
 monkeypatch.chdir(tmp_path)
 o=build_evaluator('move_towards_atom_action',str(data_dir))
 e=build_evaluator('move_towards_atom_action',str(data_dir),scorer='lattice-noop')
 row=e.data.iloc[0]
 assert not score_with_evaluator(o,row,row['input_cif'])['is_error']
 noop=score_with_evaluator(e,row,row['input_cif'])
 assert noop['wrong_type']=='NearInitialStructure' and noop['lattice_same'] is True
 correct=score_with_evaluator(e,row,row['output_cif'])
 assert not correct['is_error'] and correct['near_initial'] is False
 scaled=Structure.from_str(row['output_cif'],fmt='cif');scaled.scale_lattice(scaled.volume*1.5**3)
 assert not score_with_evaluator(o,row,scaled.to(fmt='cif'))['is_error']
 assert score_with_evaluator(e,row,scaled.to(fmt='cif'))['wrong_type']=='LatticeMismatch'
 bad=Structure.from_str(row['output_cif'],fmt='cif');bad.remove_sites([0])
 assert score_with_evaluator(e,row,bad.to(fmt='cif'))['wrong_type']=='AtomCountMismatch'
 assert correct['scorer_config']==noop['scorer_config']
 with pytest.raises(Exception):e.guard_config.noop_stol=.2

def test_tools_state_and_render(data_dir):
 row=json.loads((data_dir/'move_towards_atom_action.json').read_text())[0]
 e=StructureEnvironment(row['input'])
 assert e.run_python('structure.translate_sites([0], [2,0,0], frac_coords=False)')['ok']
 before=e.export_cif()
 assert not e.run_python("structure.remove_sites([0]);raise ValueError('test')")['ok']
 assert e.export_cif()==before
 obs=e.view(show_indices=True,highlight=[0])
 assert obs['ok'] and obs['image_base64']
 with Image.open(io.BytesIO(base64.b64decode(obs['image_base64']))) as im:assert max(im.size)<=600;im.verify()
 assert e.view(target='target')['ok'] is False
 assert e.code_calls==2 and e.view_calls==2

@pytest.mark.parametrize('view,binary,repeat',[(False,False,1),(True,True,2)])
def test_real_cli_over_mock_http(data_dir,tmp_path,view,binary,repeat):
 requests=[]
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def do_POST(self):
   body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   requests.append(body)
   tools={t['function']['name'] for t in body['tools']}
   assert ('view' in tools)==view
   assert body['enable_thinking'] is False
   msgs=body['messages']
   code_seen=any(m.get('role')=='tool' and '[run_python]' in str(m.get('content')) for m in msgs)
   submit_seen=any(m.get('role')=='tool' and 'result=FAIL' in str(m.get('content')) for m in msgs)
   view_seen=any(m.get('role')=='tool' and '[view]' in str(m.get('content')) for m in msgs)
   if code_seen: name,args='submit',{}
   elif not submit_seen:name,args='submit',{}
   elif view and not view_seen:name,args='view',{'show_indices':True,'highlight':[0]}
   else:name,args='run_python',{'code':'structure.translate_sites([0], [2,0,0], frac_coords=False); print(len(structure))'}
   payload={'id':'mock-completion','object':'chat.completion','created':0,'model':body['model'],'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[{'id':'mock-call-'+str(len(requests)),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}
   raw=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 env=dict(os.environ,DASHSCOPE_API_KEY='mock-test-key',PYTHONPATH=str(ROOT/'src'))
 cmd=[sys.executable,'-m','src.agentv2.run_agent_demo','--model','mock-local-model','--action','move_towards_atom_action','--data-dir',str(data_dir),'--samples','0','--max-steps','20','--max-repairs','5','--scorer','lattice-noop','--repeat',str(repeat),'--group','test_'+tmp_path.name,'--base-url',f'http://127.0.0.1:{server.server_port}/v1']
 if not view:cmd.append('--no-view')
 if binary:cmd+=['--feedback-mode','binary']
 try:
  run=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
 finally:server.shutdown();thread.join();server.server_close()
 assert run.returncode==0,run.stdout+run.stderr
 batch=next(line.split(': ',1)[1] for line in run.stdout.splitlines() if line.startswith('[Log] batch dir:'))
 d=json.loads((Path(batch)/'batch.json').read_text())
 assert d['n_run']==repeat and d['n_pass']==repeat
 assert d['scorer_config']['name']=='lattice-noop'
 for rep in range(repeat):
  r=json.loads((Path(batch)/f's0_r{rep}'/'run.json').read_text())
  assert r['repair_attempts']==1 and r['repair_history']==['NearInitialStructure',None]
  assert r['guard_checks']['lattice_same'] is True and r['guard_checks']['near_initial'] is False
  assert r['usage']['code_calls']==1
  assert r['usage']['view_calls']==(1 if view else 0)
  assert r['trace'][1]['attempt']==0
 assert (Path(batch)/'plot_stacked.png').exists() and (Path(batch)/'plot_cumulative.png').exists()
 for req in requests:
  for m in req['messages']:
   if m.get('role')=='tool' and 'result=FAIL' in str(m.get('content')):
    assert ('NearInitialStructure' in m['content'])==(not binary)
   if binary and m.get('role')=='user' and 'Your submission was FAILED' in str(m.get('content')):
    assert 'error_type' not in m['content'] and 'NearInitialStructure' not in m['content']

def test_cli_key_required_and_help(tmp_path):
 env=dict(os.environ);env.pop('DASHSCOPE_API_KEY',None);env.pop('OPENAI_API_KEY',None)
 run=subprocess.run([sys.executable,'-m','src.agentv2.run_agent_demo','--help'],cwd=ROOT,env=env,capture_output=True,text=True)
 assert run.returncode==0 and '--scorer' in run.stdout and '--feedback-mode' in run.stdout
 run=subprocess.run([sys.executable,'-m','src.agentv2.run_agent_demo'],cwd=ROOT,env=env,capture_output=True,text=True)
 assert run.returncode!=0 and 'DASHSCOPE_API_KEY' in run.stdout

def test_budget_exhaustion_and_reset(data_dir,tmp_path,monkeypatch):
 from openai.types.chat import ChatCompletion
 class Fake:
  def create(self,**kw):
   return ChatCompletion.model_validate({'id':'mock','object':'chat.completion','created':0,'model':'mock','choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[{'id':'mock','type':'function','function':{'name':'run_python','arguments':json.dumps({'code':'print(len(structure))'})}}]}}]})
 from types import SimpleNamespace
 client=SimpleNamespace(chat=SimpleNamespace(completions=Fake()))
 agent=VisionModelingAgent(client,'mock',max_steps=2,use_view=False,verbose=False)
 row=json.loads((data_dir/'move_towards_atom_action.json').read_text())[0]
 for _ in range(2):
  result=agent.run(row['input'],row['action_prompt'],max_repairs=1)
  assert agent.usage['steps']==4 and agent.usage['repair_attempts']==1 and agent.usage['code_calls']==4
  assert agent.repair_history==['StepBudgetExhausted','StepBudgetExhausted']
  assert len(result)==2

def test_statistics_separate_repeat_and_repair():
 s=[{'sample':0,'repeat':0,'passed':True,'repair_attempts':1},{'sample':0,'repeat':1,'passed':False},{'sample':2,'repeat':0,'passed':True,'repair_attempts':0}]
 d=compute_stats(s,2,5)
 assert d['n_runs']==3 and d['n_samples']==2 and d['cumulative_pass_at_k']==[1/3,2/3,2/3,2/3,2/3,2/3]
 assert d['pass_at_k_by_rep']==[1.,1.]
