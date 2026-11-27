#!/usr/bin/env python3
"""Plot the real tolerance scan in the existing paper/SI visual idiom."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
TASKS=['Delete Below','Delete Around','Insert Between','Move Towards','Rotate Around','Remove Molecule']
COLORS=['#1f6f43','#5ba17b','#b25928','#6087a5','#865993','#c39e42']
MARKERS=['o','s','^','D','v','P']
def main():
 data=json.loads((ROOT/'tolerance_results.json').read_text());t=data['tolerances'];x=np.arange(len(t));stats=data['pooled_tasks']
 assert not data['candidate_disagreements'],'Cannot plot a single-value curve when candidates disagree'
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,(ax,delta)=plt.subplots(1,2,figsize=(9.4,3.5),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
 plotted={}
 for name,col,marker in zip(TASKS,COLORS,MARKERS):
  row=next(r for r in stats if r['task']==name);y=[row['by_stol'][str(v)]['rate_min_pct'] for v in t]
  line,=ax.plot(x,y,color=col,marker=marker,lw=1.5,ms=4,label=name)
  assert np.array_equal(line.get_ydata(),y)
  plotted[name]=y
 total=stats[-1];y=[total['by_stol'][str(v)]['rate_min_pct'] for v in t]
 ax.plot(x,y,'--',color='#333333',lw=1.2,label='All six tasks')
 ax.set_xticks(x,[str(v) for v in t]);ax.set_xlabel('Reference site tolerance, stol');ax.set_ylabel('Final-submission pass rate (%)');ax.set_ylim(0,103);ax.set_yticks(range(0,101,20));ax.grid(color='#dddddd',lw=.5);ax.set_axisbelow(True);ax.set_title('(a) Acceptance of fixed final submissions',fontsize=9,loc='left')
 # A second panel exposes small changes hidden by the full-rate scale.
 for name,col,marker in zip(TASKS,COLORS,MARKERS):
  yy=np.array(plotted[name]);delta.plot(x,yy[0]-yy,color=col,marker=marker,lw=1.5,ms=4)
 delta.set_xticks(x,[str(v) for v in t]);delta.set_xlabel('Reference site tolerance, stol');delta.set_ylabel('Drop from stol = 0.5 (percentage points)');delta.set_ylim(bottom=0);delta.grid(color='#dddddd',lw=.5);delta.set_axisbelow(True);delta.set_title('(b) Sensitivity to stricter matching',fontsize=9,loc='left')
 handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
 fig.savefig(ROOT/'fig_s3_tolerance.pdf',bbox_inches='tight');fig.savefig(ROOT/'fig_s3_tolerance.png',dpi=300,bbox_inches='tight');plt.close(fig)
 (ROOT/'figure_data.json').write_text(json.dumps({'stol':t,'rates_pct':plotted,'all_tasks_pct':y,'series_values_verified':True},indent=2))
 print('Reference-tolerance curve and drop panel generated; exact data checked.')
if __name__=='__main__':main()
