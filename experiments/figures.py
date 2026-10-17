import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'figures';F.mkdir(exist_ok=True)
rows=json.loads((ROOT/'results/study.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
colors={'nominal':'#28588c','scenario':'#d48526','box':'#1c806b'}
labels={'nominal':'Nominal fit','scenario':'8-scenario fit','box':'Box fit'}

fig,ax=plt.subplots(figsize=(7.1,2.2));ax.axis('off')
items=[(.02,'Bounded material law',r'$\ell_e\leq c_e\leq h_e$'+'\nLearned or prescribed'),
       (.35,'Fixed sparse graph',r'$\widehat L(c)=\sum_e s_ec_e b_eb_e^T$'+'\nNonnegative multipliers'),
       (.68,'Uniform spectral test',r'$U_\epsilon\preceq0,\quad V_\epsilon\preceq0$'+'\nTwo extremal settings')]
for x,title,body in items:
 ax.add_patch(FancyBboxPatch((x,.25),.29,.57,boxstyle='round,pad=.015',facecolor='#eff5f7',edgecolor='#8ca6b2',transform=ax.transAxes))
 ax.text(x+.145,.68,title,ha='center',weight='bold',transform=ax.transAxes)
 ax.text(x+.145,.46,body,ha='center',va='center',linespacing=1.7,transform=ax.transAxes,fontsize=8)
for x in [.315,.645]:ax.annotate('',xy=(x+.025,.53),xytext=(x,.53),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#405564'})
ax.text(.5,.06,'Preserving mass, positivity, and dissipation does not by itself certify spectral accuracy.',ha='center',transform=ax.transAxes,fontsize=8)
fig.savefig(F/'contract.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.1,2.75),sharex=True)
for method in labels:
 for ax,key in zip(axes,['certified','learned_test_max']):
  ys=[np.mean([r[key] for r in rows if r['method']==method and r['kappa']==k]) for k in [1,2,4,8]]
  ax.plot([1,2,4,8],ys,'o-',color=colors[method],label=labels[method])
for ax in axes:
 ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],['1','2','4','8']);ax.set_xlabel('Conductance contrast $\kappa$');ax.grid(alpha=.2)
axes[0].set_ylabel('Mean maximum relative energy error');axes[0].set_title('(a) Exact box worst case')
axes[1].set_title('(b) Maximum of 128 learned test settings');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(F/'contrast.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),sharex=True,sharey=True)
for ax,method in zip(axes,['nominal','box']):
 subset=[r for r in rows if r['method']==method and r['kappa']>1]
 ax.scatter([r['certified'] for r in subset],[r['random_box_max'] for r in subset],s=13,color=colors[method],alpha=.6,label='128 independent box samples')
 ax.scatter([r['certified'] for r in subset],[r['learned_test_max'] for r in subset],s=12,marker='x',color='#7b4965',alpha=.6,label='128 learned test settings')
 ax.plot([0,2],[0,2],':',c='gray');ax.set_xlim(.15,1.9);ax.set_ylim(.15,1.9)
 ax.set_title(labels[method]);ax.set_xlabel('Exact box worst case');ax.grid(alpha=.15)
axes[0].set_ylabel('Sampled maximum error');axes[1].legend(fontsize=7,loc='upper left')
fig.tight_layout();fig.savefig(F/'sampling.pdf');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(7.1,4.3),sharex=True,sharey=True)
for ax,name in zip(axes.flat,['grid','geometric','clustered','complete']):
 for method in labels:
  data=[[r['certified'] for r in rows if r['graph']==name and r['method']==method and r['kappa']==4 and r['fraction']==f] for f in [.5,.75,.9]]
  ax.plot([.5,.75,.9],np.mean(data,axis=1),'o-',color=colors[method],label=labels[method],markersize=3)
  ax.fill_between([.5,.75,.9],np.min(data,axis=1),np.max(data,axis=1),color=colors[method],alpha=.10)
 ax.set_title(name.capitalize());ax.grid(alpha=.2)
for ax in axes[1]:ax.set_xlabel('Requested retained edge fraction')
for ax in axes[:,0]:ax.set_ylabel('Exact box worst-case error')
axes[0,1].legend(fontsize=7)
fig.tight_layout();fig.savefig(F/'budget.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.1,2.7))
k=np.linspace(1,12,200)
axes[0].plot(k,np.maximum(.5,1-1.5/(1+2*k)),color=colors['nominal'],label='Nominal weights')
axes[0].plot(k,k/(k+1),color=colors['box'],label='Box-optimal weights')
axes[0].set_xlabel('Triangle conductance contrast $\kappa$');axes[0].set_ylabel('Exact worst-case error');axes[0].legend(fontsize=7)
sc=json.loads((ROOT/'results/scaling.json').read_text())
for kind,c in [('geometric',colors['nominal']),('complete',colors['box'])]:
 for frac,style in [(.5,':'),(.75,'-'),(.9,'--')]:
  y=[np.mean([r['epsilon'] for r in sc if r['kind']==kind and r['n']==n and r['fraction']==frac and r['kappa']==4]) for n in [32,64,128]]
  axes[1].plot([32,64,128],y,style,color=c,marker='o',markersize=3,label=f'{kind}, {frac:g}')
axes[1].set_xlabel('Vertices (scalar refits only)');axes[1].set_ylabel('Exact box worst-case error');axes[1].legend(fontsize=6,ncol=2)
for ax in axes:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(F/'limits.pdf');plt.close(fig)

with (ROOT/'paper/aggregate.tex').open('w') as f:
 f.write('\\begin{tabular}{lrrrrrr}\\toprule\n')
 f.write('&\\multicolumn{3}{c}{Box worst case}&\\multicolumn{3}{c}{Learned test maximum}\\\\\n')
 f.write('$\\kappa$ & Nominal & Scenario & Box & Nominal & Scenario & Box\\\\\\midrule\n')
 for k in [1,2,4,8]:
  values=[np.mean([r[key] for r in rows if r['method']==m and r['kappa']==k]) for key in ['certified','learned_test_max'] for m in labels]
  f.write(str(k)+' & '+' & '.join(f'{v:.3f}' for v in values)+'\\\\\n')
 f.write('\\bottomrule\\end{tabular}\n')
with (ROOT/'paper/families.tex').open('w') as f:
 f.write('\\begin{tabular}{lrrlll}\\toprule\nGraph & Edges & Retained & Nominal & Scenario & Box\\\\\\midrule\n')
 for name in ['grid','geometric','clustered','complete']:
  selected=[r for r in rows if r['graph']==name and r['fraction']==.75 and r['kappa']==4]
  vals=[]
  for m in labels:
   a=[r['certified'] for r in selected if r['method']==m]
   vals.append(f'${np.mean(a):.3f} \\pm {np.std(a,ddof=1):.3f}$')
  f.write(f'{name.capitalize()} & {selected[0]["m"]} & {selected[0]["retained"]} & '+' & '.join(vals)+'\\\\\n')
 f.write('\\bottomrule\\end{tabular}\n')

metrics={}
metrics['max_objective_recheck_gap']=max(abs(r['objective']-r['certified']) for r in rows if r['method']=='box')
metrics['statuses']={s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}
metrics['dominance_cells']=sum(a['certified']<b['certified']-1e-6 for a,b in zip([r for r in rows if r['method']=='box' and r['kappa']>1],[r for r in rows if r['method']=='nominal' and r['kappa']>1]))
metrics['median_solve_seconds']={m:float(np.median([r['solve_seconds'] for r in rows if r['method']==m])) for m in labels}
(ROOT/'results/analysis.json').write_text(json.dumps(metrics,indent=2))
