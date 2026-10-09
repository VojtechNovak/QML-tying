"""Complete-block, paired summaries, exploratory intervals, plots and LaTeX tables."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect_results import write_csv


FAMILIES=('reupload','qcnn','autoencoder','recurrent','graph','su2')
STRUCTURED={'reupload':('depth','random_depth'),'qcnn':('spatial','random_spatial'),
            'autoencoder':('inverse_depth','inverse_random_depth'),'recurrent':('temporal','random_temporal'),
            'graph':('node_edge','random_node_edge'),'su2':('su2','random_su2')}


def read_csv(path):
    if not path.exists():return []
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def numeric(row,key):
    value=row.get(key,'')
    if value in ('',None,'None'):return None
    try:value=float(value)
    except (ValueError,TypeError):return None
    return value if np.isfinite(value) else None


def interval(values,rng):
    v=np.asarray(values,float);n=len(v)
    if not n:return dict(mean=None,sd=None,low=None,high=None,n_dataset_seeds=0)
    b=np.mean(rng.choice(v,(3000,n),replace=True),axis=1) if n>1 else v
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)) if n>1 else None,
                low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n_dataset_seeds=n,
                interval_role='exploratory dataset-seed bootstrap; no multiple-testing significance claim')


def primary_metric(row):return 'test_mean_infidelity' if row['family']=='autoencoder' else 'test_balanced_accuracy'


def summarize(rows,rng):
    groups=defaultdict(list)
    keys=('case','family','tier','dataset','n','layers','encoding','map_seed','tie','optimizer','d','p','budget')
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    output=[]
    for group,rs in sorted(groups.items()):
        base=dict(zip(keys,group));metrics=[primary_metric(rs[0]),'best_train_loss','selected_val_loss','wall_seconds','nfev']
        if rs[0]['family']!='autoencoder':metrics+=['test_log_loss','test_roc_auc','test_f1_macro','test_brier']
        for metric in metrics:
            if not any(numeric(row,metric) is not None for row in rs):continue
            seeds=defaultdict(list)
            for row in rs:
                value=numeric(row,metric)
                if value is not None:seeds[row['split_seed']].append(value)
            output.append(base|dict(metric=metric,n_runs=len(rs))|interval([np.mean(v) for v in seeds.values()],rng))
    return output


def contrasts(rows,rng,axis):
    # Numerical repetitions are paired before averaging; no seed is treated as
    # an independent test observation within the same generated dataset.
    groups=defaultdict(dict)
    invariant=('case','family','tier','dataset','n','layers','encoding','map_seed','budget')
    other='optimizer' if axis=='tie' else 'tie'
    for row in rows:
        key=tuple(row[k] for k in invariant)+(row[other],row['split_seed'],row['run_seed'])
        groups[key][row[axis]]=row
    paired=defaultdict(lambda:defaultdict(list));metadata={}
    for key,by in groups.items():
        ref='untied' if axis=='tie' else 'cobyla';pairs=[(v,ref,'versus_untied' if axis=='tie' else 'versus_COBYLA') for v in by if v!=ref and ref in by]
        family=key[invariant.index('family')]
        if axis=='tie':
            structured,random=STRUCTURED[family]
            if structured in by and random in by:pairs.append((structured,random,'structured_versus_matched_random'))
        for target,reference,label in pairs:
            metric=primary_metric(by[target]);v=numeric(by[target],metric);r=numeric(by[reference],metric)
            if v is None or r is None:continue
            group=key[:len(invariant)]+(key[len(invariant)],target,reference,label,metric)
            paired[group][key[-2]].append(v-r)
            metadata[group]=dict(zip(invariant,key[:len(invariant)]))|dict(**{other:key[len(invariant)]},target=target,reference=reference,comparison=label,metric=metric,
                            benefit_sign='negative' if family=='autoencoder' else 'positive')
    return [metadata[k]|interval([np.mean(v) for v in by.values()],rng) for k,by in sorted(paired.items())]


def escape(value):return str(value).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')


def tex_table(path,headers,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{tabular}{'+('l'*len(headers))+'}',r'\hline',' & '.join(map(escape,headers))+r' \\',r'\hline']
    for row in rows:lines.append(' & '.join(map(escape,row))+r' \\')
    lines += [r'\hline',r'\end{tabular}'];path.write_text('\n'.join(lines)+'\n')


def figure_metrics(root,rows):
    core=[r for r in rows if r['tier']=='core'];fig,axes=plt.subplots(2,3,figsize=(13,7));colors=plt.get_cmap('tab10')
    for ax,family in zip(axes.ravel(),FAMILIES):
        rs=[r for r in core if r['family']==family];ties=list(dict.fromkeys(r['tie'] for r in rs));opts=list(dict.fromkeys(r['optimizer'] for r in rs))
        for j,opt in enumerate(opts):
            xs=[];ys=[];es=[]
            for i,tie in enumerate(ties):
                by=defaultdict(list)
                for row in rs:
                    if row['tie']==tie and row['optimizer']==opt:
                        value=numeric(row,primary_metric(row))
                        if value is not None:by[row['split_seed']].append(value)
                v=np.array([np.mean(a) for a in by.values()])
                if not len(v):continue
                xs.append(i+(j-(len(opts)-1)/2)*.12);ys.append(v.mean());es.append(v.std(ddof=1) if len(v)>1 else 0.)
            ax.errorbar(xs,ys,yerr=es,fmt='o',ms=4,capsize=2,label=opt,color=colors(j))
        ax.set_title(family);ax.set_xticks(range(len(ties)),ties,rotation=35,ha='right',fontsize=8)
        ax.set_ylabel('test mean infidelity' if family=='autoencoder' else 'test balanced accuracy');ax.grid(axis='y',alpha=.2)
    if core:axes[0,0].legend(fontsize=7)
    fig.suptitle('Core study: dataset-seed mean ± standard deviation; runs averaged within seed')
    fig.tight_layout();fig.savefig(root/'figures/core_metrics.pdf');fig.savefig(root/'figures/core_metrics.png',dpi=160);plt.close(fig)


def figure_landscape(root,rows):
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,family in zip(axes.ravel(),FAMILIES):
        rs=[r for r in rows if r['family']==family and r['tier']=='core'];ties=list(dict.fromkeys(r['tie'] for r in rs))
        for i,tie in enumerate(ties):
            values=[numeric(r,'cluster_effective_clusters') for r in rs if r['tie']==tie];values=[v for v in values if v is not None]
            ax.scatter(np.full(len(values),i),values,s=24)
        ax.set_title(family);ax.set_xticks(range(len(ties)),ties,rotation=35,ha='right',fontsize=8);ax.set_ylabel('effective endpoint clusters');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Independent BFGS finite-start endpoint diversity; not basin-volume estimates');fig.tight_layout()
    fig.savefig(root/'figures/endpoint_diversity.pdf');fig.savefig(root/'figures/endpoint_diversity.png',dpi=160);plt.close(fig)


def figure_convergence(root,rows):
    fig,axes=plt.subplots(2,3,figsize=(13,7));grid=np.linspace(0,1,101)
    for ax,family in zip(axes.ravel(),FAMILIES):
        selected=[r for r in rows if r['family']==family and r['tier']=='core' and r['tie']==STRUCTURED[family][0]]
        by=defaultdict(list)
        for row in selected:
            path=root/'fit'/f'{row["id"]}.npz'
            with np.load(path,allow_pickle=False) as z:t=z['trace']
            # Step interpolation of the known incumbent, retaining its last
            # value after early convergence. No extra calls are imputed.
            indices=np.clip(np.searchsorted(t[:,0]/float(row['budget']),grid,side='right')-1,0,len(t)-1)
            by[row['optimizer']].append(t[indices,2])
        for optimizer,v in by.items():ax.plot(grid,np.mean(v,axis=0),label=optimizer,lw=1.3)
        ax.set_title(family+' / '+STRUCTURED[family][0]);ax.set_xlabel('fraction of call ceiling');ax.set_ylabel('best training loss');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'figures/core_convergence.pdf');fig.savefig(root/'figures/core_convergence.png',dpi=160);plt.close(fig)


def figure_hessians(root,rows):
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,family in zip(axes.ravel(),FAMILIES):
        rs=[r for r in rows if r['family']==family and r['tier']=='core'];ties=list(dict.fromkeys(r['tie'] for r in rs))
        for i,tie in enumerate(ties):
            for row in rs:
                if row['tie']!=tie:continue
                if row.get('fd_reliable') not in ('True','true','1'):continue
                value=numeric(row,'eigen_min')
                if value is None:continue
                ax.scatter(i+(-.07 if row['point']=='best_observed' else .07),value,marker='o' if row['mode']=='full' else 'x',
                           color='C0' if row['point']=='best_observed' else 'C1',s=24)
        ax.axhline(0,color='gray',lw=.6);ax.set_title(family);ax.set_xticks(range(len(ties)),ties,rotation=35,ha='right',fontsize=8)
        ax.set_ylabel('smallest diagnosed curvature');ax.grid(axis='y',alpha=.2)
    fig.suptitle('FD-consistent Hessians: blue best observed, orange random; ○ full, × projected')
    fig.tight_layout();fig.savefig(root/'figures/hessian_curvature.pdf');fig.savefig(root/'figures/hessian_curvature.png',dpi=160);plt.close(fig)


def optimizer_links(fits,landscapes):
    output=[]
    for landscape in landscapes:
        matched=[r for r in fits if all(r[k]==landscape[k] for k in ('case','split_seed','tie','map_seed'))]
        groups=defaultdict(list)
        for row in matched:groups[row['optimizer']].append(row)
        reference=[numeric(r,'best_train_loss') for r in groups.get('cobyla',[])];reference=[v for v in reference if v is not None]
        for optimizer,rs in groups.items():
            losses=[numeric(r,'best_train_loss') for r in rs];losses=[v for v in losses if v is not None]
            output.append(dict(landscape_id=landscape['id'],case=landscape['case'],family=landscape['family'],split_seed=landscape['split_seed'],tie=landscape['tie'],map_seed=landscape['map_seed'],
                    optimizer=optimizer,run_count=len(rs),mean_best_train_loss=float(np.mean(losses)) if losses else None,
                    mean_training_loss_gain_vs_COBYLA=float(np.mean(reference)-np.mean(losses)) if losses and reference else None,
                    gradient_probe_median=numeric(landscape,'gradient_physical_rms_median'),effective_endpoint_clusters=numeric(landscape,'cluster_effective_clusters'),
                    normal_gradient_rms=numeric(landscape,'best_normal_gradient_rms'),shared_group_cancellation=numeric(landscape,'best_shared_group_cancellation')))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out)
    complete=set(json.loads((root/'block_completion.json').read_text())['complete_block_ids'])
    rows=[r for r in read_csv(root/'fits.csv') if r['block_id'] in complete]
    baselines=[r for r in read_csv(root/'baselines.csv') if r['block_id'] in complete]
    landscape=[r for r in read_csv(root/'landscape.csv') if r['block_id'] in complete]
    hessians=[r for r in read_csv(root/'hessians.csv') if r['block_id'] in complete]
    transfers=[r for r in read_csv(root/'transfers.csv') if r['block_id'] in complete]
    rng=np.random.default_rng(316019);summary=summarize(rows,rng)
    write_csv(root/'summary.csv',summary);write_csv(root/'baseline_summary.csv',summarize(baselines,rng))
    write_csv(root/'transfer_summary.csv',summarize(transfers,rng))
    tying=contrasts(rows,rng,'tie');opts=contrasts(rows,rng,'optimizer')
    write_csv(root/'tying_contrasts.csv',tying);write_csv(root/'optimizer_contrasts.csv',opts)
    write_csv(root/'landscape_optimizer_links.csv',optimizer_links(rows,landscape))
    (root/'figures').mkdir(exist_ok=True);(root/'tables').mkdir(exist_ok=True)
    if rows:figure_metrics(root,rows);figure_convergence(root,rows)
    if landscape:figure_landscape(root,landscape)
    if hessians:figure_hessians(root,hessians)
    counts=defaultdict(lambda:[0,set()])
    for row in rows:counts[row['family']][0]+=1;counts[row['family']][1].add(row['split_seed'])
    tex_table(root/'tables/coverage.tex',('Family','Complete primary fits','Dataset seeds'),[(f,v[0],len(v[1])) for f,v in counts.items()])
    out=[]
    for family in FAMILIES:
        # One compact table per family, preserving optimizer and map; do not
        # pool incomparable task metrics or dimensions across architectures.
        rs=[r for r in summary if r['family']==family and r['tier']=='core' and r['metric'].startswith('test_') and r['metric'] in ('test_balanced_accuracy','test_mean_infidelity')]
        tr=[(r['tie'],r['optimizer'],f'{r["mean"]:.4f}',f'[{r["low"]:.4f}, {r["high"]:.4f}]',r['n_dataset_seeds']) for r in rs]
        tex_table(root/f'tables/{family}_metrics.tex',('Tying','Optimizer','Mean','Exploratory 95% interval','Seeds'),tr)
    (root/'analysis_notes.md').write_text('Primary summaries use complete blocks. Numerical runs are averaged within dataset seed. Bootstrap intervals are exploratory, especially for compact profiles. Repeated graph trials can draw overlapping finite graph orbits; they describe procedure variability. No pooled architecture ranking, causal quantum advantage, or optimizer dominance is inferred. Landscape diagnostics have their own training-only quench budget.\n')
    print(f'Analyzed {len(rows)} primary fits from {len(complete)} complete blocks; {len(tying)} tying contrasts and {len(opts)} optimizer contrasts.')


if __name__=='__main__':main()
