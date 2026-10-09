"""Predeclared complete coverage tiers; no outcome-based expansion or stopping."""
from copy import deepcopy
from qml_tying.models import Model,ModelSpec
from qml_tying.io import record_id


def make_plan(protocol,profile=None):
    profile=profile or protocol.get('profile','standard')
    if profile=='auto': profile='standard'
    settings=protocol['profiles'][profile];blocks=[];fit_ids={}
    def add_block(case,kind,split,run=None):
        block_id=record_id('block-v3',case,kind,split,run,profile)
        block=dict(id=block_id,family=case['spec']['family'],case=case['name'],kind=kind,tier=case['tier'],tasks=[])
        def task(kind,mode,**extra):
            cfg=dict(case=deepcopy(case),split_seed=split,run_seed=run if run is not None else 0,
                     mode=mode,map_seed=case.get('map_seed',17),profile=profile,
                     n_samples=case.get('n_samples',settings['n_samples']),teacher_seed=case.get('teacher_seed',77131),
                     checkpoints=settings.get('checkpoints',6),**extra)
            cfg['_cluster']=dict(block_id=block_id,tier=case['tier'],case=case['name'])
            tid=record_id('task-v3',kind,cfg)
            t=dict(id=tid,block_id=block_id,family=case['spec']['family'],kind=kind,config=cfg,record=f'{kind}/{tid}.json')
            block['tasks'].append(t);return t
        if kind=='comparison':
            dimensions=set()
            for mode in case['ties']:
                dimensions.add(Model(ModelSpec(**case['spec']),mode,case.get('map_seed',17)).tying.d)
                for optimizer in protocol['optimizers']:
                    t=task('fit',mode,optimizer=optimizer,budget=settings['fit_budget'])
                    fit_ids[(case['name'],split,run,mode,optimizer)]=t['id']
            if case['spec']['family']!='autoencoder':
                for d in sorted(dimensions): task('baseline',f'mlp_D{d}',target_parameters=d,budget=settings['mlp_budget'])
        elif kind=='expressivity':
            for mode in case['ties']: task('expressivity',mode,expressivity_cfg=settings['expressivity'])
        elif kind=='landscape':
            lc=dict(settings['landscape']);lc.update(case.get('landscape',{}) if profile=='standard' else {})
            for mode in case['ties']:
                ids=[]
                for start in range(lc['starts']):
                    ids.append(task('quench',mode,start=start,budget=lc['quench_budget'])['id'])
                t=task('landscape',mode,landscape_cfg=lc,quench_ids=ids)
                t['dependencies']=[f'quench/{i}.json' for i in ids]
        else: raise ValueError(kind)
        blocks.append(block)
    tiers=protocol.get('tier_order',['core','scale','teacher','symmetry_mismatch','encoding'])
    for tier in tiers:
        if tier=='transfer':
            lookup={c['name']:c for c in protocol['cases']}
            for transfer in protocol.get('transfers',[]):
                source=lookup[transfer['source']];target=deepcopy(lookup[transfer['target']]);target['tier']='transfer'
                for split in settings['split_seeds']:
                    for run in settings['run_seeds']:
                        bid=record_id('transfer-block-v3',transfer,split,run,profile)
                        block=dict(id=bid,family=target['spec']['family'],case=target['name'],kind='transfer',tier='transfer',tasks=[])
                        for mode in transfer['modes']:
                            sm=Model(ModelSpec(**source['spec']),mode,source.get('map_seed',17));tm=Model(ModelSpec(**target['spec']),mode,source.get('map_seed',17))
                            if sm.tying.d!=tm.tying.d:raise ValueError('Transfer must reuse an unchanged parameter vector; no padding permitted.')
                            for optimizer in protocol['optimizers']:
                                parent=fit_ids[(source['name'],split,run,mode,optimizer)]
                                cfg=dict(case=target,split_seed=split,run_seed=run,mode=mode,map_seed=source.get('map_seed',17),profile=profile,
                                         n_samples=target.get('n_samples',settings['n_samples']),teacher_seed=target.get('teacher_seed',77131),
                                         source_fit_id=parent,source_case=source['name'],optimizer=optimizer,budget=settings['fit_budget'],
                                         _cluster=dict(block_id=bid,tier='transfer',case=target['name']))
                                tid=record_id('task-v3','transfer',cfg)
                                block['tasks'].append(dict(id=tid,block_id=bid,family=target['spec']['family'],kind='transfer',config=cfg,
                                                           record=f'transfer/{tid}.json',dependencies=[f'fit/{parent}.json']))
                        blocks.append(block)
            continue
        cases=[c for c in protocol['cases'] if c['tier']==tier]
        # Interleave all architectures within every repetition. Each selected
        # tier has the full paired optimizer/map comparison and its diagnostics.
        for split in settings['split_seeds']:
            for run in settings['run_seeds']:
                for case in cases: add_block(case,'comparison',split,run)
        for case in cases:
            add_block(case,'expressivity',settings['split_seeds'][0])
        for case in cases:
            diagnostic_splits=min(settings.get('diagnostic_splits',1),case.get('diagnostic_splits',999))
            for split in settings['split_seeds'][:diagnostic_splits]: add_block(case,'landscape',split)
    return dict(schema=3,protocol=protocol,protocol_hash=record_id(protocol),profile=profile,
                profile_settings=settings,blocks=blocks,comparison_unit='complete map-by-optimizer block, with numerical runs averaged within independent dataset seed',
                selected=False,coverage_policy='all core families first; timing selection includes whole optional tiers')
