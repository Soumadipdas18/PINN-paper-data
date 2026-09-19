"""Fixed structured stress tests; held-out rows never enter their own fit or normalization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import *
from .data import _split,validate_partitions
from .metrics import nearest_training_points,build_case_prediction_table,metrics_from_structured_prediction,aggregate_run_metrics,build_distance_bin_tables
from .io import atomic_csv


def create_protocols(bundle,random_splits,strict_reference=True):
    x=bundle.x; n=len(x); indices=np.arange(n,dtype=np.int64); tol=1e-5
    band=(x[:,3]>=128-tol)&(x[:,3]<=132+tol)
    corner=(x[:,3]>=136-tol)&(x[:,1]<=8+tol)
    width=.05*(INPUT_UPPER_BOUNDS-INPUT_LOWER_BOUNDS)
    boundary=np.any((x<=INPUT_LOWER_BOUNDS+width+tol)|(x>=INPUT_UPPER_BOUNDS-width-tol),axis=1)
    masks=dict(zip(PROTOCOL_ORDER[1:],[band,corner,boundary]))
    definitions={
        'random_interpolation':'Stratified 70/15/15 split',
        'retort_temperature_band':'128 <= temp_retort <= 132 degC',
        'high_temperature_short_heating_corner':'temp_retort >= 136 degC and time_heat <= 8 min',
        'domain_boundary_shell':'Any input within the outer 5% of its prescribed range: CUT <= 1.2 or >= 4.8 min; time_heat <= 5.75 or >= 19.25 min; time_cool <= 1.2 or >= 4.8 min; temp_retort <= 121 or >= 139 degC',
    }
    interpretations=dict(zip(PROTOCOL_ORDER,['In-domain interpolation control','Prediction across an unsampled internal temperature band','Prediction in an excluded operating-space corner','Prediction toward excluded edges of the defined domain']))
    protocols={'random_interpolation':random_splits}
    for offset,name in enumerate(PROTOCOL_ORDER[1:],1):
        test=indices[masks[name]]; eligible=indices[~masks[name]]
        if len(test)==0:
            raise ValueError(f'No cases satisfy holdout definition {name}.')
        train,val=_split(eligible,x,bundle.logf,VALIDATION_FRACTION/(1-TEST_FRACTION),SPLIT_SEED+100*offset,strict=not bundle.software_check)
        protocols[name]={'train':np.sort(train),'validation':np.sort(val),'test':np.sort(test)}
        if strict_reference and not bundle.software_check and len(test)!=EXPECTED_HOLDOUT_COUNTS[name]:
            raise ValueError(f'{name} has {len(test)} test cases; reference expects {EXPECTED_HOLDOUT_COUNTS[name]}.')
    meta=[]; assignments=[]
    for name,splits in protocols.items():
        validate_partitions(splits,n)
        meta.append({'protocol':name,'protocol_label':PROTOCOL_LABELS[name],'interpretation':interpretations[name],'definition':definitions[name],'train_cases':len(splits['train']),'validation_cases':len(splits['validation']),'test_cases':len(splits['test'])})
        for role,ii in splits.items():
            assignments.append(pd.DataFrame({'protocol':name,'protocol_label':PROTOCOL_LABELS[name],'row_index':ii,'simulation_id':bundle.ids[ii],'role':role}))
    return protocols,pd.DataFrame(meta),pd.concat(assignments,ignore_index=True)


def run_structured(runner,selection,protocols,metadata,assignments):
    config=ModelConfig(**selection['selected_prnn_config'])
    rows=[]; cases=[]
    for order,name in enumerate(PROTOCOL_ORDER):
        splits=protocols[name]
        distance,nearest=nearest_training_points(runner.bundle.x,splits['train'],splits['test'])
        for seed in runner.options.seeds:
            result=runner.run(f'{name}_seed{seed}',config,seed,selection['final_epochs'],False,splits,fit_context=f'structured:{name}',evaluate_validation=False)
            p=runner.test_prediction(result,splits)
            frame=build_case_prediction_table(name,seed,splits['test'],runner.bundle.ids,runner.bundle.x,p,distance,nearest,config)
            metric=metrics_from_structured_prediction(frame,p,result.norm,config)
            metric.update({'protocol':name,'protocol_label':PROTOCOL_LABELS[name],'protocol_order':order,'seed':seed,'train_cases':len(splits['train']),'validation_cases':len(splits['validation']),'test_cases':len(splits['test']),'fixed_epochs':selection['final_epochs'],'best_validation_epoch_observed_but_not_restored':result.metadata['best_epoch'],'training_seconds':result.metadata['training_seconds'],'parameter_count':result.metadata['parameter_count']})
            rows.append(metric); cases.append(frame)
            atomic_csv(runner.options.output/'_internal/holdout_progress.csv',pd.DataFrame(rows))
    runs=pd.DataFrame(rows); all_cases=pd.concat(cases,ignore_index=True)
    by_seed,summary=build_distance_bin_tables(all_cases)
    return {'Holdout protocols':metadata,'Holdout runs':runs,'Holdout summary':aggregate_run_metrics(runs),'Distance bins runs':by_seed,'Distance bins summary':summary,'Holdout predictions':all_cases,'Holdout assignments':assignments}
