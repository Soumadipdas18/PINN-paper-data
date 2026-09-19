"""Validation-only sequential resolution, capacity, task-weight and physics searches."""
from __future__ import annotations
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
from .config import *
from .metrics import rmse
from .physics import logf_from_temperature_np
from .io import atomic_csv,atomic_json


def choose_resolution(rows,tolerance=0.01):
    frame=pd.DataFrame(rows)
    minimum=frame['validation_score'].min()
    candidates=frame[frame['validation_score']<=minimum*(1+tolerance)].copy()
    candidates['points']=candidates['n_heat']+candidates['n_cool']
    return candidates.sort_values(['points','validation_score','trajectory_RMSE_C','direct_logF_RMSE']).iloc[0]


def config_from_row(row):
    fields=ModelConfig.__dataclass_fields__
    payload={k:row[k] for k in fields}
    for key in ('n_heat','n_cool','units','encoder_units','batch_size'):
        payload[key]=int(payload[key])
    return ModelConfig(**payload)


def best_row(rows):
    return pd.DataFrame(rows).sort_values(['validation_score','trajectory_RMSE_C','direct_logF_RMSE']).iloc[0]


def run_stages_a_d(runner,splits):
    check_mode=runner.options.mode=='check'
    resolutions=((8,8),(16,16),(24,24)) if check_mode else RESOLUTION_CANDIDATES
    units=(8,12) if check_mode else UNIT_CANDIDATES
    base=ModelConfig(units=units[0],encoder_units=units[0],batch_size=16 if check_mode else 64)
    rows=[]; quadrature=[]
    def run(stage,name,config):
        result=runner.run(name,config,SEARCH_SEED,runner.options.epochs,True,splits)
        row=dict(result.metrics,stage=stage,selected=False)
        rows.append(row)
        atomic_csv(runner.options.output/'_internal/stages_A_D_progress.csv',pd.DataFrame(rows))
        return row
    def mark(row):
        for r in rows:
            if r['run_name']==row['run_name']:
                r['selected']=True
    for nh,nc in resolutions:
        cfg=replace(base,n_heat=nh,n_cool=nc)
        run('A',f'resolution_{nh}_{nc}',cfg)
        # Kaggle computed the resampling/quadrature diagnostic over all 10,526 cases.
        ii=np.arange(len(runner.bundle.x),dtype=np.int64)
        truth=runner.bundle.logf[ii].reshape(-1)
        logpred=logf_from_temperature_np(runner.bundle.temperature(nh,nc)[ii],runner.bundle.hm[ii],runner.bundle.cm[ii],nh,nc).reshape(-1)
        relative=100*np.abs(np.power(10.,logpred-truth)-1.)
        quadrature.append({'n_heat':nh,'n_cool':nc,'logF_RMSE_resampled_true_trajectory_vs_stored_F':rmse(logpred,truth),'F_relative_difference_mean_percent':float(relative.mean()),'F_relative_difference_p95_percent':float(np.percentile(relative,95)),'F_relative_difference_p99_percent':float(np.percentile(relative,99)),'F_relative_difference_max_percent':float(relative.max())})
    selected=choose_resolution([r for r in rows if r['stage']=='A'],runner.options.resolution_tolerance); mark(selected)
    cfg=config_from_row(selected)
    for u in units:
        for d in DROPOUT_CANDIDATES:
            run('B',f'capacity_u{u}_d{d:.2f}',replace(cfg,units=u,encoder_units=u,dropout=d))
    selected=best_row([r for r in rows if r['stage']=='B']); mark(selected); cfg=config_from_row(selected)
    for wt,wf,wq in SUPERVISED_WEIGHT_CANDIDATES:
        run('C',f'weights_T{wt:g}_F{wf:g}_Q{wq:g}',replace(cfg,temperature_weight=wt,logF_weight=wf,retention_weight=wq))
    selected=best_row([r for r in rows if r['stage']=='C']); mark(selected); cfg=config_from_row(selected)
    for lam in PHYSICS_WEIGHT_CANDIDATES:
        run('D',f'lambda_{lam:g}',replace(cfg,lambda_phys=lam))
    selected=best_row([r for r in rows if r['stage']=='D' and r['lambda_phys']>0]); mark(selected)
    cfg=config_from_row(selected)
    atomic_json(runner.options.output/'_internal/stage_A_D_selection.json',{'selected':asdict(cfg),'resolution_rule':'smallest tested resolution within 1% of minimum validation score','quadrature_diagnostic_partition':'all cases','test_examined':False})
    return cfg,{'Stages A-D runs':pd.DataFrame(rows),'Resolution sensitivity':pd.DataFrame(quadrature)}
