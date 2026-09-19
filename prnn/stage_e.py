"""Repeated-seed validation comparison of the PRNN and baseline architectures."""
from __future__ import annotations
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
from .config import ModelConfig
from .metrics import validation_safety_margin
from .io import atomic_json,atomic_csv
from .workbook import SCHEMA


def summarize_stage_e(frame):
    columns=SCHEMA['Stage E summary']['columns']
    rows=[]
    for model,group in frame.groupby('comparison_model',sort=False):
        row={'comparison_model':model}
        for name in columns[1:]:
            field,stat=name.rsplit('_',1)
            values=pd.to_numeric(group[field],errors='coerce')
            row[name]=float(values.mean()) if stat=='mean' else float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def run_stage_e(runner,config,splits):
    # Full-study Stage E is the frozen 200+200 comparison used in Kaggle.
    if runner.options.mode=='full':
        config=replace(config,n_heat=200,n_cool=200)
    candidates=[('PRNN_LSTM',config),('data_LSTM',replace(config,lambda_phys=0.)),('GRU',replace(config,model_kind='gru',lambda_phys=0.)),('time_conditioned_MLP',replace(config,model_kind='mlp',lambda_phys=0.))]
    rows=[]; predictions=[]
    for label,cfg in candidates:
        for seed in runner.options.seeds:
            name='stage_E_'+label+f'_seed{seed}'
            result=runner.run(name,cfg,seed,runner.options.epochs,True,splits)
            rows.append(dict(result.metrics,comparison_model=label,comparison_type='primary',selected_model=False))
            if label=='PRNN_LSTM':
                predictions.append(runner.validation_prediction(result))
            atomic_csv(runner.options.output/'_internal/stage_E_progress.csv',pd.DataFrame(rows))
    frame=pd.DataFrame(rows)
    summary=summarize_stage_e(frame)
    winner=str(summary.loc[summary['validation_score_mean'].idxmin(),'comparison_model'])
    frame['selected_model']=frame['comparison_model'].eq(winner)
    epochs={label:int(np.rint(np.median(group['best_epoch']))) for label,group in frame.groupby('comparison_model',sort=False)}
    margin,margin_info=validation_safety_margin(predictions)
    frozen={'selected_prnn_config':asdict(config),'validation_winner':winner,'final_reported_model':'PRNN_LSTM','final_seed':runner.options.final_seed,'final_epochs':epochs['PRNN_LSTM'],'epochs_by_model':epochs,'F_target_min':3.,'design_margin_min':margin,'margin_diagnostic':margin_info,'seeds':list(runner.options.seeds),'test_examined_during_selection':False,'prnn_is_validation_winner':winner=='PRNN_LSTM'}
    if winner!='PRNN_LSTM':
        print(f'Validation winner is {winner}; final PRNN evaluation follows the predefined study protocol.',flush=True)
    atomic_json(runner.options.output/'_internal/stage_E_selection.json',frozen)
    return frozen,{'Stage E runs':frame,'Stage E summary':summary}
