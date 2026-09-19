"""Final PRNN fit, one-time random-test prediction, and paper data tables."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
import pandas as pd
from .config import *
from .data import build_sequence_features
from .metrics import final_test_metrics,rmse
from .physics import logf_from_temperature_np
from .workbook import SCHEMA


def prediction_table(bundle,idx,p):
    truth=p['true_logf'].reshape(-1); head=p['logf'].reshape(-1); trajectory=p['logf_phys'].reshape(-1)
    q=p['q'].reshape(-1); qtrue=p['true_q'].reshape(-1)
    return pd.DataFrame({'row_index':idx,'simulation_id':bundle.ids[idx],'CUT_min':bundle.x[idx,0],'heating_time_min':bundle.x[idx,1],'cooling_time_min':bundle.x[idx,2],'retort_temperature_C':bundle.x[idx,3],'true_log10_F':truth,'predicted_direct_log10_F':head,'predicted_trajectory_log10_F':trajectory,'true_F_min':10.**truth,'predicted_direct_F_min':10.**head,'predicted_trajectory_F_min':10.**trajectory,'true_retention_index':qtrue,'predicted_retention_index':q,'direct_minus_trajectory_log10_F':head-trajectory,'direct_log10_F_error':head-truth,'trajectory_log10_F_error':trajectory-truth,'retention_error':q-qtrue})


def trajectory_tables(bundle,idx,p,config):
    selected=np.argsort(bundle.x[idx,3])[np.linspace(0,len(idx)-1,min(8,len(idx))).astype(int)]
    frames=[]; metadata=[]
    for example,j in enumerate(selected,1):
        i=idx[j]; cut,hold,cool,tret=map(float,bundle.x[i]); duration=cut+hold
        h=bundle.heat[i]; c=bundle.cool[i]
        heat_t=np.linspace(0,duration,len(h))
        cool_t=np.linspace(0,cool,len(c)+1)[1:]
        predh=np.interp(heat_t,np.linspace(0,duration,config.n_heat),p['temp'][j,:config.n_heat,0])
        predc=np.interp(cool_t,np.linspace(0,cool,config.n_cool),p['temp'][j,config.n_heat:,0])
        times=np.concatenate([heat_t,duration+cool_t])
        frames.append(pd.DataFrame({'example_id':example,'time_min':times,'CFD_temperature_C':np.concatenate([h,c]),'predicted_temperature_C':np.concatenate([predh,predc]),'heating_stage_duration_min':duration}))
        metadata.append({'example_id':example,'simulation_id':bundle.ids[i],'CUT_min':cut,'heating_hold_min':hold,'cooling_min':cool,'retort_temperature_C':tret,'heating_stage_duration_min':duration})
    return pd.concat(frames,ignore_index=True),pd.DataFrame(metadata)


def error_trend_table(bundle,idx,p):
    terr=np.sqrt(np.mean((p['temp']-p['true_temp'])**2,axis=(1,2)))
    ferr=np.abs(p['logf'].reshape(-1)-p['true_logf'].reshape(-1))
    rows=[]
    for column,name in [(3,'temp_retort'),(1,'time_heat'),(2,'time_cool')]:
        edges=np.linspace(INPUT_LOWER_BOUNDS[column],INPUT_UPPER_BOUNDS[column],11)
        vals=bundle.x[idx,column]
        for metric,errors in [('trajectory_RMSE_C',terr),('direct_logF_MAE',ferr)]:
            for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                mask=(vals>=lo)&((vals<hi) if j<9 else (vals<=hi))
                sub=errors[mask]
                rows.append({'input_variable':name,'error_metric':metric,'bin_lower':lo,'bin_upper':hi,'bin_center':(lo+hi)/2,'mean':float(sub.mean()) if len(sub) else np.nan,'standard_deviation':float(sub.std()) if len(sub) else np.nan,'count':len(sub)})
    return pd.DataFrame(rows)


def design_tradeoff(runner,model,config,norm,splits,margin):
    """First feasible grid point plus local bracket refinement, not a global optimum."""
    xtrain=runner.bundle.x[splits['train']]
    lower=xtrain.min(axis=0); upper=xtrain.max(axis=0)
    retorts=np.linspace(120,140,21)
    heating=np.linspace(max(5,float(lower[1])),min(20,float(upper[1])),301)
    threshold=F_SAFETY_THRESHOLD_MIN+margin
    def predict(x):
        xs=((x-np.asarray(norm.x_mean))/np.asarray(norm.x_std)).astype(np.float32)
        seq=build_sequence_features(x,config.n_heat,config.n_cool,norm)
        t,lf,q=runner.backend.raw_predict(model,xs,seq,config.batch_size)
        lft=logf_from_temperature_np(t,x[:,0]+x[:,1],x[:,2],config.n_heat,config.n_cool).reshape(-1)
        return 10.**lf.reshape(-1),10.**lft,q.reshape(-1)
    rows=[]
    for tret in retorts:
        base={'retort_temperature_C':float(tret),'CUT_min':3.,'cooling_time_min':3.,'heating_time_min':np.nan,'total_process_time_min':np.nan,'direct_F_min':np.nan,'trajectory_derived_F_min':np.nan,'conservative_F_min':np.nan,'retention_index':np.nan,'retention_percent':np.nan,'F0_target_min':F_SAFETY_THRESHOLD_MIN,'design_margin_min':margin,'design_threshold_min':threshold,'monotonicity_violations_on_grid':0,'within_training_domain':False,'requires_CFD_verification':True}
        probe=np.array([3.,float(heating[0]),3.,tret])
        if np.any(probe<lower-1e-6) or np.any(probe>upper+1e-6):
            rows.append(base); continue
        cases=np.column_stack([np.full_like(heating,3.),heating,np.full_like(heating,3.),np.full_like(heating,tret)]).astype(np.float32)
        head,traj,q=predict(cases); conservative=np.minimum(head,traj)
        base['monotonicity_violations_on_grid']=int(np.sum(np.diff(conservative)<-1e-4))
        feasible=np.flatnonzero(conservative>=threshold)
        if not len(feasible):
            rows.append(base); continue
        first=int(feasible[0]); high=float(heating[first])
        if first:
            low=float(heating[first-1])
            for _ in range(24):
                middle=(low+high)/2
                h,t,_=predict(np.array([[3.,middle,3.,tret]],np.float32))
                if min(h[0],t[0])>=threshold:
                    high=middle
                else:
                    low=middle
        selected=np.array([[3.,high,3.,tret]],np.float32)
        h,t,q=predict(selected)
        if min(h[0],t[0])<threshold:
            # Stay on the known feasible side of float32 rounding.
            selected[0,1]=np.nextafter(selected[0,1],np.float32(np.inf))
            h,t,q=predict(selected)
        in_domain=bool(np.all(selected[0]>=lower-1e-6) and np.all(selected[0]<=upper+1e-6))
        if not in_domain or min(h[0],t[0])<threshold:
            rows.append(base); continue
        base.update({'heating_time_min':float(selected[0,1]),'total_process_time_min':6+float(selected[0,1]),'direct_F_min':float(h[0]),'trajectory_derived_F_min':float(t[0]),'conservative_F_min':float(min(h[0],t[0])),'retention_index':float(q[0]),'retention_percent':float(100*q[0]),'within_training_domain':True})
        rows.append(base)
    return pd.DataFrame(rows)


def run_main(runner,selection,splits):
    config=ModelConfig(**selection['selected_prnn_config'])
    seed=selection['final_seed']; epochs=selection['final_epochs']
    result=runner.run('main_PRNN_LSTM',config,seed,epochs,False,splits,fit_context='main_final',evaluate_validation=False)
    p=runner.test_prediction(result,splits,warmup_indices=splits['validation'])
    ii=splits['test']
    metrics=final_test_metrics(p,config,result.norm)
    lftrue=logf_from_temperature_np(p['true_temp'],runner.bundle.hm[ii],runner.bundle.cm[ii],config.n_heat,config.n_cool)
    metrics.update({'training_seconds':result.metadata['training_seconds'],'final_epoch':epochs,'training_seed':seed,'parameter_count':result.metadata['parameter_count'],'resampled_true_logF_vs_stored_logF_RMSE':rmse(lftrue,runner.bundle.logf[ii])})
    metrics_table=pd.DataFrame(list(metrics.items()),columns=['Metric','Value'])
    paper=pd.DataFrame([
      ('SHZ temperature trajectory','RMSE (degC)',metrics['trajectory_RMSE_C']),
      ('SHZ temperature trajectory','MAE (degC)',metrics['trajectory_MAE_C']),
      ('Direct log10(F)','RMSE',metrics['direct_logF_RMSE']),
      ('Direct F','MAPE (%)',metrics['direct_F_MAPE_percent']),
      ('Trajectory-derived log10(F)','RMSE',metrics['trajectory_logF_RMSE']),
      ('Direct vs trajectory-derived log10(F)','RMSE',metrics['direct_vs_trajectory_logF_RMSE']),
      ('Ascorbic-acid retention index','RMSE',metrics['retention_RMSE']),
      ('Ascorbic-acid retention index','MAE',metrics['retention_MAE'])],columns=['Output','Metric','Value'])
    history=result.history.copy()
    history.insert(0,'weights_used_for_final_model',history['epoch'].eq(epochs))
    history.insert(0,'training_seed',seed); history.insert(0,'run_name','main_PRNN_LSTM')
    trajectories,trajectory_metadata=trajectory_tables(runner.bundle,ii,p,config)
    prepared=runner.prepare(config,splits)
    model=runner.backend.load_model(result.directory/'model.keras')
    attention=runner.backend.attention(model,prepared,ii,config)[:,:,0]
    attention_table=pd.DataFrame({'sequence_index':np.arange(config.n_heat+config.n_cool),'phase':['heating']*config.n_heat+['cooling']*config.n_cool,'local_normalized_time':np.concatenate([np.linspace(0,1,config.n_heat),np.linspace(0,1,config.n_cool)]),'mean_attention_weight':attention.mean(axis=0),'p05_attention_weight':np.percentile(attention,5,axis=0),'p95_attention_weight':np.percentile(attention,95,axis=0)})
    design=design_tradeoff(runner,model,config,result.norm,splits,selection['design_margin_min'])
    del model
    inf=metrics['inference_ms_per_case']/1000; cfd=3600/CFD_CASES_PER_HOUR
    runtime=pd.DataFrame([{'training_seconds':result.metadata['training_seconds'],'inference_seconds_per_case':inf,'CFD_cases_per_hour_assumption':CFD_CASES_PER_HOUR,'CFD_seconds_per_case_assumption':cfd,'inference_speedup_over_CFD':cfd/inf if inf>0 else np.nan,'training_break_even_cases':result.metadata['training_seconds']/(cfd-inf) if cfd>inf else np.nan}])
    tables={'Main paper table':paper,'Main test metrics':metrics_table,'Main training history':history,'Main trajectories':trajectories,'Trajectory metadata':trajectory_metadata,'Main predictions':prediction_table(runner.bundle,ii,p),'Error trends':error_trend_table(runner.bundle,ii,p),'Design tradeoff':design,'Attention profile':attention_table,'Runtime':runtime}
    return result,tables
