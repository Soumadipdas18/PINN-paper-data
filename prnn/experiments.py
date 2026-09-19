"""Shared fit/predict service with durable, scientifically scoped reuse."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import gc
import json
import os
import numpy as np
import pandas as pd
from .config import ModelConfig,Normalization
from .io import atomic_json,atomic_csv,atomic_npz,identity_hash,sha256_array,sha256_file,source_fingerprint
from .losses import weighted_loss_history
from .metrics import detailed_metrics


@dataclass
class ExperimentResult:
    key: str
    config: ModelConfig
    norm: Normalization
    directory: Path
    metadata: dict
    metrics: dict
    history: pd.DataFrame
    reused: bool


class ExperimentRunner:
    def __init__(self,bundle,backend,options):
        self.bundle=bundle; self.backend=backend; self.options=options
        self.root=options.output/'_internal/cache'; self.root.mkdir(parents=True,exist_ok=True)
        self.source_hash=source_fingerprint()
        self.selection_frozen=False
        self.events=[]
        self.trained_keys=set()
        self._prepared_key=None
        self._prepared=None

    def prepare(self,config,splits):
        key=(config.n_heat,config.n_cool,sha256_array(splits['train']))
        if key!=self._prepared_key:
            self._prepared=self.bundle.prepare(config,splits)
            self._prepared_key=key
        return self._prepared

    def signature(self,config,seed,epochs,restore_best,splits,fit_context=None,evaluate_validation=True):
        return {
            'dataset_sha256':self.bundle.dataset_hash,
            'source_sha256':self.source_hash,
            'config':asdict(config),
            'seed':int(seed),'epochs':int(epochs),'restore_best_validation_weights':bool(restore_best),
            'splits':{k:sha256_array(v) for k,v in splits.items()},
            'environment':self.backend.environment,
            'mode':self.options.mode,
            'fit_context':fit_context or 'shared',
            'evaluate_validation':bool(evaluate_validation),
        }

    def _event(self,**payload):
        self.events.append(payload)
        atomic_json(self.options.output/'_internal/run_events.json',self.events)

    def run(self,name,config,seed,epochs,restore_best,splits,fit_context=None,evaluate_validation=True):
        signature=self.signature(config,seed,epochs,restore_best,splits,fit_context,evaluate_validation)
        key=identity_hash(signature)
        directory=self.root/key
        complete=directory/'complete.json'
        reused=False
        if complete.exists() and (self.options.resume or key in self.trained_keys):
            meta=json.loads(complete.read_text())
            if meta['signature']!=signature:
                raise RuntimeError('Cache identity mismatch; refusing to reuse a different experiment.')
            for filename,expected in meta['files'].items():
                path=directory/filename
                if not path.exists() or sha256_file(path)!=expected:
                    raise RuntimeError(f'Incomplete or corrupt cache artifact: {path}. Remove this cache directory and rerun.')
            metrics=json.loads((directory/'validation_metrics.json').read_text())
            norm=Normalization(**json.loads((directory/'normalization.json').read_text()))
            history=pd.read_csv(directory/'history.csv')
            reused=True
            self._event(event='reuse_training',name=name,key=key,source=meta['run_name'])
            print(f'Reusing {meta["run_name"]} for {name}',flush=True)
        else:
            print(f'Training {name}: {config.model_kind}, {config.n_heat}+{config.n_cool} points, seed={seed}, epochs={epochs}',flush=True)
            directory.mkdir(parents=True,exist_ok=True)
            prepared=self.prepare(config,splits)
            model=self.backend.make_model(config,prepared.norm,seed)
            history,best_epoch,seconds=self.backend.fit(model,prepared,splits,config,seed,epochs,restore_best)
            metrics={}
            prediction=None
            if evaluate_validation:
                prediction=self.backend.predict(model,prepared,splits['validation'],config)
                self._check_prediction(prediction)
                metrics=detailed_metrics(prediction,config,prepared.norm)
            metrics.update({'seed':int(seed),'best_epoch':int(best_epoch),'epochs_executed':int(epochs),'training_seconds':seconds,'parameter_count':int(model.count_params()),**asdict(config)})
            # Completed-model saves, not per-epoch checkpoints. The completion
            # marker is written last so interrupted runs are never treated as valid.
            model_tmp=directory/'model.tmp.keras'
            model.save(model_tmp)
            os.replace(model_tmp,directory/'model.keras')
            atomic_csv(directory/'history.csv',history)
            atomic_csv(directory/'weighted_losses.csv',weighted_loss_history(history,config))
            atomic_json(directory/'normalization.json',asdict(prepared.norm))
            atomic_json(directory/'validation_metrics.json',metrics)
            if prediction is not None:
                atomic_npz(directory/'validation_predictions.npz',**prediction)
            artifact_names=['model.keras','history.csv','normalization.json','validation_metrics.json','weighted_losses.csv']
            if prediction is not None:
                artifact_names.append('validation_predictions.npz')
            files={n:sha256_file(directory/n) for n in artifact_names}
            meta={'signature':signature,'run_name':name,'files':files,'training_seconds':seconds,'best_epoch':best_epoch,'parameter_count':int(model.count_params())}
            atomic_json(complete,meta)
            norm=prepared.norm
            self.trained_keys.add(key)
            self._event(event='train',name=name,key=key,training_seconds=seconds)
            del model
            gc.collect()
        metrics=dict(metrics,run_name=name)
        return ExperimentResult(key,config,norm,directory,meta,metrics,history,reused)

    @staticmethod
    def _check_prediction(prediction):
        for name,value in prediction.items():
            if not np.isfinite(np.asarray(value)).all():
                raise FloatingPointError(f'Nonfinite prediction in {name}; refusing to export fabricated or clipped results.')

    def validation_prediction(self,result):
        path=result.directory/'validation_predictions.npz'
        if not path.exists():
            raise RuntimeError('This run did not request detailed validation predictions.')
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def freeze_selection(self,selection):
        atomic_json(self.options.output/'_internal/frozen_selection.json',selection)
        self.selection_frozen=True
        self._event(event='selection_frozen')

    def test_prediction(self,result,splits,warmup_indices=None):
        if not self.selection_frozen:
            raise RuntimeError('Test evaluation is forbidden until model-development decisions are frozen.')
        idx=splits['test']
        key=identity_hash({'model':result.key,'test_indices':sha256_array(idx),'warmup_indices':None if warmup_indices is None else sha256_array(warmup_indices)})
        path=result.directory/f'test_{key}.npz'
        stamp=result.directory/f'test_{key}.json'
        if path.exists() and stamp.exists():
            if sha256_file(path)!=json.loads(stamp.read_text())['sha256']:
                raise RuntimeError('Cached test predictions are corrupt.')
            with np.load(path,allow_pickle=False) as z:
                prediction={k:z[k] for k in z.files}
            self._event(event='reuse_test_prediction',key=result.key,test_key=key)
            return prediction
        prepared=self.prepare(result.config,splits)
        model=self.backend.load_model(result.directory/'model.keras')
        prediction=self.backend.predict(model,prepared,idx,result.config,warmup_indices=warmup_indices)
        self._check_prediction(prediction)
        atomic_npz(path,**prediction)
        atomic_json(stamp,{'sha256':sha256_file(path),'cases':len(idx)})
        self._event(event='evaluate_test',key=result.key,test_key=key,cases=len(idx))
        del model
        gc.collect()
        return prediction

    def reuse_statistics(self):
        return {
            'training_executions_this_invocation':sum(e['event']=='train' for e in self.events),
            'training_reuses_this_invocation':sum(e['event']=='reuse_training' for e in self.events),
            'test_forward_passes_this_invocation':sum(e['event']=='evaluate_test' for e in self.events),
            'test_prediction_reuses_this_invocation':sum(e['event']=='reuse_test_prediction' for e in self.events),
            'unique_training_seconds_this_invocation':sum(e.get('training_seconds',0.) for e in self.events if e['event']=='train'),
        }
