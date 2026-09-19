"""End-to-end orchestration. The final results workbook has exactly 23 separate worksheets."""
from __future__ import annotations
from dataclasses import asdict
import json
import time
import shutil
import numpy as np
import pandas as pd
from .config import *
from .data import find_dataset,load_dataset,make_real_check
from .io import atomic_json,atomic_csv,atomic_npz
from .workbook import writer_preflight,write_workbook,align_table
from .stages_a_d import run_stages_a_d
from .stage_e import run_stage_e
from .main_run import run_main
from .structured_holdouts import create_protocols,run_structured
from .experiments import ExperimentRunner
from .reporting import paper_tables


def setup_table(runner, selection, main_result, splits):
    """Current model, selected hyperparameters, and evaluation settings."""
    env = runner.backend.environment
    rows = []
    def add(section, name, value, units=""):
        rows.append((section, name, value, units))
    add("Dataset", "Simulation cases", len(runner.bundle.df), "cases")
    add("Dataset", "Split seed", SPLIT_SEED)
    for role, label in [("train", "Training cases"), ("validation", "Validation cases"), ("test", "Untouched test cases")]:
        add("Dataset", label, len(splits[role]), "cases")
    add("Process target", "F0 target", F_SAFETY_THRESHOLD_MIN, "min")
    add("Process target", "Design margin", selection["design_margin_min"], "min")
    add("Process target", "Design threshold", F_SAFETY_THRESHOLD_MIN + selection["design_margin_min"], "min")
    add("Process target", "Reference temperature", T_REF_C, "degC")
    add("Process target", "Lethality z-value", Z_F_C, "degC")
    add("Stages A-D", "Search seed", SEARCH_SEED)
    add("Stages A-D", "Fixed epochs per candidate", runner.options.epochs, "epochs")
    add("Stage A", "Resolution selection rule", "Smallest resolution within 1% of the lowest validation score.")
    add("Stage E", "Repeat seeds", ", ".join(map(str, runner.options.seeds)))
    add("Stage E", "Fixed epochs per candidate", runner.options.epochs, "epochs")
    for model, count in selection["epochs_by_model"].items():
        add("Stage E", model + " selected epochs", count, "epochs")
    add("Stage E", "Validation winner", selection["validation_winner"])
    add("Final model", "Training seed", selection["final_seed"])
    add("Final model", "Fixed training epochs", selection["final_epochs"], "epochs")
    for key, value in selection["selected_prnn_config"].items():
        add("Final model", key, value)
    add("Final model", "Parameter count", main_result.metadata["parameter_count"], "parameters")
    add("Final model", "Training time", main_result.metadata["training_seconds"], "s")
    add("Structured holdouts", "Training seeds", ", ".join(map(str, runner.options.seeds)))
    add("Structured holdouts", "Fixed epochs per run", selection["final_epochs"], "epochs")
    add("Hardware", "GPU devices", ", ".join(env["gpu_devices"]) or "CPU")
    add("Hardware", "Distributed replicas", env["replicas"])
    add("Hardware", "Compute policy", env["precision"])
    for key in ("python", "tensorflow", "keras"):
        if key in env:
            add("Software", key, str(env[key]).split()[0])
    add("Runtime", "CFD throughput assumption", CFD_CASES_PER_HOUR, "cases/hour (assumed)")
    add("Runtime", "Sequential-equivalent dataset-generation estimate", len(runner.bundle.df) / CFD_CASES_PER_HOUR, "hours (sequential estimate)")
    if runner.options.mode != "full":
        add("Analysis", "Result type", "SOFTWARE CHECK ONLY; not manuscript results")
    return pd.DataFrame(rows, columns=["Section", "Parameter", "Value", "Units or definition"])


def run_pipeline(options):
    options.validate()
    options.output=options.output.resolve()
    options.output.mkdir(parents=True,exist_ok=True)
    writer_preflight(options.output)
    start=time.perf_counter()
    if options.mode=='check':
        path=find_dataset(options.data)
        bundle,splits=make_real_check(path,options.sample_cases,options.strict_reference)
    else:
        path=find_dataset(options.data)
        bundle,splits=load_dataset(path,options.strict_reference)
    atomic_json(options.output/'_internal/dataset_audit.json',bundle.audit)
    # Freeze all protocol definitions before any training starts.
    protocols,metadata,assignments=create_protocols(bundle,splits,options.strict_reference)
    for name,parts in protocols.items():
        atomic_npz(options.output/'_internal/splits'/f'{name}.npz',**parts)
    from .tf_backend import TensorFlowBackend
    backend=TensorFlowBackend(options)
    atomic_json(options.output/'_internal/environment.json',backend.environment)
    runner=ExperimentRunner(bundle,backend,options)
    selected,tables=run_stages_a_d(runner,splits)
    selection,stage_e=run_stage_e(runner,selected,splits)
    tables.update(stage_e)
    runner.freeze_selection(selection)
    main_result,main_tables=run_main(runner,selection,splits)
    tables.update(main_tables)
    tables.update(run_structured(runner,selection,protocols,metadata,assignments))
    rows=[]
    for role,ii in splits.items():
        rows.extend((int(i),bundle.ids[i],role) for i in ii)
    tables['Data split']=pd.DataFrame(rows,columns=['row_index','simulation_id','split']).sort_values('row_index')
    tables['Run setup']=setup_table(runner,selection,main_result,splits)
    tables = paper_tables(tables)
    for name,frame in tables.items():
        atomic_csv(options.output/'tables'/(name.replace(' ','_')+'.csv'),align_table(name,frame))
    provenance = {
        'check': 'SOFTWARE CHECK ONLY: reduced training-data subset; not manuscript results.',
        'full': 'PRNN model results for thermal sterilization of canned solid-liquid foods.',
    }[options.mode]
    workbook=options.output/'revised_model_data.xlsx'
    manifest,verification=write_workbook(tables,workbook,provenance=provenance)
    stats=runner.reuse_statistics()
    report={'mode':options.mode,'software_check_only':options.mode=='check','backend':backend.environment,'workbook':str(workbook),'verification':verification,**stats,'wall_seconds':time.perf_counter()-start,'full_research_training_completed':options.mode=='full'}
    atomic_json(options.output/'_internal/execution_report.json',report)
    model_dir = options.output / "models"
    model_dir.mkdir(exist_ok=True)
    shutil.copy2(main_result.directory / "model.keras", model_dir / "prnn_lstm.keras")
    shutil.copy2(main_result.directory / "normalization.json", model_dir / "normalization.json")
    atomic_json(model_dir / "model_config.json", selection["selected_prnn_config"])
    print(f"Saved {verification['sheet_count']} worksheets; output validation passed.", flush=True)
    return report
