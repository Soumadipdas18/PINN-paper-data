# Dataset: Physics-Regularized Recurrent Neural Network for Thermal Sterilization

## Description

This repository implements the physics regularized recurrent neural network (PRNN) used to model thermal sterilization of canned peas in water. The model maps four operating conditions to the slowest heating zone (SHZ) temperature history, a direct F value, a trajectory derived F value, and a bounded model based ascorbic acid retention index.

The workflow performs validation based model selection, final test evaluation, and structured holdout comparisons. It produces `outputs/revised_model_data.xlsx` with 23 worksheets containing the numerical results and data used for manuscript tables and figures. The repository exports numerical data only and does not generate figures.

The dataset contains 10,526 CFD simulations. The four operating inputs are come up time, heating time, cooling time, and retort temperature.

## Contents

The code is organized into four main parts:

1. **Stages A to D**
2. **Stage E**
3. **Full model run**
4. **Structured holdout evaluation**

Each part is described below.

## 1. Stages A to D

Stages A to D are used to select the PRNN settings using the validation set.

### Stage A: trajectory resolution

Stage A tests three trajectory resolutions:

- 100 heating points + 100 cooling points
- 200 heating points + 200 cooling points
- 300 heating points + 300 cooling points

The smallest resolution whose validation score is within 1% of the lowest Stage A validation score is selected. The selected resolution is then used in Stages B to D.

### Stage B: model capacity and dropout

Stage B tests:

- 128 or 192 recurrent and encoder units
- dropout of 0 or 0.1

The setting with the lowest validation score is selected.

### Stage C: supervised loss weights

Stage C tests the following standardized temperature, F value, and quality retention loss weights:

- 1:1:1
- 1:2:1
- 1:1:2
- 2:1:1

The setting with the lowest validation score is selected.

### Stage D: physics weight

Stage D tests:

- 0
- 0.1
- 0.5
- 1.0

The zero value is included as the physics ablation. The final PRNN setting is selected from the positive physics weights.

For Stages A to D, each candidate receives 70 epochs, batch size 64, and an initial learning rate of 0.001. Early stopping is not used. Validation is used to select the best model weights and to control learning rate reduction.

The code for these stages is mainly in:

```text
prnn/stages_a_d.py
prnn/experiments.py
prnn/losses.py
prnn/tf_backend.py
```

## 2. Stage E

Stage E compares the selected PRNN with three other sequence or data driven models:

- `PRNN_LSTM`
- `data_LSTM`
- `GRU`
- `time_conditioned_MLP`

The comparison uses seeds 17 and 42. Each candidate receives the same training budget.

Stage E uses the settings selected from Stages A to D. The PRNN best epochs from the two seeds are also used to determine the fixed number of epochs for the final model run.

The main Stage E code is in:

```text
prnn/stage_e.py
```

The common training and prediction functions are in:

```text
prnn/experiments.py
prnn/tf_backend.py
```

## 3. Full model run

After model selection is complete, the selected PRNN is trained for the fixed number of epochs obtained from Stage E.

The full model run then evaluates the untouched test set and saves the final numerical results. It also saves the trained model, normalization data, model settings, predictions, temperature trajectories, error data, attention data, runtime data, and design tradeoff data.

The main code for this part is:

```text
prnn/main_run.py
```

The final model files are written to:

```text
outputs/models/
```

The main files are:

```text
prnn_lstm.keras
model_config.json
normalization.json
```

## 4. Structured holdout evaluation

The selected PRNN is also tested under four structured holdout protocols:

1. random interpolation control
2. withheld retort temperature band
3. withheld high temperature and short heating time corner
4. withheld operating domain boundary shell

Each protocol is trained and evaluated with seeds 17 and 42.

For every structured holdout, the holdout cases are removed before model training and normalization. The random interpolation control is trained separately from the final model run.

The structured holdout code is in:

```text
prnn/structured_holdouts.py
```

The output includes the protocol definitions, individual run results, summary statistics, holdout predictions, holdout assignments, and error as a function of distance from the training data.

## Dataset

The input dataset is stored in the `dataset/` folder.

The code can read either:

```text
dataset/dataset_all.csv
```

or:

```text
dataset/dataset_all.csv.gz
```

The operating ranges are:

| Variable | Range |
|---|---:|
| Come up time | 1 to 5 min |
| Heating time | 5 to 20 min |
| Cooling time | 1 to 5 min |
| Retort temperature | 120 to 140 °C |

The deterministic main split contains:

| Split | Cases |
|---|---:|
| Training | 7,368 |
| Validation | 1,579 |
| Test | 1,579 |
| Total | 10,526 |

The validation set is used during model selection. The test set is kept untouched until the final model settings are fixed.

## Main repository files

`main.py`  
Main command line entry point. It starts either the complete study or the reduced software check.

`prnn/config.py`  
Contains study constants, search values, seeds, runtime settings, and command options.

`prnn/data.py`  
Reads the CFD dataset, checks the data, creates the deterministic split, prepares temperature trajectories, and calculates normalization values.

`prnn/model.py`  
Contains the neural network definitions used by the PRNN and comparison models.

`prnn/physics.py`  
Calculates the trajectory based F value from the predicted temperature history.

`prnn/losses.py`  
Contains the training losses and validation score calculations.

`prnn/tf_backend.py`  
Contains TensorFlow training, validation, prediction, GPU use, and mixed precision handling.

`prnn/experiments.py`  
Controls individual model fits, saved progress, and reuse of completed compatible runs.

`prnn/stages_a_d.py`  
Runs Stages A to D.

`prnn/stage_e.py`  
Runs the Stage E model comparison.

`prnn/main_run.py`  
Runs final model training and final test evaluation.

`prnn/structured_holdouts.py`  
Runs the four structured holdout evaluations.

`prnn/reporting.py`  
Prepares the numerical tables used in the output workbook.

`prnn/workbook.py`  
Writes and checks the 23 sheet Excel workbook.

`prnn/resources/`  
Contains the workbook template and workbook schema.

`requirements.txt`  
Lists the Python packages required to run the study.

## Installation

Python 3.11 or 3.12 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Software check

A short real data check can be run before starting the complete study:

```bash
python main.py \
  --data dataset/dataset_all.csv \
  --mode check \
  --sample-cases 256 \
  --check-epochs 3 \
  --device gpu \
  --precision mixed_float16
```

The check run uses a small real data subset, small model sizes, short trajectories, and three training epochs. It checks the complete software path, including training and Excel export.

Check results are written to:

```text
outputs_check/
```

These values are software check results and are not manuscript results.

## Full study run

Run the complete study with:

```bash
python main.py \
  --data dataset/dataset_all.csv \
  --mode full \
  --device gpu \
  --precision mixed_float16
```

The full run uses all 10,526 CFD cases and performs Stages A to D, Stage E, the final model run, and the structured holdout evaluations.

Completed fits are saved during the run. If the computer or GPU session stops, run the same command again. Compatible completed fits are reused automatically.

Do not use `--no-resume` when continuing an interrupted run.

## Output data

The complete study writes results to:

```text
outputs/
```

The main output structure is:

```text
outputs/
├── revised_model_data.xlsx
├── tables/
├── models/
└── _internal/
```

`revised_model_data.xlsx` is the main numerical result workbook.

`tables/` contains a CSV copy of each workbook sheet.

`models/` contains the trained final model and its saved settings.

`_internal/` contains saved progress, data split information, run records, and cache files used to continue interrupted runs.

## Excel workbook

The workbook contains 23 worksheets:

| Worksheet | Contents |
|---|---|
| Run setup | Dataset counts, training settings, selected model settings, hardware, and runtime information |
| Stages A-D runs | Individual validation results from Stages A to D |
| Resolution sensitivity | F value sensitivity to trajectory resolution |
| Stage E runs | Individual Stage E model and seed results |
| Stage E summary | Mean and standard deviation for the Stage E models |
| Main paper table | Main final model results used for reporting |
| Main test metrics | Detailed final test metrics |
| Main training history | Epoch level training and validation data for the final model |
| Main trajectories | Selected true and predicted SHZ temperature trajectories |
| Trajectory metadata | Operating conditions and case information for the selected trajectories |
| Main predictions | Case level final test predictions |
| Error trends | Numerical data for error trends across operating variables |
| Design tradeoff | Predicted process conditions and quality results used for the design analysis |
| Attention profile | Numerical temporal attention data |
| Runtime | Training and inference runtime data |
| Holdout protocols | Definitions and sample counts for the structured holdouts |
| Holdout runs | Individual structured holdout results for each seed |
| Holdout summary | Mean and standard deviation for each structured holdout |
| Distance bins runs | Error results grouped by distance from the training data for each run |
| Distance bins summary | Summary of the distance bin results |
| Holdout predictions | Case level predictions for structured holdout cases |
| Holdout assignments | Case membership for each structured holdout protocol |
| Data split | Main training, validation, and test assignments |

The same tables are also saved individually as CSV files in:

```text
outputs/tables/
```

The repository produces numerical data only. It does not create PNG, PDF, or other figure files.
