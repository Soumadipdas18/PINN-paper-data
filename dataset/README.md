# Dataset

`dataset_all.csv.gz` contains the 10,526 CFD cases used by the study. The program reads the compressed file directly.

Each case provides four operating inputs, heating and cooling temperature histories, an F-value, and an ascorbic-acid retention target. Time is expressed in minutes and temperature in degrees Celsius. The required input-column names and accepted aliases are defined in `prnn/data.py`.

Use `python main.py --data PATH` to select an explicit CSV or gzip-compressed CSV. Keep simulation identifiers and row order intact when reproducing a run.
