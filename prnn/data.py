"""Dataset parsing, trajectory resampling, and training-only normalization."""
from __future__ import annotations
import ast
import gzip
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import *
from .io import sha256_file, sha256_array, atomic_json

def resample_curve(curve: np.ndarray, n_new: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, curve.size, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, n_new, dtype=np.float32)
    return np.interp(new_x, old_x, curve).astype(np.float32)

def resample_all_curves(
    heat_cells: Sequence, cool_cells: Sequence, n_heat: int, n_cool: int
) -> np.ndarray:
    """Resample each process stage, sharing its heating-to-cooling endpoint."""
    n = len(heat_cells)
    y = np.empty((n, n_heat + n_cool, 1), dtype=np.float32)
    for i, (h_cell, c_cell) in enumerate(zip(heat_cells, cool_cells)):
        heat_curve = parse_curve_cell(h_cell)
        cool_curve = parse_curve_cell(c_cell)

        # Cooling local time tau=0 is the instant at which heating ends.  The
        # first stored CFD cooling value is at tau=1 s, so use the final
        # heating temperature as the missing tau=0 cooling value.
        cool_curve_aligned = np.concatenate((heat_curve[-1:], cool_curve))

        y[i, :n_heat, 0] = resample_curve(heat_curve, n_heat)
        y[i, n_heat:, 0] = resample_curve(cool_curve_aligned, n_cool)
    return y

def make_stratification_labels(x: np.ndarray, logf: np.ndarray) -> np.ndarray:
    """Joint coarse bins improve coverage of temperature, duration, and F."""
    pieces = []
    for values, q in [(x[:, 3], 5), (x[:, 1], 4), (logf.reshape(-1), 4)]:
        pieces.append(
            np.asarray(pd.qcut(values, q=q, labels=False, duplicates="drop")).astype(str)
        )
    labels = np.char.add(np.char.add(pieces[0], "_"), pieces[1])
    labels = np.char.add(np.char.add(labels, "_"), pieces[2])
    counts = pd.Series(labels).value_counts()
    rare = set(counts[counts < 4].index)
    if rare:
        labels = np.asarray(["rare" if v in rare else v for v in labels])
    return labels

def safe_std(a: np.ndarray) -> np.ndarray:
    s = np.asarray(a).std(axis=0)
    return np.where(s < 1.0e-8, 1.0, s)

def compute_normalization(
    x: np.ndarray,
    y_temp: np.ndarray,
    logf: np.ndarray,
    retention: np.ndarray,
    train_idx: np.ndarray,
    n_heat: int,
    n_cool: int,
) -> Normalization:
    xt = x[train_idx]
    x_mean = xt.mean(axis=0)
    x_std = safe_std(xt)
    yt = y_temp[train_idx]

    # Estimate sequence-feature moments on training data only.
    walls, dts = [], []
    tau_h = np.linspace(0.0, 1.0, n_heat, dtype=np.float32)[None, :]
    for row in xt:
        cut, hold, cool, tret = row
        heat_duration = cut + hold
        t_heat = tau_h.reshape(-1) * heat_duration
        wall_heat = INITIAL_TEMPERATURE_C + (tret - INITIAL_TEMPERATURE_C) * np.minimum(
            t_heat / max(cut, 1.0e-6), 1.0
        )
        walls.append(wall_heat)
        walls.append(np.full(n_cool, INITIAL_TEMPERATURE_C, dtype=np.float32))
        dts.append(np.full(n_heat, heat_duration / max(n_heat - 1, 1), dtype=np.float32))
        dts.append(np.full(n_cool, cool / max(n_cool - 1, 1), dtype=np.float32))
    wall_values = np.concatenate(walls)
    dt_values = np.concatenate(dts)

    return Normalization(
        x_mean=x_mean.astype(float).tolist(),
        x_std=x_std.astype(float).tolist(),
        temperature_mean=float(yt.mean()),
        temperature_std=float(max(yt.std(), 1.0e-6)),
        logF_mean=float(logf[train_idx].mean()),
        logF_std=float(max(logf[train_idx].std(), 1.0e-6)),
        retention_mean=float(retention[train_idx].mean()),
        retention_std=float(max(retention[train_idx].std(), 1.0e-6)),
        max_total_time_min=float(np.max(xt[:, 0] + xt[:, 1] + xt[:, 2])),
        wall_temperature_mean=float(wall_values.mean()),
        wall_temperature_std=float(max(wall_values.std(), 1.0e-6)),
        dt_mean=float(dt_values.mean()),
        dt_std=float(max(dt_values.std(), 1.0e-6)),
    )

def build_sequence_features(
    x: np.ndarray,
    n_heat: int,
    n_cool: int,
    norm: Normalization,
) -> np.ndarray:
    """Physical time, stage, wall-temperature, step-size, and transition cues."""
    n = len(x)
    features = np.empty((n, n_heat + n_cool, 6), dtype=np.float32)
    tau_h = np.linspace(0.0, 1.0, n_heat, dtype=np.float32)
    tau_c = np.linspace(0.0, 1.0, n_cool, dtype=np.float32)
    for i, (cut, hold, cool, tret) in enumerate(x):
        heat_duration = float(cut + hold)
        t_h = tau_h * heat_duration
        t_c = tau_c * float(cool)
        wall_h = INITIAL_TEMPERATURE_C + (float(tret) - INITIAL_TEMPERATURE_C) * np.minimum(
            t_h / max(float(cut), 1.0e-6), 1.0
        )
        wall_c = np.full(n_cool, INITIAL_TEMPERATURE_C, dtype=np.float32)
        local_tau = np.concatenate([tau_h, tau_c])
        elapsed = np.concatenate([t_h, heat_duration + t_c]) / norm.max_total_time_min
        stage = np.concatenate(
            [np.zeros(n_heat, dtype=np.float32), np.ones(n_cool, dtype=np.float32)]
        )
        wall = (np.concatenate([wall_h, wall_c]) - norm.wall_temperature_mean) / norm.wall_temperature_std
        dt = np.concatenate(
            [
                np.full(n_heat, heat_duration / max(n_heat - 1, 1), dtype=np.float32),
                np.full(n_cool, float(cool) / max(n_cool - 1, 1), dtype=np.float32),
            ]
        )
        dt = (dt - norm.dt_mean) / norm.dt_std
        transition = np.zeros(n_heat + n_cool, dtype=np.float32)
        transition[n_heat] = 1.0
        features[i] = np.stack([local_tau, elapsed, stage, wall, dt, transition], axis=1)
    return features


def find_dataset(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        p=Path(explicit_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p
    folder=ROOT/'dataset'
    matches=[p for p in folder.iterdir() if p.is_file() and p.name.lower() in {'dataset_all.csv','dataset_all.csv.gz'}] if folder.exists() else []
    if len(matches)==1:
        return matches[0]
    if len(matches)>1:
        raise ValueError('Both compressed/uncompressed or duplicate datasets are present; provide --data explicitly.')
    raise FileNotFoundError(
        'Dataset not found. Place it at dataset/dataset_all.csv (or dataset_all.csv.gz), '
        'or pass --data PATH. '
        'Raw temperature histories are required.'
    )


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df.columns=[str(c).strip() for c in df.columns]
    aliases={
        'Simulation':['Simulation','simulation','case','case_id'],
        'F_SHZ':['F_SHZ','F','F_value','F-value'],
        'quality':['quality','qual','Q','retention'],
        'temp_heat_arr':['temp_heat_arr','temperature_heating'],
        'temp_cool_arr':['temp_cool_arr','temperature_cooling'],
        'CUT':['CUT','cut','come_up_time'],
        'time_heat':['time_heat','heating_time'],
        'time_cool':['time_cool','cooling_time'],
        'temp_retort':['temp_retort','retort_temperature'],
    }
    rename={}
    for target,choices in aliases.items():
        hits=[c for c in choices if c in df.columns]
        if len(hits)!=1:
            raise ValueError(f"Expected one column for {target}; found {hits}. Columns: {list(df.columns)}")
        rename[hits[0]]=target
    return df.rename(columns=rename)


def parse_curve_cell(value) -> np.ndarray:
    """Parse the entire cell; never accept a partially parsed numeric prefix."""
    if isinstance(value,(list,tuple,np.ndarray)):
        raw=value
    elif isinstance(value,str):
        text=value.strip()
        try:
            raw=ast.literal_eval(text)
        except (ValueError,SyntaxError) as exc:
            raise ValueError(f"Invalid trajectory list: {text[:80]}") from exc
    else:
        raise ValueError('Temperature-history cells must contain a list of numeric temperatures.')
    arr=np.asarray(raw,dtype=np.float32)
    if arr.ndim!=1 or len(arr)<2 or not np.isfinite(arr).all():
        raise ValueError('A temperature history must be a one-dimensional list of at least two finite values.')
    return arr


def validate_partitions(splits: dict[str,np.ndarray], n: int) -> None:
    values=[]
    for name in ('train','validation','test'):
        idx=np.asarray(splits[name])
        if len(idx)==0 or idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer):
            raise ValueError(f'Invalid or empty {name} split.')
        values.append(idx)
    merged=np.concatenate(values)
    if len(merged)!=n or not np.array_equal(np.sort(merged),np.arange(n)):
        raise ValueError('Data partitions overlap, omit rows, or contain invalid indices.')


def _split(indices,x,logf,fraction,seed,strict=True):
    labels=make_stratification_labels(x[indices],logf[indices])
    try:
        return train_test_split(indices,test_size=fraction,random_state=seed,stratify=labels)
    except ValueError as exc:
        if strict:
            raise ValueError('Joint stratification failed; no unreported fallback is permitted for research runs.') from exc
        # Small software-check subsets may not populate every joint stratum.
        # This fallback is only enabled for explicitly labelled software-check data.
        return train_test_split(indices,test_size=fraction,random_state=seed)


def create_splits(x,logf,strict=True):
    # Use the full-dataset stratum labels for both random partitions.
    indices=np.arange(len(x),dtype=np.int64)
    labels=make_stratification_labels(x,logf)
    try:
        dev,test=train_test_split(indices,test_size=TEST_FRACTION,random_state=SPLIT_SEED,stratify=labels)
        train,val=train_test_split(dev,test_size=VALIDATION_FRACTION/(1-TEST_FRACTION),random_state=SPLIT_SEED+1,stratify=labels[dev])
    except ValueError as exc:
        if strict:
            raise ValueError('Random split stratification failed. Inspect the dataset instead of silently changing the split.') from exc
        dev,test=train_test_split(indices,test_size=TEST_FRACTION,random_state=SPLIT_SEED)
        train,val=train_test_split(dev,test_size=VALIDATION_FRACTION/(1-TEST_FRACTION),random_state=SPLIT_SEED+1)
    result={'train':np.sort(train),'validation':np.sort(val),'test':np.sort(test)}
    validate_partitions(result,len(x))
    return result


@dataclass
class Prepared:
    x_scaled: np.ndarray
    seq: np.ndarray
    temperature: np.ndarray
    norm: Normalization
    heat_minutes: np.ndarray
    cool_minutes: np.ndarray
    logf: np.ndarray
    retention: np.ndarray

    def arrays(self,indices):
        return tuple(a[indices] for a in (self.x_scaled,self.seq,self.heat_minutes,self.cool_minutes,self.temperature,self.logf,self.retention))


class DatasetBundle:
    """Validated raw data. Temperature resampling is cached, normalization is split-specific."""
    def __init__(self,df: pd.DataFrame, dataset_hash: str):
        self.df=canonicalize_columns(df)
        self.dataset_hash=dataset_hash
        self.software_check=False
        scalar=['F_SHZ','quality']+INPUT_NAMES
        for c in scalar:
            self.df[c]=pd.to_numeric(self.df[c],errors='coerce')
        self.audit={
            'rows':len(self.df),
            'dataset_sha256':dataset_hash,
            'missing_or_nonfinite_scalar_values':int((~np.isfinite(self.df[scalar].to_numpy(dtype=float))).sum()),
            'missing_simulation_ids':int(self.df['Simulation'].isna().sum()+(self.df['Simulation'].astype(str).str.strip()=='').sum()),
            'duplicate_simulation_ids':int(self.df['Simulation'].duplicated().sum()),
            'duplicate_input_rows':int(self.df[INPUT_NAMES].duplicated().sum()),
            'nonpositive_F':int((self.df['F_SHZ']<=0).sum()),
            'retention_outside_0_1':int(((self.df['quality']<0)|(self.df['quality']>1)).sum()),
        }
        for k in list(self.audit)[3:]:
            if self.audit[k]:
                raise ValueError(f'Dataset audit failed: {self.audit}')
        self.x=self.df[INPUT_NAMES].to_numpy(np.float32)
        if np.any(self.x < INPUT_LOWER_BOUNDS-1e-5) or np.any(self.x > INPUT_UPPER_BOUNDS+1e-5):
            raise ValueError('Operating inputs are outside the prescribed domain. Do not silently change the holdout definitions.')
        self.ids=self.df['Simulation'].to_numpy()
        self.logf=np.log10(self.df['F_SHZ'].to_numpy(np.float64)).astype(np.float32).reshape(-1,1)
        self.q=self.df['quality'].to_numpy(np.float32).reshape(-1,1)
        self.heat=[parse_curve_cell(c) for c in self.df['temp_heat_arr']]
        self.cool=[parse_curve_cell(c) for c in self.df['temp_cool_arr']]
        self.hm=(self.x[:,0]+self.x[:,1]).reshape(-1,1)
        self.cm=self.x[:,2].reshape(-1,1)
        self._temperatures={}
        for i,name in enumerate(INPUT_NAMES):
            self.audit[name+'_min']=float(self.x[:,i].min())
            self.audit[name+'_max']=float(self.x[:,i].max())
        self.audit['retention_spatial_domain']='Not independently verifiable from these scalar labels; confirm pea-only volume averaging in the CFD export.'

    def temperature(self,nh,nc):
        key=(nh,nc)
        if key not in self._temperatures:
            self._temperatures[key]=resample_all_curves(self.heat,self.cool,nh,nc)
        return self._temperatures[key]

    def prepare(self,config,splits):
        t=self.temperature(config.n_heat,config.n_cool)
        norm=compute_normalization(self.x,t,self.logf,self.q,splits['train'],config.n_heat,config.n_cool)
        xs=((self.x-np.asarray(norm.x_mean))/np.asarray(norm.x_std)).astype(np.float32)
        seq=build_sequence_features(self.x,config.n_heat,config.n_cool,norm)
        return Prepared(xs,seq,t,norm,self.hm,self.cm,self.logf,self.q)


def load_dataset(path: Path,strict_reference=True):
    digest=dataset_sha256(path)
    if strict_reference and digest != EXPECTED_DATASET_SHA256:
        raise ValueError(f'Dataset identity differs from the configured study input. Use --allow-different-dataset only for an intentional dataset change.')
    bundle=DatasetBundle(pd.read_csv(path),digest)
    if len(bundle.df)!=EXPECTED_DATASET_ROWS:
        raise ValueError(f'Expected {EXPECTED_DATASET_ROWS} cases, got {len(bundle.df)}.')
    splits=create_splits(bundle.x,bundle.logf)
    if strict_reference:
        for role,indices in splits.items():
            if sha256_array(indices)!=EXPECTED_SPLIT_SHA256[role]:
                raise ValueError(f'{role} partition differs from the configured split. Check library versions, row ordering, and dataset values.')
    return bundle,splits


def dataset_sha256(path: Path) -> str:
    """Hash uncompressed CSV bytes so gzip storage does not change provenance."""
    path = Path(path)
    if path.suffix.lower() != '.gz':
        return sha256_file(path)
    digest = hashlib.sha256()
    with gzip.open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_real_check(path: Path, sample_cases: int = 256, strict_reference: bool = True):
    """Use only full-data TRAINING rows for a reduced end-to-end software check.

    Full-dataset validation/hash/split checks run first. The full 1,579-case
    final test partition is never used by this diagnostic. Sampling and the
    deliberately small check splits are recorded, never labelled as research
    results. Sparse check partitions may use a documented unstratified split.
    """
    full, full_splits = load_dataset(path, strict_reference)
    if sample_cases > len(full_splits['train']):
        raise ValueError('Requested check subset exceeds the full-data training partition.')
    rng = np.random.default_rng(42)
    chosen = np.sort(rng.choice(full_splits['train'], size=sample_cases, replace=False))
    subset_hash = hashlib.sha256(
        (full.dataset_hash + ':' + sha256_array(chosen)).encode('ascii')
    ).hexdigest()
    subset = DatasetBundle(full.df.iloc[chosen].reset_index(drop=True), subset_hash)
    subset.software_check = True
    subset.audit.update({
        'software_check_only': True,
        'source_dataset_sha256': full.dataset_hash,
        'source_dataset_rows': len(full.df),
        'sampling_rule': '256 by default, sampled without replacement from the full-data training partition only; seed 42',
        'source_row_indices': chosen.tolist(),
        'original_final_test_cases_used': 0,
        'small_partition_stratification_fallback_allowed': True,
    })
    check_splits = create_splits(subset.x, subset.logf, strict=False)
    return subset, check_splits
