"""Atomic outputs and content-based identifiers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _finite_json(value):
    if isinstance(value, dict):
        return {k: _finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as handle:
            json.dump(_finite_json(payload),handle,indent=2,default=_json_default,allow_nan=False)
            handle.write('\n')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    try:
        frame.to_csv(tmp,index=False)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.tmp.npz')
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def identity_hash(payload) -> str:
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),default=_json_default,allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint() -> str:
    digest=hashlib.sha256()
    for name in ('config.py','data.py','model.py','losses.py','physics.py','metrics.py','io.py','tf_backend.py','experiments.py'):
        p=Path(__file__).parent/name
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()
