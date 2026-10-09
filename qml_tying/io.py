import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy
import sklearn
import hashlib


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if hasattr(x, '__dataclass_fields__'): return asdict(x)
    raise TypeError(type(x).__name__)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, default=jsonable, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def record_id(*items):
    return hashlib.sha256(json.dumps(items,sort_keys=True,default=jsonable).encode()).hexdigest()[:32]


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, platform=platform.platform())


def arguments(description):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default='configs/pilot.json')
    p.add_argument('--out', default='results/pilot')
    p.add_argument('--limit', type=int, help='maximum NEW jobs, for a smoke run')
    p.add_argument('--no-aggregate', action='store_true', help='cluster driver aggregates once at the end')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return args, cfg
