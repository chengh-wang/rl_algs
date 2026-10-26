"""Train a Flow motion prior from task-independent feature-window arrays."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import torch

from .training import FlowTrainingConfig, train_flow_prior


def _read_windows(path):
    with np.load(path, allow_pickle=False) as arrays:
        if 'features' not in arrays:
            raise ValueError('Window NPZ must contain features [N,W,F]')
        return torch.from_numpy(arrays['features'].copy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', required=True, help='NPZ with features [N,W,F]')
    parser.add_argument('--output', required=True)
    parser.add_argument('--feature-schema', help='Optional JSON documenting your feature layout')
    parser.add_argument('--validation-windows', help='Separate held-out feature-window NPZ')
    parser.add_argument('--device', default='cpu')
    defaults = FlowTrainingConfig()
    for item in fields(defaults):
        value = getattr(defaults, item.name)
        parser.add_argument('--'+item.name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args(argv)
    config = FlowTrainingConfig(**{f.name: getattr(args, f.name) for f in fields(defaults)})
    schema = None if args.feature_schema is None else json.loads(Path(args.feature_schema).read_text())
    validation = None if args.validation_windows is None else _read_windows(args.validation_windows)
    target = train_flow_prior(_read_windows(args.windows), args.output, config,
        feature_schema=schema, device=args.device, validation_windows=validation)
    print(json.dumps({'checkpoint': str(target)}))


if __name__ == '__main__':
    main()
