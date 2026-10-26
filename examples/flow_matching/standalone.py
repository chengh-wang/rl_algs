"""Train and score synthetic motion windows to exercise the standalone API."""
import argparse
import json
import torch
from mbdac.flow_matching import (
    FlowTrainingConfig, FlowMotionPrior, FlowMotionScorer, train_flow_prior,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Synthetic tensors only: this example is an interface smoke, not motion data.
    windows = torch.randn(8, 4, 6, generator=torch.Generator().manual_seed(7))
    schema = {'layout': 'synthetic_v1', 'sample_dt': 0.02}
    checkpoint = train_flow_prior(
        windows, args.output,
        FlowTrainingConfig(epochs=2, batch_size=4, num_noise_samples=2,
                           d_model=16, nhead=2, num_layers=1),
        feature_schema=schema,
    )
    prior = FlowMotionPrior.load_checkpoint(
        checkpoint, expected_feature_dim=6, expected_window_size=4,
        expected_feature_schema=schema,
    )
    scorer = FlowMotionScorer(prior, times=(0.2, 0.5, 0.8))
    noise = torch.randn(3, 2, 4, 6, generator=torch.Generator().manual_seed(8))
    result = scorer.score(windows[:2], noise=noise, update_normalizer=True)
    print(json.dumps({'checkpoint': str(checkpoint),
                      'reward': result.reward.tolist(),
                      'raw_error': result.raw_error.tolist(),
                      'normalizer_count': scorer.normalizer.count.tolist()}))


if __name__ == '__main__':
    main()
