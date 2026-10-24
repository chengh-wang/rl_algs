"""Regression for the external loader's stateful, batch-indexed clip selection."""
import pytest
import torch


def test_velocity_expert_iterator_covers_all_clips_and_preserves_upstream_cycle():
    pytest.importorskip('mujoco')
    from examples.mjlab_amp.model_task import BaselineVelocityTask
    class Loader:
        def feed_forward_generator(self,batches,count):
            for batch in range(batches):
                value=torch.full((count,1),batch%18)
                yield value,value+1
    task=BaselineVelocityTask.__new__(BaselineVelocityTask)
    task.expert=Loader(); task._expert_iterator=None; task._expert_batch_size=None; task._expert_batches=20
    observed=[int(task.sample_expert_features(4)[0][0,0]) for _ in range(40)]
    assert observed==list(range(18))+[0,1]+list(range(18))+[0,1]
