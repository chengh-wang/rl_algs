"""Optional GPU precision checks against the unchanged Holosoma network math."""
import torch
import pytest
from mbdac.modules.holosoma import HolosomaActor,HolosomaCritic
from mbdac.algorithms.mbdac_fast_sac import FAST_SAC_DEFAULTS


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is unavailable')
def test_bf16_actor_q_loss_and_gradients_match_source_autocast():
    torch.manual_seed(83)
    options={**FAST_SAC_DEFAULTS,'precision':'bf16','actor_hidden_dim':64,'critic_hidden_dim':64,'num_atoms':51}
    actor=HolosomaActor(7,3,2,1,options).cuda();critic=HolosomaCritic(11,2,options).cuda()
    h=torch.randn(32,1,7,device='cuda');g=torch.randn(32,3,device='cuda');x=torch.randn(32,11,device='cuda')
    rng=torch.cuda.get_rng_state()
    a,logp=actor.sample(h,g);actual=(.001*logp-critic.value(x,a)).mean()
    torch.cuda.set_rng_state(rng)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ar,lp=actor.net.get_actions_and_log_probs(actor.inputs(h,g))
        q=critic.net.get_value(torch.softmax(critic.net(critic.norm(x),ar),-1)).mean(0)
        expected=(.001*lp-q).mean()
    assert torch.equal(a,ar) and torch.equal(logp,lp) and torch.equal(actual,expected)
    ga=torch.autograd.grad(actual,tuple(actor.parameters()),retain_graph=True)
    ge=torch.autograd.grad(expected,tuple(actor.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(ga,ge))
    rewards=torch.randn(32,device='cuda');bootstrap=torch.ones_like(rewards);bootstrap[:8]=0
    target=critic.projection(x,a.detach(),rewards,bootstrap,.97)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        reference=critic.net.projection(critic.norm(x),a.detach(),rewards,bootstrap,torch.full_like(rewards,.97))
        expected_loss=-(reference*torch.log_softmax(critic.net(critic.norm(x),a.detach()),-1)).sum(-1).mean(1).sum()
    logits=critic(x,a.detach())
    actual_loss=-(target*torch.log_softmax(logits.float(),-1)).sum(-1).mean(1).sum()
    # C51 uses CUDA scatter_add atomics; repeated identical projections can
    # differ at the last fp32 bit from summation ordering.
    print({'c51_target_max_abs_error':float((target-reference).abs().max().detach()),
           'c51_loss_abs_error':float((actual_loss-expected_loss).abs().detach())})
    torch.testing.assert_close(target,reference,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(actual_loss,expected_loss,rtol=2e-6,atol=2e-7)
