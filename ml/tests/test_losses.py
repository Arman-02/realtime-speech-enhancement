from __future__ import annotations

import torch

from ans.losses import EnhancementLoss, LossConfig, MultiResSTFTLoss, si_sdr


def test_loss_zero_for_perfect_estimate():
    torch.manual_seed(0)
    ref = torch.randn(2, 8000) * 0.1
    loss_fn = EnhancementLoss(LossConfig(mrstft_weight=1.0, sisdr_weight=0.1))
    loss, terms = loss_fn(ref.clone(), ref)
    assert terms["spec"] < 1e-6 and terms["mrstft"] < 1e-3
    assert set(terms) == {"spec", "mrstft", "sisdr"}
    worse, _ = loss_fn(ref + 0.05 * torch.randn_like(ref), ref)
    assert worse > loss


def test_loss_gradients_finite_with_silence_and_mask():
    est = torch.zeros(2, 4000, requires_grad=True)
    ref = torch.randn(2, 4000) * 0.1
    mask = torch.ones(2, 4000)
    mask[1, 2000:] = 0
    loss, _ = EnhancementLoss(LossConfig(mrstft_weight=0.5, sisdr_weight=0.1))(est, ref, mask)
    loss.backward()
    assert torch.isfinite(est.grad).all()


def test_si_sdr_torch():
    x = torch.randn(3, 4000)
    assert (si_sdr(x, x) > 60).all()
    assert MultiResSTFTLoss()(x, x) < 1e-3
