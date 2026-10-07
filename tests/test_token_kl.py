"""The pathwise KL anchor (train._token_kl): KL(pi || pi_ref) per position, gradient through pi only."""
import torch

from openjev_rlcd.train import _token_kl


def test_token_kl_value_and_gradient():
    torch.manual_seed(0)
    head = torch.nn.Linear(6, 9, bias=False)
    ref_head = torch.nn.Linear(6, 9, bias=False).requires_grad_(False)
    h, h_ref = torch.randn(5, 6), torch.randn(5, 6)
    kl = _token_kl(head, ref_head, h, h_ref)
    p, q = head(h).softmax(-1), ref_head(h_ref).softmax(-1)
    torch.testing.assert_close(kl, (p * (p.log() - q.log())).sum(-1))
    assert (kl >= 0).all()
    kl.sum().backward()
    assert head.weight.grad is not None and ref_head.weight.grad is None
    assert _token_kl(head, head, h, h).abs().max() < 1e-6  # zero against itself
