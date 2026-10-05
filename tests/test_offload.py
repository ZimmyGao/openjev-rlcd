"""OffloadedAdamW = torch.optim.AdamW on fp32 weights, given the same gradients (accumulated over
several backward calls, clipped to a max norm), on a model with bf16 Linear layers, fp32 embeddings and
a tied embedding / output matrix (bf16_matmul_weights)."""
import pytest
import torch

from openjev_rlcd.offload import OffloadedAdamW, bf16_matmul_weights

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(11, 8)
        self.mid = torch.nn.Linear(8, 8)
        self.out = torch.nn.Linear(8, 11, bias=False)
        self.out.weight = self.emb.weight  # tied, as in Qwen3-0.6B and Phi-4-mini

    def forward(self, x):
        return self.out(torch.tanh(self.mid(self.emb(x))))


@cuda
@pytest.mark.parametrize("headroom_gb,chunk_floats,on_gpu", [
    (1e6, 2**28, 0),       # everything on the host, one chunk
    (1e6, 9, 0),           # everything on the host, one parameter per chunk
    (-1e6, 2**28, 3),      # everything on the GPU
])
def test_matches_fp32_adamw_given_the_same_gradients(headroom_gb, chunk_floats, on_gpu):
    torch.manual_seed(0)
    net = bf16_matmul_weights(Tiny()).cuda()
    params = list(net.parameters())
    assert len(params) == 3  # the tied matrix appears once
    assert [p.dtype for p in params] == [torch.float32, torch.bfloat16, torch.bfloat16]  # emb (tied), mid.w, mid.b
    ref = [p.detach().float().clone().requires_grad_(True) for p in params]
    opt = OffloadedAdamW(net, lr=1e-2, weight_decay=0.1, gpu_headroom_gb=headroom_gb, chunk_floats=chunk_floats)
    assert len(opt.gpu_idx) == on_gpu
    ref_opt = torch.optim.AdamW(ref, lr=1e-2, weight_decay=0.1)
    for _ in range(4):
        opt.zero_grad()
        x = torch.randint(0, 11, (6, 3), device="cuda")
        expected = [torch.zeros_like(r) for r in ref]
        for chunk in (x[:2], x[2:]):  # two backward calls per step, as in train.py (one per question)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net(chunk).float().pow(2).mean()
            for e, g in zip(expected, torch.autograd.grad(loss, params, retain_graph=True)):
                e.add_(g.float())
            loss.backward()
        assert all(p.grad is None for p in params)  # moved into the fp32 buffers by the hooks
        for b, e in zip(opt.gpu_grad, expected):
            torch.testing.assert_close(b, e, rtol=1e-3, atol=1e-6)
        for r, b in zip(ref, opt.gpu_grad):
            r.grad = b.clone()
        norm_ref = torch.nn.utils.clip_grad_norm_(ref, 0.05)
        norm = opt.clip_grad_norm_(0.05)
        torch.testing.assert_close(norm, norm_ref)
        for b, r in zip(opt.gpu_grad, ref):
            torch.testing.assert_close(b, r.grad)
        ref_opt.step()
        opt.step()
        for m, r, p in zip(opt.master, ref, params):
            torch.testing.assert_close(m.cpu(), r.detach().cpu(), rtol=1e-5, atol=1e-6)
            assert torch.equal(p.detach(), m.to(p.device, p.dtype))  # working copy = master, rounded for Linear
        assert all(float(b.abs().max()) == 0.0 for b in opt.gpu_grad)  # ready for the next step


@cuda
def test_split_between_gpu_and_host():
    """Budget for about the first parameter (the 11 x 8 tied matrix) on the GPU: the 8 x 8 matrix goes to
    the host (the tiny bias may fit in what the caching allocator leaves over)."""
    torch.manual_seed(0)
    net = bf16_matmul_weights(Tiny()).cuda()
    params = list(net.parameters())
    ref = [p.detach().float().clone().requires_grad_(True) for p in params]
    free = torch.cuda.mem_get_info()[0] - sum(4 * p.numel() for p in params)  # after the fp32 gradient buffers
    headroom = (free - 12 * params[0].numel()) / 2**30
    opt = OffloadedAdamW(net, lr=1e-2, gpu_headroom_gb=headroom, chunk_floats=8)
    assert 0 in opt.gpu_idx and 1 in opt.cpu_idx
    ref_opt = torch.optim.AdamW(ref, lr=1e-2, weight_decay=0.0)  # torch default is 0.01
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            net(torch.randint(0, 11, (4, 3), device="cuda")).float().pow(2).mean().backward()
        for r, b in zip(ref, opt.gpu_grad):
            r.grad = b.clone()
        ref_opt.step()
        opt.step()
        for m, r, p in zip(opt.master, ref, params):
            torch.testing.assert_close(m.cpu(), r.detach().cpu(), rtol=1e-5, atol=1e-6)
            assert torch.equal(p.detach(), m.to(p.device, p.dtype))
