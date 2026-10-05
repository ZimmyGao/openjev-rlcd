"""Full-parameter AdamW for models whose fp32 weights and optimizer state do not fit on one GPU.

GPU side (bf16_matmul_weights): every nn.Linear is stored in bf16, and everything else (embeddings, a
tied output matrix, norms) stays in fp32. Under bf16 autocast this forward pass is the fp32 recipe's:
autocast rounds Linear weights and inputs to bf16 for every matmul anyway, while the residual stream,
the norms and the embeddings stay fp32. Each step's gradients are accumulated in fp32 on the GPU.

Host side (OffloadedAdamW): the fp32 master weights and the AdamW state live in pinned host memory and
are updated there with torch.optim.AdamW, i.e. the same update rule as the default recipe. The GPU
weights are then refreshed from the master weights. This is standard mixed-precision training
(Micikevicius et al. 2018) with the optimizer offloaded, as in ZeRO-Offload.

Memory per parameter: host 16 bytes (master, gradient, exp_avg, exp_avg_sq). GPU 4 bytes for the fp32
gradient buffer, plus 2 bytes per Linear weight or 4 bytes for everything else.
"""
import torch


def bf16_matmul_weights(model):
    """Store every nn.Linear (weight and bias) in bf16 and keep all other parameters in fp32. A Linear
    whose weight is tied to an nn.Embedding stays fp32. The model must then be run under bf16 autocast."""
    tied = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear) and id(mod.weight) not in tied:
            mod.to(torch.bfloat16)
    model.openjev_mixed_precision = True
    return model


class OffloadedAdamW:
    """torch.optim.AdamW with fp32 master weights and fp32 state, kept on the GPU for as many parameters as
    the free GPU memory allows (minus gpu_headroom_gb for activations and generation) and in host memory
    for the rest. Every parameter has its own AdamW state, so splitting the parameters between two
    optimizers and stepping the host part chunk by chunk gives the same update as a single AdamW."""

    def __init__(self, model, lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, gpu_headroom_gb=8.0,
                 chunk_floats=2**28):
        self.params = [p for p in model.parameters() if p.requires_grad]
        dev = self.params[0].device
        self.gpu_grad = [torch.zeros(p.shape, dtype=torch.float32, device=dev) for p in self.params]
        budget = (torch.cuda.mem_get_info(dev)[0] - gpu_headroom_gb * 2**30) if dev.type == "cuda" else 0
        self.master, on_gpu, used = [], [], 0
        for p in self.params:  # fp32 master + exp_avg + exp_avg_sq = 12 bytes per parameter
            here = used + 12 * p.numel() <= budget
            used += 12 * p.numel() if here else 0
            if here:
                m = p.detach().float().clone()
            else:
                m = torch.empty(p.shape, dtype=torch.float32, pin_memory=True)
                m.copy_(p.detach())
            self.master.append(m.requires_grad_(True))
            on_gpu.append(here)
        self.gpu_idx = [i for i, g in enumerate(on_gpu) if g]
        self.cpu_idx = [i for i, g in enumerate(on_gpu) if not g]
        kw = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.gpu_opt = torch.optim.AdamW([self.master[i] for i in self.gpu_idx], fused=True, **kw) if self.gpu_idx else None
        self.cpu_opt = None
        if self.cpu_idx:
            try:
                self.cpu_opt = torch.optim.AdamW([self.master[i] for i in self.cpu_idx], fused=True, **kw)
            except (RuntimeError, TypeError):
                self.cpu_opt = torch.optim.AdamW([self.master[i] for i in self.cpu_idx], foreach=False, **kw)
        # the host gradients are staged chunk by chunk through one pinned buffer
        self.chunks, cur, size = [], [], 0
        for i in self.cpu_idx:
            n = self.params[i].numel()
            if cur and size + n > chunk_floats:
                self.chunks.append(cur)
                cur, size = [], 0
            cur.append(i)
            size += n
        if cur:
            self.chunks.append(cur)
        need = max((sum(self.params[i].numel() for i in c) for c in self.chunks), default=0)
        self.staging = torch.empty(need, dtype=torch.float32, pin_memory=True) if need else None
        self.hooks = [p.register_post_accumulate_grad_hook(self._accumulate(b)) for p, b in zip(self.params, self.gpu_grad)]
        self.gpu_params_b = sum(self.params[i].numel() for i in self.gpu_idx) / 1e9
        self.cpu_params_b = sum(self.params[i].numel() for i in self.cpu_idx) / 1e9

    @staticmethod
    def _accumulate(buf):
        def hook(p):  # runs once per backward, after autograd has summed all contributions to p
            buf.add_(p.grad)
            p.grad = None
        return hook

    def zero_grad(self, set_to_none=True):
        for b in self.gpu_grad:
            b.zero_()

    def clip_grad_norm_(self, max_norm):
        """Same rule as torch.nn.utils.clip_grad_norm_ (coefficient max_norm / (norm + 1e-6), at most 1)."""
        total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(b) for b in self.gpu_grad]))
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        for b in self.gpu_grad:
            b.mul_(coef)
        return total

    def _refresh(self, i):
        b = self.gpu_grad[i]
        b.copy_(self.master[i], non_blocking=True)  # the fp32 gradient buffer doubles as the staging area
        self.params[i].copy_(b)                      # rounds to bf16 for the Linear weights
        b.zero_()

    @torch.no_grad()
    def step(self):
        if self.gpu_opt is not None:
            for i in self.gpu_idx:
                self.master[i].grad = self.gpu_grad[i]
            self.gpu_opt.step()
            for i in self.gpu_idx:
                self.master[i].grad = None
                self.params[i].copy_(self.master[i])
                self.gpu_grad[i].zero_()
        for chunk in self.chunks:
            off = 0
            for i in chunk:
                n = self.params[i].numel()
                g = self.staging[off:off + n].view(self.params[i].shape)
                g.copy_(self.gpu_grad[i], non_blocking=True)
                self.master[i].grad = g
                off += n
            torch.cuda.synchronize()
            self.cpu_opt.step()  # parameters without a gradient (other chunks) are skipped
            for i in chunk:
                self.master[i].grad = None
                self._refresh(i)
        torch.cuda.synchronize()
