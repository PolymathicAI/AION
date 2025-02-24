import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko
from jaxtyping import Float
from torchdiffeq import odeint, odeint_adjoint


def replace_linear(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module)

        if isinstance(module, zuko.nn.Linear):
            new_linear = torch.nn.Linear(
                module.in_features, module.out_features, bias=module.bias is not None
            )
            new_linear.weight = module.weight
            if module.bias is not None:
                new_linear.bias = module.bias
            setattr(model, n, new_linear)


class Flow(nn.Module):
    flow: zuko.flows.LazyDistribution

    def __init__(self, data_dim: int, context_dim: int):
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim

    def condition(
        self,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
    ):
        if self.context_dim == 0 and context is None:
            return self.flow()

        else:
            if cfg_w == 0.0:
                assert null_context is not None
                return self.flow(null_context)
            elif cfg_w == 1.0:
                return self.flow(context)
            else:
                context = (1 - cfg_w) * null_context + cfg_w * context
                return self.flow(context)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).log_prob(
            x
        )

    def sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
        **kwargs,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).sample(
            (num_samples,)
        )

    def step(self, batch, batch_idx=None, **kwargs):
        if len(batch) == 2:
            x, context = batch
            return -self.log_prob(x, context=context).mean()
        # unconditional
        return -self.log_prob(
            batch
        ).mean()  # minimize negative log likelihood over all elements


class NSF(Flow):
    def __init__(
        self,
        data_dim: int,
        context_dim: int = 0,
        layers: int = 4,
        bins: int = 8,
        hidden_dims: list[int] = [64, 64],
        activation: str = "GELU",
        rescale: float = 1.0,
    ):
        super().__init__(data_dim, context_dim)
        self.flow = zuko.flows.NSF(
            features=data_dim,
            context=context_dim,
            bins=bins,
            transforms=layers,
            hidden_features=hidden_dims,
            activation=getattr(torch.nn, activation),
        )
        self.rescale = rescale
        replace_linear(self.flow)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).log_prob(
            x * self.rescale
        )

    def sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
        **kwargs,
    ):
        return (
            self.condition(context, null_context=null_context, cfg_w=cfg_w).sample(
                (num_samples,)
            )
            / self.rescale
        )

    def loss(self, target_sample, context=None):
        return -self.log_prob(target_sample, context=context)


def autograd_trace(outputs, inputs):
    trJ = 0.0
    dims = inputs.shape[1]
    for i in range(dims):
        trJ += torch.autograd.grad(outputs[:, i].sum(), inputs, create_graph=True)[0][
            :, i
        ]
    return trJ


class TimeSchedule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_size: int):
        raise NotImplementedError


class BetaSchedule(TimeSchedule):
    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, batch_size: int):
        return torch.distributions.beta.Beta(self.alpha, self.beta).sample(
            (batch_size,)
        )


class ODEFnWrapper(nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, *args, **kwargs):
        return self.ode_func(*args, **kwargs)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        dropout=0.0,
        activation=None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation or nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps using sinusoidal embeddings and an MLP.
    """

    def __init__(self, hidden_dim, embedding_dim=256):
        super().__init__()
        self.mlp = MLP(
            input_dim=embedding_dim, output_dim=embedding_dim, hidden_dim=hidden_dim
        )
        self.embedding_dim = embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=t.dtype)
            / half
        ).to(t)
        args = t[:, None].to(t) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.cond_proj = nn.Linear(embedding_dim, input_dim * 2)
        nn.init.trunc_normal_(self.cond_proj.weight, std=0.02)
        nn.init.constant_(self.cond_proj.bias, 0.0)

    def forward(
        self,
        x: Float[torch.Tensor, "b ... c"],
        cond: Float[torch.Tensor, "b d"],
    ) -> Float[torch.Tensor, "b ... c"]:
        c = x.shape[-1]
        num_spatial_dims = len(x.shape) - 2
        assert (
            c == self.input_dim
        ), f"input_dim must match the last dimension of x, got {c} != {self.input_dim}"

        # cond = self.cond_proj(cond)[:, *((None,) * num_spatial_dims), :]
        cond = self.cond_proj(cond)[
            (slice(None),) + (None,) * num_spatial_dims + (slice(None),)
        ]  # ugly hack for older python versions

        scale, shift = cond.chunk(2, dim=-1)

        x = F.layer_norm(x, [c])

        x = x * (1 + scale) + shift

        return x


class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        embed_dim: int = 256,
        hidden_dim: int = 1024,
        num_layers: int = 16,
        activation="SiLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert cond_dim > 0, "conditional dimension must be greater than 0"
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.act = getattr(nn, activation)()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.adanorms = nn.ModuleList(
            [AdaLayerNorm(embed_dim, cond_dim) for _ in range(num_layers)]
        )
        self.blocks = nn.ModuleList(
            [
                MLP(
                    input_dim=embed_dim,
                    output_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=self.act,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, input_dim)
        self.temb = TimestepEmbedder(hidden_dim=hidden_dim, embedding_dim=embed_dim)

        self.apply(self.init_weights)
        nn.init.constant_(self.out_proj.weight, 0.0)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, t, context=None):
        temb = self.temb(t)
        x = self.in_proj(x)
        for norm, block in zip(self.adanorms, self.blocks):
            shortcut = x
            x = norm(x, context)
            x = block(x + temb)
            x = x + shortcut
        x = self.out_proj(x)
        return x


class RectifiedFlow(nn.Module):
    def __init__(self, data_dim: int, time_schedule: nn.Module, score_model: nn.Module):
        super().__init__()
        self.data_dim = data_dim
        self.time_schedule = time_schedule
        self.score_model = score_model
        self.register_buffer("base_mu", torch.zeros(data_dim))
        self.register_buffer("base_sigma", torch.ones(data_dim))
        self.base_dist = torch.distributions.Normal

    def sample_base(self, num_samples: int, temperature: float = 1.0):
        return self.base_dist(self.base_mu, self.base_sigma * temperature).sample(
            (num_samples,)
        )

    def log_prob_base(self, inputs):
        return (
            self.base_dist(self.base_mu, self.base_sigma).log_prob(inputs).sum(axis=-1)
        )  # sum over all data dims (not batch)

    def loss(self, target_samples, context=None):
        batch_size = target_samples.shape[0]
        base_samples = self.sample_base(batch_size)
        times = self.time_schedule(batch_size).to(base_samples)
        inputs = (
            times.reshape(-1, 1) * target_samples
            + (1.0 - times.reshape(-1, 1)) * base_samples
        )
        velocity = self.score_model(inputs, times, context=context)
        target = target_samples - base_samples
        loss = torch.sum(
            (target - velocity) ** 2, dim=tuple(range(1, base_samples.ndim))
        )
        return loss

    def step(self, batch, batch_idx=None, **kwargs):
        if len(batch) == 2:
            x, context = batch
            return self.loss(x, context).mean()
        # otherwise, just one element (unconditional)
        return self.loss(batch).mean()

    def make_ode_fn(self, context, cfg_w, null_context=None):
        def ode_func(t, yt):
            t = torch.full((yt.shape[0],), t.item(), device=yt.device, dtype=yt.dtype)
            if cfg_w == 0.0:  # unconditional model
                assert null_context is not None
                v = self.score_model(yt, t, context=null_context)
            elif cfg_w == 1.0:  # conditional model
                v = self.score_model(yt, t, context=context)
            else:  # combination of conditional and unconditional
                assert null_context is not None
                batched_context = torch.cat([null_context, context], dim=0)
                # repeat yt and t twice along dim 0
                yt = torch.cat([yt, yt], dim=0)
                t = torch.cat([t, t], dim=0)
                # batched forward
                v = self.score_model(yt, t, context=batched_context)
                # split and combine
                v_uncond, v_cond = torch.split(
                    v, [context.shape[0], context.shape[0]], dim=0
                )
                v = v_uncond + cfg_w * (v_cond - v_uncond)
            return v

        return ode_func

    @torch.no_grad()
    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        atol=1e-10,
        rtol=1e-5,
        trace: str = "exact",
        adjoint: bool = False,
        cfg_w: float = 1.0,
        with_logprob: bool = False,
        **kwargs,
    ):
        """Sample from the flow. First draw from the base distribution, and then
        send the samples through the score model to get the velocity. Integrate
        the velocity to get the resulting samples.
        """

        B, N, D = context.shape if context is not None else (1, 1, 1)

        if context is not None:
            context = context.reshape(B * N, D)

        inputs = self.sample_base(
            B * N, temperature=temperature
        )  # this is [num_samples * batch_size, data_dim], need to unflatten after

        _ode_fn = self.make_ode_fn(context, cfg_w, null_context=null_context)

        integrator = odeint_adjoint if adjoint else odeint

        if with_logprob:
            noise = torch.randn_like(inputs)

            def ode_wrapper_hutchinson(t, y):
                y, _ = y
                fn_out, dfdy = torch.autograd.functional.vjp(
                    lambda x: _ode_fn(t=t, yt=x), y, noise
                )  # hardcode noise in here
                logp = (dfdy * noise).sum(axis=-1)
                return fn_out, logp

            def ode_wrapper_exact(t, y):
                y, _ = y
                with torch.enable_grad():
                    y = y.requires_grad_(True)
                    fn_out = _ode_fn(t=t, yt=y)
                    dfdy = autograd_trace(fn_out, y)
                return fn_out, dfdy

            ode_fn = dict(
                exact=ode_wrapper_exact,
                hutchinson=ode_wrapper_hutchinson,
            )[trace]

            if adjoint:
                ode_fn = ODEFnWrapper(ode_fn)

            init_delta_logp = torch.zeros(
                inputs.shape[0], device=inputs.device, dtype=inputs.dtype
            )

            ret, delta_logp = integrator(
                ode_fn,
                (inputs, init_delta_logp),
                torch.tensor([0.0, 1.0], device=inputs.device, dtype=inputs.dtype),
                atol=atol,
                rtol=rtol,
                **kwargs,
            )

            delta_logp = delta_logp[-1]
            log_prob = self.log_prob_base(inputs)
            lp = (log_prob - delta_logp).reshape(samples_per_batch, -1)
        else:
            ret = integrator(
                _ode_fn,
                inputs,
                torch.tensor([0.0, 1.0], device=inputs.device, dtype=inputs.dtype),
                atol=atol,
                rtol=rtol,
                **kwargs,
            )

            lp = None

        ret = ret[-1].reshape(
            B, N, self.data_dim
        )  # [num_samples, batch_size, data_dim]
        if context is None:
            ret = ret.squeeze(1)

        return ret, lp

    def log_prob(
        self,
        inputs,
        context=None,
        null_context: Optional[torch.Tensor] = None,
        atol=1e-10,
        rtol=1e-5,
        trace="exact",
        adjoint: bool = False,
        cfg_w: float = 1.0,
        **kwargs,
    ):
        """Compute the log probability of the inputs under the flow. Do this by
        taking the input samples and reverse integrating them back to the base distribution.
        Then compute the log probability of the base samples under the base distribution.
        """

        noise = torch.randn_like(inputs)

        _ode_fn = self.make_ode_fn(context, cfg_w, null_context=null_context)

        integrator = odeint_adjoint if adjoint else odeint

        def ode_wrapper_hutchinson(t, y):
            y, _ = y
            fn_out, dfdy = torch.autograd.functional.vjp(
                lambda x: _ode_fn(t=t, yt=x), y, noise
            )  # hardcode noise in here
            logp = (dfdy * noise).sum(axis=-1)
            return fn_out, logp

        def ode_wrapper_exact(t, y):
            y, _ = y
            with torch.enable_grad():
                y = y.requires_grad_(True)
                fn_out = _ode_fn(t=t, yt=y)
                dfdy = autograd_trace(fn_out, y)
            return fn_out, dfdy

        ode_fn = dict(
            exact=ode_wrapper_exact,
            hutchinson=ode_wrapper_hutchinson,
        )[trace]

        if adjoint:
            ode_fn = ODEFnWrapper(ode_fn)

        init_delta_logp = torch.zeros(
            inputs.shape[0], device=inputs.device, dtype=inputs.dtype
        )

        out, delta_logp = integrator(
            ode_fn,
            (inputs, init_delta_logp),
            torch.tensor([1.0, 0.0], device=inputs.device, dtype=inputs.dtype),
            atol=atol,
            rtol=rtol,
            **kwargs,
        )
        out, delta_logp = out[-1], delta_logp[-1]

        log_prob = self.log_prob_base(out)
        return log_prob + delta_logp
