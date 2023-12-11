# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from inspect import signature
import torch.nn.functional as F
from torch.autograd import Variable
import torchcde
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


PAD = 0


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class ConcatLinearNorm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinearNorm, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)
        self.norm = nn.LayerNorm(dim_out, eps=1e-6)

    def forward(self, t, x):
        return self.norm(self._layer(x) + self._hyper_bias(t.view(-1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
               + self._hyper_bias(t.view(-1, 1))


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y):
        if len(signature(self.module.forward).parameters) == 1:
            return self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            return self.module(t, y)
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def __repr__(self):
        return self.module.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers.
    """

    def __init__(self, *layers):
        super(SequentialDiffEq, self).__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


class TimeDependentSwish(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4)),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim),
            nn.Softplus(),
        )

    def forward(self, t, x):
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid_(x * beta)


class WrapRegularization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x

    @staticmethod
    def backward(ctx, *grad_x):
        reg, = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)


class TimeVariableODE(nn.Module):
    start_time = -1.0
    end_time = 1.0

    def __init__(self, func, atol=1e-6, rtol=1e-6, method="dopri5", energy_regularization=0.01, regularize=False,
                 ortho=False):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.regularize = regularize
        self.ortho = ortho
        self.nfe = 0
        self.gauss_legendre = {
            1: torch.tensor([-1, 0, 1]),
            2: torch.tensor([-1, -0.57735, 0.57735, 1]),
            3: torch.tensor([-1, -0.77459, 0, 0.77459, 1]),
            4: torch.tensor([-1, -0.86113, -0.33998, 0.33998, 0.86113, 1]),
            5: torch.tensor([-1, -0.90618, -0.53846, 0, 0.53846, 0.90618, 1])
        }

    def integrate(self, t0, t1, x0, nlinspace=1, method=None, gauss=False, save=None, atol=None):
        """
        t0: start time of shape [n]
        t1: end time of shape [n]
        x0: initial state of shape [n, d]
        """
        if save is not None:
            save_timestamp = save
        elif gauss:
            save_timestamp = self.gauss_legendre[nlinspace].to(t0)
        else:
            save_timestamp = torch.linspace(self.start_time, self.end_time, nlinspace + 1).to(t0)
        method = method or self.method
        atol = atol or self.atol

        solution = odeint(
            self,
            (t0, t1, torch.zeros(1).to(x0[0]), x0),
            save_timestamp,
            rtol=self.rtol,
            atol=self.atol,
            method=method,
            options=dict(step_size=atol)
        )
        _, _, energy, xs = solution
        if gauss:
            xs = xs[1: -1, ...]

        if self.regularize:
            reg = energy * self.energy_regularization
            return WrapRegularization.apply(reg, xs)[0]
        else:
            return xs

    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, _, x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        with torch.enable_grad():
            dx = self.func(t, x)

            if self.ortho:
                dx = dx - (dx * x).sum(dim=-1, keepdim=True) / (x * x).sum(dim=-1, keepdim=True) * x
            dx = dx * ratio.reshape(-1, *([1] * (dx.ndim - 1)))

            d_energy = torch.sum(dx * dx) / x.numel()

        if not self.training:
            dx = dx.detach()

        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), d_energy, dx])

    def extra_repr(self):
        return f"method={self.method}, atol={self.atol}, rtol={self.rtol}, energy={self.energy_regularization}"


ACTFNS = {
    "softplus": (lambda dim: nn.Softplus()),
    "tanh": (lambda dim: nn.Tanh()),
    "swish": (lambda dim: TimeDependentSwish(dim)),
    "relu": (lambda dim: nn.ReLU()),
    'leakyrelu': (lambda dim: nn.LeakyReLU()),
    'sigmoid': (lambda dim: nn.Sigmoid()),
}

LAYERTYPES = {
    "concatsquash": ConcatSquashLinear,
    "concat": ConcatLinear_v2,
    "concatlinear": ConcatLinear_v2,
    "concatnorm": ConcatLinearNorm,
}


def build_fc_odefunc(dim=2, hidden_dims=[64, 64, 64], out_dim=None, nonzero_dim=None, actfn="softplus",
                     layer_type="concat",
                     zero_init=True, actfirst=False):
    assert layer_type in LAYERTYPES.keys(), f"layer_type must be one of {LAYERTYPES.keys()} but was given {layer_type}"
    layer_fn = LAYERTYPES[layer_type]
    if layer_type == "concatlinear":
        hidden_dims = None

    nonzero_dim = dim if nonzero_dim is None else nonzero_dim
    out_dim = out_dim or hidden_dims[-1]
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(layer_fn(d_in, d_out))
            layers.append(ACTFNS[actfn](d_out))
        layers.append(layer_fn(hidden_dims[-1], out_dim))
        layers.append(ACTFNS[actfn](out_dim))
    else:
        layers = [layer_fn(dim, out_dim), ACTFNS[actfn](out_dim)]

    if actfirst and len(layers) > 1:
        layers = layers[1:]

    if nonzero_dim < dim:
        # zero out weights for auxiliary inputs.
        layers[0]._layer.weight.data[:, nonzero_dim:].fill_(0)

    if zero_init:
        for m in layers[-2].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    """
    else:
        for l in layers:
            for m in l.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=1e-3)
                    if m.bias is not None:
                        m.bias.data.zero_()
    """

    return SequentialDiffEq(*layers)


def _setup_hermite_cubic_coeffs_w_backward_differences(times, coeffs, derivs, device=None):
    """Compute backward hermite from linear coeffs."""
    x_prev = coeffs[..., :-1, :]
    x_next = coeffs[..., 1:, :]
    # Let x_0 - x_{-1} = x_1 - x_0
    derivs_prev = torch.cat((derivs[..., [0], :], derivs[..., :-1, :]), axis=-2)
    derivs_next = derivs
    x_diff = x_next - x_prev
    t_diff = (times[1:] - times[:-1]).unsqueeze(-1)
    # Coeffs
    a = x_prev
    b = derivs_prev
    two_c = 2 * (3 * (x_diff / t_diff - b) - derivs_next + derivs_prev) / t_diff
    three_d = (1 / t_diff ** 2) * (derivs_next - b) - (two_c) / t_diff
    coeffs = torch.cat([a, b, two_c, three_d], dim=-1).to(device)
    return coeffs


def hermite_cubic_coefficients_with_backward_differences(x, t=None):
    """Computes the coefficients for hermite cubic splines with backward differences.

    Arguments:
        As `torchcde.linear_interpolation_coeffs`.

    Returns:
        A tensor, which should in turn be passed to `torchcde.CubicSpline`.
    """
    # Linear coeffs
    coeffs = linear_interpolation_coeffs(x, t=t)

    if t is None:
        t = torch.linspace(0, coeffs.size(-2) - 1, coeffs.size(-2), dtype=coeffs.dtype, device=coeffs.device)

    # Linear derivs
    derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1]).unsqueeze(-1)

    # Use the above to compute hermite coeffs
    hermite_coeffs = _setup_hermite_cubic_coeffs_w_backward_differences(t, coeffs, derivs, device=coeffs.device)

    return hermite_coeffs


def linear_interpolation_coeffs(x, t):
    coeffs = []
    for i in range(x.size(0)):
        index = torch.where(torch.isnan(x[i, :, 0]) == False)[0]

        if len(index) == 1:
            coeff = x[i][index[0]].unsqueeze(0)
        elif len(index) == 0:
            coeff = torch.zeros_like(x[i])
        else:
            coeff = x[i].clone()
            if coeff[0].isnan().any():
                coeff[:index[0], :] = x[i][index[0]].unsqueeze(0)
            if coeff[-1].isnan().any():
                coeff[index[-1] + 1:, :] = x[i][index[-1]].unsqueeze(0)
            for (pre, post) in zip(index[:-1], index[1:]):
                rate = []
                for j in range(pre + 1, post):
                    rate.append(((t[j] - t[pre]) / (t[post] - t[pre])))
                rate = torch.tensor(rate).to(coeff)
                coeff[pre + 1: post] = x[i][pre].unsqueeze(0) \
                                       + rate.unsqueeze(-1) * (x[i][post] - x[i][pre]).unsqueeze(0)
        coeffs.append(coeff.unsqueeze(0))
    return torch.cat(coeffs, dim=0)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x


class InterpLinear(nn.Module):
    def __init__(self, d_model, d_out=None, args_interp=None, norm=True):
        super().__init__()
        d_out = d_out or d_model
        if args_interp.linear_type != 'inside':
            self.lin_outside = nn.Linear(d_model, d_out)
            nn.init.xavier_uniform_(self.lin_outside.weight)
        self.norm = norm

        self.gauss_weight = {
            1: torch.tensor([2]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([0.55555, 0.88888, 0.55555]),
            4: torch.tensor([0.34785, 0.65214, 0.65214, 0.34785]),
            5: torch.tensor([0.23692, 0.47863, 0.56888, 0.47863, 0.23692])
        }
        self.gauss_legendre = {
            1: torch.tensor([0]),
            2: torch.tensor([-0.57735, 0.57735]),
            3: torch.tensor([-0.77459, 0, 0.77459]),
            4: torch.tensor([-0.86113, -0.33998, 0.33998, 0.86113]),
            5: torch.tensor([-0.90618, -0.53846, 0, 0.53846, 0.90618])
        }

        self.nlinspace = args_interp.nlinspace
        self.approximate_method = args_interp.approximate_method
        self.interpolation = args_interp.interpolate
        self.linear_type = args_interp.linear_type
        if self.approximate_method == 'bilinear':
            self.nlinspace = 1
        self.atol = args_interp.itol
        self.d_model = d_model

    def pre_integrals(self, x, t):
        """
        x: hidden state of shape: [B, T, D]
        t: event time of shape: [B, T]
        """
        # fill PAD in t with the last observed time
        pad_num = torch.sum(t[:, 1:] == PAD, dim=-1)

        T = t.clone()
        for i in range(t.size(0)):
            if pad_num[i] == 0:
                continue
            T[i][-pad_num[i]:] = T[i][-pad_num[i] - 1]

        tt, sort_index = torch.unique(T, sorted=True, return_inverse=True)

        xx = torch.full((x.size(0), len(tt), x.size(-1)), float('nan')).to(x)

        r = torch.arange(x.size(0)).reshape(-1, 1).repeat(1, x.size(1)).reshape(-1)

        # fill in non-nan values
        xx[r.numpy(), sort_index.reshape(-1).cpu().numpy(), :] = x.reshape(x.size(0) * x.size(1), -1)  # [B, TT, D]

        # interpolation
        if self.interpolation == 'linear':
            coeffs = linear_interpolation_coeffs(xx, t=tt)
        elif self.interpolation == 'cubic':
            coeffs = hermite_cubic_coefficients_with_backward_differences(xx, t=tt)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        return coeffs, tt

    def forward(self, x, t, approximate_method=None):
        """
            x: hidden state vector for q or k with shape [B, T, D]
            t: event time with shape [B, T]

            Return: [B, T, T, K, D]
        """
        T, Q = t.shape[1], t.shape[1]

        if self.linear_type == 'before':
            x = self.lin_outside(x)

        coeffs, intervals = self.pre_integrals(x, t)
        if self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs, t=intervals)
        elif self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=intervals)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = t.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        approximate_method = approximate_method or self.approximate_method
        if approximate_method == 'bilinear':
            interp_t = torch.cat((t0.unsqueeze(-1), t1.unsqueeze(-1)), dim=-1)
        elif approximate_method == 'gauss':
            interp_rate = self.gauss_legendre[self.nlinspace].to(t0.device)
            interp_t = t0.unsqueeze(-1) + \
                       (t1.unsqueeze(-1) - t0.unsqueeze(-1)) * (interp_rate + 1) / 2
        else:
            interp_t = t1.unsqueeze(-1)

        discret_t = (interp_t / self.atol).long()  # [B, T, T, K]

        linspace_t = torch.linspace(0, t.max().item() + 5 * self.atol, int(t.max().item() / self.atol + 6))
        interp_f = X.evaluate(linspace_t)  # [B, L, D]

        discret_t = discret_t.reshape(t0.shape[0], -1).detach().cpu()  # [B, T * T * K]
        idx = torch.arange(t0.shape[0]).unsqueeze(-1).repeat(1, discret_t.shape[1])  # [B, T * T * K]

        x = interp_f[idx.reshape(-1), discret_t.reshape(-1), ...]
        x = x.reshape(t0.shape[0], t0.shape[1], t0.shape[2], -1, self.d_model)

        if self.linear_type == 'after':
            x = self.lin_outside(x)

        if self.approximate_method == 'bilinear':
            x = x * 0.5
        else:
            x = x * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5

        if not self.norm:
            x = x * (t1 - t0).unsqueeze(-1).unsqueeze(-1)
        return x

    def interpolate(self, x, t, qt, mask=None, approximate_method=None):
        """
            x: hidden state vector for q or k with shape [B, T, D]
            t: event time with shape [B, T]
            qt: query time with shape [B, Q]
            mask: mask for unknown events [B, Q, T]

            Return: [B, Q, T, K, D]
        """

        T, Q = t.shape[1], qt.shape[1]

        # current_times = torch.ones(t.shape[0], Q).to(t.device) * 100000

        if self.linear_type == 'before':
            x = self.lin_outside(x)

        coeffs, intervals = self.pre_integrals(x, t)
        if self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs, t=intervals)
        elif self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=intervals)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = qt.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        approximate_method = approximate_method or self.approximate_method
        if approximate_method == 'bilinear':
            interp_t = torch.cat((t0.unsqueeze(-1), t1.unsqueeze(-1)), dim=-1)
        elif approximate_method == 'gauss':
            interp_rate = self.gauss_legendre[self.nlinspace].to(t0.device)
            interp_t = t0.unsqueeze(-1) + \
                       (t1.unsqueeze(-1) - t0.unsqueeze(-1)) * (interp_rate + 1) / 2
        else:
            interp_t = t1.unsqueeze(-1)

        # current_times = current_times.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, interp_t.shape[2], interp_t.shape[3])
        # interp_t = torch.min(interp_t, current_times)

        discret_t = (interp_t / self.atol).long()  # [B, Q, T, K]

        linspace_t = torch.linspace(0, qt.max().item() + 5 * self.atol, int(qt.max().item() / self.atol + 6))
        interp_f = X.evaluate(linspace_t)  # [B, L, D]

        discret_t = discret_t.reshape(t0.shape[0], -1).detach().cpu()  # [B, Q * T * K]
        idx = torch.arange(t0.shape[0]).unsqueeze(-1).repeat(1, discret_t.shape[1])  # [B, Q * T * K]

        x = interp_f[idx.reshape(-1), discret_t.reshape(-1), ...]
        x = x.reshape(t0.shape[0], t0.shape[1], t0.shape[2], -1, self.d_model)

        if self.linear_type == 'after':
            x = self.lin_outside(x)

        if self.approximate_method == 'bilinear':
            x = x * 0.5
        else:
            x = x * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5

        if not self.norm:
            x = x * (t1 - t0).reshape(x.size(0), x.size(-2), x.size(-2)).unsqueeze(-1).unsqueeze(-1)

        return x


class ODELinear(nn.Module):
    def __init__(self, d_model, d_out=None, args_ode=None, norm=True):
        super().__init__()
        d_out = d_out or d_model
        self.norm = norm
        if args_ode.linear_type == 'inside':
            self.ode_func = build_fc_odefunc(d_model, out_dim=d_model, actfn=args_ode.actfn,
                                             layer_type=args_ode.layer_type,
                                             zero_init=args_ode.zero_init, hidden_dims=[d_model])
        else:
            if args_ode.linear_type == 'before':
                self.ode_func = build_fc_odefunc(d_out, out_dim=d_out, actfn=args_ode.actfn,
                                                 layer_type=args_ode.layer_type,
                                                 zero_init=args_ode.zero_init, hidden_dims=None)
                self.lin_outside = nn.Linear(d_model, d_out)
                nn.init.xavier_uniform_(self.lin_outside.weight)
            else:
                self.ode_func = build_fc_odefunc(d_model, out_dim=d_model, actfn=args_ode.actfn,
                                                 layer_type=args_ode.layer_type,
                                                 zero_init=args_ode.zero_init, hidden_dims=None)
                self.lin_outside = nn.Linear(d_model, d_out)
                nn.init.xavier_uniform_(self.lin_outside.weight)

        self.linear_type = args_ode.linear_type
        self.ode = TimeVariableODE(self.ode_func, atol=args_ode.atol, rtol=args_ode.rtol,
                                   method=args_ode.method, regularize=args_ode.regularize)
        self.approximate_method = args_ode.approximate_method
        self.gauss_weight = {
            1: torch.tensor([2]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([0.55555, 0.88888, 0.55555]),
            4: torch.tensor([0.34785, 0.65214, 0.65214, 0.34785]),
            5: torch.tensor([0.23692, 0.47863, 0.56888, 0.47863, 0.23692])
        }
        self.d_model = d_model
        self.nlinspace = args_ode.nlinspace
        if self.approximate_method == 'bilinear' or self.approximate_method == 'last':
            self.nlinspace = 1

    def forward(self, x, t, approximate_method=None):
        """
        x: hidden state vector for q or k with shape [B, T, D]
        t: event time with shape [B, T]

        Return: [B, T, T, K, D]
        """

        T, Q = t.shape[1], t.shape[1]
        BS = t.shape[0]

        x0 = x.unsqueeze(-3).repeat(1, Q, 1, 1)  # [B, Q, T, D]
        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = t.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        x0 = x0.reshape(-1, self.d_model)
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)

        if self.linear_type == 'before':
            x0 = self.lin_outside(x0)

        y = self.ode.integrate(t0, t1, x0, nlinspace=self.nlinspace, gauss=self.approximate_method == 'gauss')

        if self.linear_type == 'after':
            y = self.lin_outside(y)

        y = y.reshape(y.size(0), BS, Q, T, -1)  # [K, B, Q, T, D]
        y = y.permute(1, 2, 3, 0, 4)

        approximate_method = approximate_method or self.approximate_method

        if approximate_method == 'bilinear':
            y = y * 0.5
        elif approximate_method == 'gauss':
            y = y * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5
        else:
            y = y[..., -1, :]
        if not self.norm:
            y = y * (t1 - t0).reshape(x.size(0), x.size(-2), x.size(-2)).unsqueeze(-1).unsqueeze(-1)
        return y

    def interpolate(self, x, t, qt, approximate_method=None):
        """
            x: hidden state vector for q or k with shape [B, T, D]
            t: event time with shape [B, T]
            qt: query time with shape [B, Q]
            Return: [B, Q, T, K, D]
        """

        T, Q = t.shape[1], qt.shape[1]

        x0 = x.unsqueeze(-3).repeat(1, Q, 1, 1)  # [B, Q, T, D]
        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = qt.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        x0 = x0.reshape(-1, self.d_model)
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)

        if self.linear_type == 'before':
            x0 = self.lin_outside(x0)

        y = self.ode.integrate(t0, t1, x0, nlinspace=self.nlinspace, gauss=self.approximate_method == 'gauss')

        if self.linear_type == 'after':
            y = self.lin_outside(y)

        y = y.reshape(y.size(0), -1, Q, T, self.d_model)  # [K, B, Q, T, D]
        y = y.permute(1, 2, 3, 0, 4)

        approximate_method = approximate_method or self.approximate_method

        if approximate_method == 'bilinear':
            y = y * 0.5
        elif approximate_method == 'gauss':
            y = y * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5
        else:
            y = y[..., -1:, :]

        if not self.norm:
            y = y * (t1 - t0).reshape(x.size(0), x.size(-2), x.size(-2)).unsqueeze(-1).unsqueeze(-1)
        return y


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True, args_ode=None):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        assert args_ode.use_ode
        assert d_model == n_head * d_k
        assert d_model == n_head * d_v

        self.w_qs = InterpLinear(d_model, n_head * d_k, args_ode)
        self.w_ks = ODELinear(d_model, n_head * d_k, args_ode)
        self.w_vs = ODELinear(d_model, n_head * d_v, args_ode)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, t, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q, t).view(sz_b, len_q, len_q, -1, n_head, d_k)
        k = self.w_ks(k, t).view(sz_b, len_k, len_k, -1, n_head, d_k)
        v = self.w_vs(v, t).view(sz_b, len_v, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

    def interpolate(self, q, k, v, t, qt, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len_qt = qt.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs.interpolate(q, t, qt, mask=mask).view(sz_b, len_qt, len_q, -1, n_head, d_k)
        k = self.w_ks.interpolate(k, t, qt).view(sz_b, len_qt, len_k, -1, n_head, d_k)
        v = self.w_vs.interpolate(v, t, qt).view(sz_b, len_qt, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention.interpolate(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_qt, -1)
        output = self.fc(output)

        return output


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # if q is ODELinear, attn = (q.transpose(2, 3).flip(dims=[-2]) / self.temperature * k).sum(dim=-1).sum(dim=-1)
        attn = (q / self.temperature * k).sum(dim=-1).sum(dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = (attn.unsqueeze(-1) * v.sum(dim=-2)).sum(dim=-2)
        return output, attn

    def interpolate(self, q, k, v, mask=None):
        attn = (q / self.temperature * k).sum(dim=-1).sum(dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        output = (attn.unsqueeze(-1) * v.sum(dim=-2)).sum(dim=-2)
        return output, attn


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, args=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, args_ode=args)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, time_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, time_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
