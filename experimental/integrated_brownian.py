# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List

import torch


@torch.jit.script
def integrated_brownian_bridge(
        ws: torch.Tensor,
        wt: torch.Tensor,
        ust: torch.Tensor,
        s: float,
        t: float,
        m: float) -> List[torch.Tensor]:
    # ws: (batch_size, d_1, ..., d_k).
    # wt: (batch_size, d_1, ..., d_k).
    # ust: (batch_size, d_1, ..., d_k).
    # Compute the mean and covariance of N(x | y) and sample.
    mu_x = torch.stack((ws, torch.zeros_like(ws)), dim=-1)
    mu_y = torch.stack((ws, torch.zeros_like(ws)), dim=-1)
    y = torch.stack((wt, ust), dim=-1)

    # These small ops should be on CPU.
    A = torch.tensor(
        [[(m - s), (m - s) ** 2 / 2],
         [(m - s) ** 2 / 2, (m - s) ** 3 / 3]]
    )
    B = torch.tensor(
        [[(t - s), (t - s) ** 2 / 2],
         [(t - s) ** 2 / 2, (t - s) ** 3 / 3]]
    )
    C = torch.tensor(
        [[(m - s), (m - s) ** 2 / 2 + (m - s) * (t - m)],
         [(m - s) ** 2 / 2, (m - s) ** 3 / 3 + (m - s) ** 2 * (t - m) / 2]]
    )

    # Start porting to GPU.
    covariance = A - C @ torch.inverse(B) @ C.T
    L = torch.cholesky(covariance).to(ws)

    left_precond = C @ torch.inverse(B).to(ws)  # CB^{-1}, size=(2, 2).
    mean = mu_x + (y - mu_y) @ left_precond.T

    sample = mean + torch.randn_like(mean) @ L.T
    wm, usm = sample[..., 0], sample[..., 1]
    return [wm, usm]


@torch.jit.script
def joint_sample(ws: torch.Tensor, s: float, t: float) -> List[torch.Tensor]:
    xi, eta = torch.randn_like(ws), torch.randn_like(ws)
    dw = xi * math.sqrt(t - s)
    ust = (t - s) ** (3 / 2) * (1 / 2 * xi + 1 / math.sqrt(12) * eta)
    return [ws + dw, ust]


torch.set_default_dtype(torch.float64)
s, m, t = 0.0, 0.5, 1.0
batch_size, d_1, d_2 = 131072, 3, 10
ws = torch.randn(batch_size, d_1, d_2)
wt, ust = joint_sample(ws=ws, s=s, t=t)

# Bridge sampling.
wm, usm = integrated_brownian_bridge(ws=ws, wt=wt, ust=ust, s=s, t=t, m=m)  # What we want.

# Direct sampling.
wm_, usm_ = joint_sample(ws=ws, s=s, t=m)

print('w(m) mean: ', wm.mean(), wm_.mean())
print('w(m) standard deviation: ', wm.std(), wm_.std())

print('U_{s, m} mean: ', usm.mean(), usm_.mean())
print('U_{s, m} standard deviation: ', usm.std(), usm_.std())

assert torch.allclose(wm.mean(), wm_.mean(), rtol=1e-3, atol=1e-3)
assert torch.allclose(wm.std(), wm_.std(), rtol=1e-3, atol=1e-3)
assert torch.allclose(usm.mean(), usm_.mean(), rtol=1e-3, atol=1e-3)
assert torch.allclose(usm.std(), usm_.std(), rtol=1e-3, atol=1e-3)
