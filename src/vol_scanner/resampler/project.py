"""Project an implied volatility surface onto the nearest no arbitrage surface.

We pose the problem as a constrained least squares in total variance space.
Let w_{i,j} = iv_{i,j}^2 t_i be the total variance on a rectangular grid.
The constraints we impose are

    w_{i,j} - w_{i-1,j} >= 0 for every j (calendar monotonicity),
    w_{i, j+1} - w_{i, j-1} lies within Gatheral moment bounds,
    second difference in k direction bounded below (butterfly proxy).

The cost is ||w - w_hat||_2^2 with w_hat the observed total variance. We
solve this via scipy.optimize.minimize with an L-BFGS-B flavour using the
SLSQP solver for linear inequality constraints. The mean absolute nudge
between w and w_hat, reported as a "market friction" metric, is the main
output along with the projected implied vol surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ResamplerReport:
    projected_iv: np.ndarray
    mean_nudge: float
    max_nudge: float
    mean_relative_nudge: float
    success: bool


def project_arbitrage_free(
    k_grid: np.ndarray,
    iv: np.ndarray,
    tenors: np.ndarray,
    max_iter: int = 80,
) -> ResamplerReport:
    n_t, n_k = iv.shape
    w_hat = (iv**2) * tenors[:, None]
    w0 = w_hat.flatten()

    def unpack(x: np.ndarray) -> np.ndarray:
        return x.reshape(n_t, n_k)

    def cost(x: np.ndarray) -> float:
        return float(np.sum((x - w0) ** 2))

    def cost_grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - w0)

    # Calendar constraint: w_{i,j} >= w_{i-1,j}.
    def calendar_con(x: np.ndarray) -> np.ndarray:
        w = unpack(x)
        return (w[1:, :] - w[:-1, :]).flatten()

    # Butterfly proxy: second difference of w in k direction >= -tol.
    def butterfly_con(x: np.ndarray) -> np.ndarray:
        w = unpack(x)
        tol = 1e-4
        diff2 = w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]
        return (diff2 + tol).flatten()

    # Vertical proxy: slope of w in k not too large (Roger Lee bound).
    def lee_con(x: np.ndarray) -> np.ndarray:
        w = unpack(x)
        # |dw/dk| <= 4 (loose Lee slope bound per slice).
        dk = k_grid[1:] - k_grid[:-1]
        dw = (w[:, 1:] - w[:, :-1]) / dk[None, :]
        return (4.0 - np.abs(dw)).flatten()

    # Positivity of total variance.
    def positivity(x: np.ndarray) -> np.ndarray:
        return x - 1e-6

    constraints = [
        {"type": "ineq", "fun": calendar_con},
        {"type": "ineq", "fun": butterfly_con},
        {"type": "ineq", "fun": lee_con},
        {"type": "ineq", "fun": positivity},
    ]

    res = minimize(
        cost,
        w0,
        jac=cost_grad,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-7, "disp": False},
    )
    w_proj = np.clip(unpack(res.x), 1e-8, None)
    iv_proj = np.sqrt(w_proj / np.clip(tenors[:, None], 1e-6, None))

    nudges = np.abs(iv_proj - iv)
    mean_nudge = float(nudges.mean())
    max_nudge = float(nudges.max())
    mean_rel = float((nudges / np.clip(iv, 1e-6, None)).mean())
    return ResamplerReport(
        projected_iv=iv_proj,
        mean_nudge=mean_nudge,
        max_nudge=max_nudge,
        mean_relative_nudge=mean_rel,
        success=bool(res.success),
    )
