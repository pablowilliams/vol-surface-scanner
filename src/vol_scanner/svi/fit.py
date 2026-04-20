"""Per-slice SLSQP SVI fitter."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .constraints import (
    b_nonneg,
    lee_wing_constraint,
    positivity_constraint,
    rho_bounds,
    sigma_floor,
)
from .parametric import SVIParams, total_variance


@dataclass
class SliceFit:
    tenor: float
    params: SVIParams
    rmse: float
    iterations: int
    success: bool


def _objective(x: np.ndarray, k: np.ndarray, w_target: np.ndarray) -> float:
    p = SVIParams.from_array(x)
    w = total_variance(k, p)
    return float(np.mean((w - w_target) ** 2))


def fit_slice(
    k: np.ndarray,
    iv: np.ndarray,
    tenor: float,
    cfg: dict,
) -> SliceFit:
    w_target = (iv**2) * tenor

    x0 = np.array([
        cfg["initial"]["a"],
        cfg["initial"]["b"],
        cfg["initial"]["rho"],
        cfg["initial"]["m"],
        cfg["initial"]["sigma"],
    ])

    b = cfg["bounds"]
    bounds = [
        (b["a_min"], b["a_max"]),
        (b["b_min"], b["b_max"]),
        (b["rho_min"], b["rho_max"]),
        (b["m_min"], b["m_max"]),
        (b["sigma_min"], b["sigma_max"]),
    ]

    constraints = [
        {"type": "ineq", "fun": positivity_constraint},
        {"type": "ineq", "fun": rho_bounds},
        {"type": "ineq", "fun": lambda x: sigma_floor(x, 0.01)},
        {"type": "ineq", "fun": b_nonneg},
        {"type": "ineq", "fun": lambda x, t=tenor: lee_wing_constraint(x, t)},
    ]

    res = minimize(
        _objective,
        x0,
        args=(k, w_target),
        method=cfg["fit"]["method"],
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": int(cfg["fit"]["maxiter"]),
            "ftol": float(cfg["fit"]["ftol"]),
            "disp": False,
        },
    )

    p = SVIParams.from_array(res.x)
    w = total_variance(k, p)
    iv_fit = np.sqrt(np.clip(w, 1e-10, None) / max(tenor, 1e-6))
    rmse = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))

    return SliceFit(
        tenor=float(tenor),
        params=p,
        rmse=rmse,
        iterations=int(res.nit) if hasattr(res, "nit") else 0,
        success=bool(res.success),
    )


def fit_surface(
    k: np.ndarray,
    iv_matrix: np.ndarray,
    tenors: np.ndarray,
    cfg: dict,
) -> list[SliceFit]:
    """Fit every tenor slice and return the list in the input order."""
    fits: list[SliceFit] = []
    for i, t in enumerate(tenors):
        fit = fit_slice(k, iv_matrix[i, :], float(t), cfg)
        fits.append(fit)
    return fits


def predict_surface(k: np.ndarray, tenors: np.ndarray, fits: list[SliceFit]) -> np.ndarray:
    out = np.zeros((tenors.size, k.size))
    for i, f in enumerate(fits):
        w = total_variance(k, f.params)
        out[i, :] = np.sqrt(np.clip(w, 1e-10, None) / max(float(tenors[i]), 1e-6))
    return out
