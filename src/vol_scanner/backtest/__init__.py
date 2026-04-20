"""Rolling window backtest of the arbitrage scanner."""
from .rolling import BacktestResult, rolling_backtest

__all__ = ["rolling_backtest", "BacktestResult"]
