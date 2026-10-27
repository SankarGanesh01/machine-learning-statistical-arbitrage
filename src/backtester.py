"""Institutional statistical arbitrage execution engine.

The module provides a path-dependent, hedge-ratio aware backtester for both
traditional pair selection and machine-learning generated candidate pairs.
The public compatibility alias ``VectorizedBacktester`` is preserved at the
bottom of the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
NIFTY_ANNUAL_RFR = 0.072
SPX_ANNUAL_RFR = 0.042
EPSILON = 1e-12


@dataclass
class Trade:
    """Executed spread trade with full lifecycle information."""

    ticker1: str
    ticker2: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str
    holding_period: int
    pnl: float
    exit_reason: str
    entry_beta: float
    exit_beta: float
    entry_z: float
    exit_z: float
    gross_return: float
    net_return: float
    transaction_cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a notebook-friendly trade record."""

        return {
            "Ticker1": self.ticker1,
            "Ticker2": self.ticker2,
            "Entry Date": self.entry_date,
            "Exit Date": self.exit_date,
            "Entry Price": self.entry_price,
            "Exit Price": self.exit_price,
            "Direction": self.direction,
            "Holding Period": self.holding_period,
            "PnL": self.pnl,
            "Exit Reason": self.exit_reason,
            "Entry Beta": self.entry_beta,
            "Exit Beta": self.exit_beta,
            "Entry Z": self.entry_z,
            "Exit Z": self.exit_z,
            "Gross Return": self.gross_return,
            "Net Return": self.net_return,
            "Transaction Cost": self.transaction_cost,
        }


@dataclass
class BacktestResult:
    """Complete pair-level backtest result."""

    ticker1: str
    ticker2: str
    rolling_beta: pd.Series
    roi: float
    cagr: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    maximum_drawdown: float
    profit_factor: float
    win_rate: float
    average_trade_return: float
    average_holding_period: float
    trade_count: int
    turnover: float
    trade_log: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    spread: pd.Series
    z_score: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary preserving the historical notebook API."""

        trade_records = [trade.to_dict() for trade in self.trade_log]
        return {
            "ticker1": self.ticker1,
            "ticker2": self.ticker2,
            "Ticker1": self.ticker1,
            "Ticker2": self.ticker2,
            "Rolling_Beta": self.rolling_beta,
            "ROI": self.roi,
            "CAGR": self.cagr * 100.0,
            "CAGR_Decimal": self.cagr,
            "Annual_Return": self.annual_return * 100.0,
            "Annual_Return_Decimal": self.annual_return,
            "Annual_Volatility": self.annual_volatility * 100.0,
            "Annual_Volatility_Decimal": self.annual_volatility,
            "Sharpe": self.sharpe,
            "Sortino": self.sortino,
            "Calmar": self.calmar,
            "Max_Drawdown": self.maximum_drawdown * 100.0,
            "Maximum_Drawdown": self.maximum_drawdown,
            "Profit_Factor": self.profit_factor,
            "Win_Rate": self.win_rate,
            "Average_Trade_Return": self.average_trade_return,
            "Average_Holding_Period": self.average_holding_period,
            "Trades": self.trade_count,
            "Trade_Count": self.trade_count,
            "Turnover": self.turnover,
            "trade_log": trade_records,
            "Trade_Log": trade_records,
            "equity_curve": self.equity_curve,
            "returns": self.returns,
            "positions": self.positions,
            "spread": self.spread,
            "z_score": self.z_score,
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a value to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _clean_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, numeric, strictly positive price frame."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty:
        return prices.copy()

    cleaned = prices.copy().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned.where(cleaned > 0.0)


def _market_annual_rfr(market: str) -> float:
    """Map a supported market name to its annual risk-free rate."""

    market_key = str(market).upper().replace(" ", "")
    nifty_names = {"NIFTY", "NIFTY500", "NSE", "INDIA", "IN"}
    spx_names = {"SP", "SPX", "S&P", "S&P500", "SP500", "USA", "US"}
    if market_key in nifty_names:
        return NIFTY_ANNUAL_RFR
    if market_key in spx_names:
        return SPX_ANNUAL_RFR
    return SPX_ANNUAL_RFR


def _clean_return_series(returns: pd.Series) -> pd.Series:
    """Replace non-finite returns and prevent impossible capital jumps."""

    cleaned = pd.to_numeric(returns, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cleaned.clip(lower=-0.999999)


def _annualized_years(index: pd.Index, observations: int) -> float:
    """Estimate elapsed years for CAGR without relying on one convention."""

    sample_years = max(observations, 1) / TRADING_DAYS_PER_YEAR
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        elapsed_days = max((index[-1] - index[0]).days, 1)
        calendar_years = elapsed_days / 365.25
        return max(calendar_years, sample_years, 1.0 / TRADING_DAYS_PER_YEAR)
    return max(sample_years, 1.0 / TRADING_DAYS_PER_YEAR)


def _drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return percentage drawdown from the running peak."""

    running_peak = equity_curve.cummax().replace(0.0, np.nan)
    drawdown = (equity_curve / running_peak) - 1.0
    return drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide two floats and return zero when the denominator is unstable."""

    if abs(denominator) <= EPSILON or not np.isfinite(denominator):
        return 0.0
    return _as_float(numerator / denominator)


def _position_weights(
    direction: int,
    beta: float,
    normalize_exposure: bool = True,
) -> Tuple[float, float]:
    """Convert spread direction and hedge ratio into leg weights."""

    if direction == 0 or not np.isfinite(beta):
        return 0.0, 0.0
    if normalize_exposure:
        gross_exposure = max(1.0 + abs(beta), EPSILON)
        return direction / gross_exposure, -direction * beta / gross_exposure
    return float(direction), -float(direction) * beta


def _portfolio_turnover(
    previous_weights: Tuple[float, float],
    next_weights: Tuple[float, float],
) -> float:
    """Calculate two-leg absolute turnover."""

    return abs(next_weights[0] - previous_weights[0]) + abs(
        next_weights[1] - previous_weights[1]
    )


def _profit_factor(trades: Sequence[Trade]) -> float:
    """Calculate profit factor from completed trade PnL."""

    if not trades:
        return 0.0
    profits = sum(trade.pnl for trade in trades if trade.pnl > 0.0)
    losses = abs(sum(trade.pnl for trade in trades if trade.pnl < 0.0))
    if profits <= EPSILON and losses <= EPSILON:
        return 0.0
    if losses <= EPSILON:
        return float("inf")
    return profits / losses


def _performance_statistics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: Sequence[Trade],
    annual_rfr: float,
    periods_per_year: int,
    turnover: float,
) -> Dict[str, float]:
    """Compute institutional-grade performance statistics."""

    returns = _clean_return_series(returns)
    equity_curve = pd.to_numeric(equity_curve, errors="coerce").fillna(1.0)
    if equity_curve.empty:
        equity_curve = pd.Series([1.0], index=pd.RangeIndex(1))

    initial_equity = max(_as_float(equity_curve.iloc[0], 1.0), EPSILON)
    final_equity = max(_as_float(equity_curve.iloc[-1], 1.0), 0.0)
    roi = (final_equity / initial_equity) - 1.0
    years = _annualized_years(equity_curve.index, len(equity_curve))
    cagr = -1.0 if final_equity <= 0.0 else (
        (final_equity / initial_equity) ** (1.0 / years)
    ) - 1.0

    annual_return = returns.mean() * periods_per_year
    annual_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = _safe_ratio(annual_return - annual_rfr, annual_volatility)

    daily_target = annual_rfr / periods_per_year
    downside = returns - daily_target
    downside = downside[downside < 0.0]
    downside_volatility = downside.std(ddof=1) * np.sqrt(periods_per_year)
    sortino = _safe_ratio(annual_return - annual_rfr, downside_volatility)

    max_drawdown = _as_float(_drawdown_series(equity_curve).min())
    calmar = _safe_ratio(cagr, abs(max_drawdown))
    trade_pnls = [trade.pnl for trade in trades]
    winning_trades = [pnl for pnl in trade_pnls if pnl > 0.0]
    trade_count = len(trades)
    win_rate = len(winning_trades) / trade_count if trade_count else 0.0
    average_trade_return = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    average_holding_period = (
        float(np.mean([trade.holding_period for trade in trades]))
        if trades
        else 0.0
    )

    return {
        "roi": _as_float(roi),
        "cagr": _as_float(cagr),
        "annual_return": _as_float(annual_return),
        "annual_volatility": _as_float(annual_volatility),
        "sharpe": _as_float(sharpe),
        "sortino": _as_float(sortino),
        "calmar": _as_float(calmar),
        "maximum_drawdown": _as_float(max_drawdown),
        "profit_factor": _profit_factor(trades),
        "win_rate": _as_float(win_rate),
        "average_trade_return": _as_float(average_trade_return),
        "average_holding_period": _as_float(average_holding_period),
        "trade_count": trade_count,
        "turnover": _as_float(turnover),
    }


class TradeExecutionEngine:
    """Path-dependent execution model for a hedge-ratio adjusted spread."""

    def __init__(self, parent: "StatisticalArbitrageEngine") -> None:
        self.parent = parent

    def execute(
        self,
        data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
    ) -> Dict[str, Any]:
        """Execute pair trades from prepared spread and z-score data."""

        if data.empty:
            return {"execution": pd.DataFrame(index=data.index), "trades": []}

        required_columns = {"price1", "price2", "beta", "spread", "z_score"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                "execution data is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        frame = data.copy().sort_index()
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["price1", "price2", "beta", "z_score"])
        if frame.empty:
            return {"execution": pd.DataFrame(index=frame.index), "trades": []}

        price1_returns = _clean_return_series(frame["price1"].pct_change())
        price2_returns = _clean_return_series(frame["price2"].pct_change())
        index = frame.index
        gross_returns = pd.Series(0.0, index=index)
        transaction_costs = pd.Series(0.0, index=index)
        net_returns = pd.Series(0.0, index=index)
        turnover = pd.Series(0.0, index=index)
        directions = pd.Series(0, index=index, dtype="int64")
        position_beta = pd.Series(0.0, index=index)
        weight1 = pd.Series(0.0, index=index)
        weight2 = pd.Series(0.0, index=index)

        trades: List[Trade] = []
        current_direction = 0
        current_beta = 0.0
        entry_date: Optional[pd.Timestamp] = None
        entry_index: Optional[int] = None
        entry_spread = 0.0
        entry_z = 0.0
        entry_beta = 0.0
        trade_gross_equity = 1.0
        trade_net_equity = 1.0
        trade_cost = 0.0

        for i, date in enumerate(index):
            z_score = _as_float(frame.at[date, "z_score"], np.nan)
            beta_today = _as_float(frame.at[date, "beta"], np.nan)
            if not np.isfinite(z_score) or not np.isfinite(beta_today):
                continue

            old_weights = _position_weights(
                current_direction,
                current_beta,
                self.parent.normalize_exposure,
            )
            gross_return = 0.0
            if current_direction != 0 and i > 0:
                gross_return = (
                    old_weights[0] * price1_returns.iloc[i]
                    + old_weights[1] * price2_returns.iloc[i]
                )
                gross_return = self.parent._clip_daily_return(gross_return)
                trade_gross_equity *= max(1.0 + gross_return, 0.0)
                trade_net_equity *= max(1.0 + gross_return, 0.0)

            daily_cost = 0.0
            daily_turnover = 0.0
            exit_reason = self._exit_reason(
                current_direction=current_direction,
                current_trade_return=trade_net_equity - 1.0,
                z_score=z_score,
                bar_index=i,
                entry_index=entry_index,
            )

            reversed_position = False
            if exit_reason is not None:
                flat_weights = (0.0, 0.0)
                exit_turnover = _portfolio_turnover(old_weights, flat_weights)
                exit_cost = self.parent.calculate_transaction_costs(
                    turnover=exit_turnover,
                    z_score=z_score,
                    event_type="exit",
                )
                daily_cost += exit_cost
                daily_turnover += exit_turnover
                trade_cost += exit_cost
                trade_net_equity = max(trade_net_equity - exit_cost, 0.0)
                trades.append(
                    self._make_trade(
                        ticker1=ticker1,
                        ticker2=ticker2,
                        entry_date=entry_date,
                        exit_date=pd.Timestamp(date),
                        entry_spread=entry_spread,
                        exit_spread=_as_float(frame.at[date, "spread"]),
                        direction=current_direction,
                        holding_period=(
                            i - entry_index if entry_index is not None else 0
                        ),
                        exit_reason=exit_reason,
                        entry_beta=entry_beta,
                        exit_beta=current_beta,
                        entry_z=entry_z,
                        exit_z=z_score,
                        gross_return=trade_gross_equity - 1.0,
                        net_return=trade_net_equity - 1.0,
                        transaction_cost=trade_cost,
                    )
                )

                reversal_direction = -current_direction
                current_direction = 0
                current_beta = 0.0
                entry_date = None
                entry_index = None
                trade_gross_equity = 1.0
                trade_net_equity = 1.0
                trade_cost = 0.0

                if (
                    exit_reason == "Reversal"
                    and self.parent.allow_reversals
                    and i < len(index) - 1
                ):
                    entry_cost, entry_turnover = self._open_position_cost(
                        reversal_direction,
                        beta_today,
                        z_score,
                    )
                    daily_cost += entry_cost
                    daily_turnover += entry_turnover
                    current_direction = reversal_direction
                    current_beta = beta_today
                    entry_date = pd.Timestamp(date)
                    entry_index = i
                    entry_spread = _as_float(frame.at[date, "spread"])
                    entry_z = z_score
                    entry_beta = beta_today
                    trade_net_equity = max(1.0 - entry_cost, 0.0)
                    trade_cost = entry_cost
                    reversed_position = True

            if current_direction == 0 and not reversed_position and i < len(index) - 1:
                entry_direction = self._entry_direction(z_score)
                if entry_direction != 0:
                    entry_cost, entry_turnover = self._open_position_cost(
                        entry_direction,
                        beta_today,
                        z_score,
                    )
                    daily_cost += entry_cost
                    daily_turnover += entry_turnover
                    current_direction = entry_direction
                    current_beta = beta_today
                    entry_date = pd.Timestamp(date)
                    entry_index = i
                    entry_spread = _as_float(frame.at[date, "spread"])
                    entry_z = z_score
                    entry_beta = beta_today
                    trade_net_equity = max(1.0 - entry_cost, 0.0)
                    trade_cost = entry_cost

            elif current_direction != 0 and exit_reason is None:
                next_weights = _position_weights(
                    current_direction,
                    beta_today,
                    self.parent.normalize_exposure,
                )
                rebalance_turnover = _portfolio_turnover(
                    old_weights,
                    next_weights,
                )
                if rebalance_turnover > EPSILON:
                    rebalance_cost = self.parent.calculate_transaction_costs(
                        turnover=rebalance_turnover,
                        z_score=z_score,
                        event_type="rebalance",
                    )
                    daily_cost += rebalance_cost
                    daily_turnover += rebalance_turnover
                    trade_cost += rebalance_cost
                    trade_net_equity = max(
                        trade_net_equity - rebalance_cost,
                        0.0,
                    )
                current_beta = beta_today

            new_weights = _position_weights(
                current_direction,
                current_beta,
                self.parent.normalize_exposure,
            )
            gross_returns.iloc[i] = gross_return
            transaction_costs.iloc[i] = daily_cost
            turnover.iloc[i] = daily_turnover
            net_returns.iloc[i] = self.parent._clip_daily_return(
                gross_return - daily_cost
            )
            directions.iloc[i] = current_direction
            position_beta.iloc[i] = current_beta
            weight1.iloc[i] = new_weights[0]
            weight2.iloc[i] = new_weights[1]

        if current_direction != 0 and entry_date is not None:
            last_date = index[-1]
            last_z = _as_float(frame.at[last_date, "z_score"])
            old_weights = _position_weights(
                current_direction,
                current_beta,
                self.parent.normalize_exposure,
            )
            exit_turnover = _portfolio_turnover(old_weights, (0.0, 0.0))
            exit_cost = self.parent.calculate_transaction_costs(
                turnover=exit_turnover,
                z_score=last_z,
                event_type="end_of_sample",
            )
            transaction_costs.iloc[-1] += exit_cost
            turnover.iloc[-1] += exit_turnover
            net_returns.iloc[-1] = self.parent._clip_daily_return(
                net_returns.iloc[-1] - exit_cost
            )
            trade_cost += exit_cost
            trade_net_equity = max(trade_net_equity - exit_cost, 0.0)
            trades.append(
                self._make_trade(
                    ticker1=ticker1,
                    ticker2=ticker2,
                    entry_date=entry_date,
                    exit_date=pd.Timestamp(last_date),
                    entry_spread=entry_spread,
                    exit_spread=_as_float(frame.at[last_date, "spread"]),
                    direction=current_direction,
                    holding_period=(
                        len(index) - 1 - entry_index
                        if entry_index is not None
                        else 0
                    ),
                    exit_reason="End of Sample",
                    entry_beta=entry_beta,
                    exit_beta=current_beta,
                    entry_z=entry_z,
                    exit_z=last_z,
                    gross_return=trade_gross_equity - 1.0,
                    net_return=trade_net_equity - 1.0,
                    transaction_cost=trade_cost,
                )
            )
            directions.iloc[-1] = 0
            position_beta.iloc[-1] = 0.0
            weight1.iloc[-1] = 0.0
            weight2.iloc[-1] = 0.0

        execution = pd.DataFrame(
            {
                "price1_return": price1_returns,
                "price2_return": price2_returns,
                "gross_return": gross_returns,
                "transaction_cost": transaction_costs,
                "net_return": net_returns,
                "turnover": turnover,
                "direction": directions,
                "position_beta": position_beta,
                "weight1": weight1,
                "weight2": weight2,
            },
            index=index,
        )
        execution["equity_curve"] = self.parent.build_equity_curve(
            execution["net_return"]
        )
        return {"execution": execution, "trades": trades}

    def _entry_direction(self, z_score: float) -> int:
        """Map z-score to spread direction."""

        if z_score < -self.parent.entry_z:
            return 1
        if z_score > self.parent.entry_z:
            return -1
        return 0

    def _exit_reason(
        self,
        current_direction: int,
        current_trade_return: float,
        z_score: float,
        bar_index: int,
        entry_index: Optional[int],
    ) -> Optional[str]:
        """Determine whether an open trade must be closed."""

        if current_direction == 0 or entry_index is None:
            return None
        holding_period = bar_index - entry_index
        if (
            self.parent.stop_loss is not None
            and current_trade_return <= self.parent.stop_loss
        ):
            return "Stop Loss"
        if (
            self.parent.take_profit is not None
            and current_trade_return >= self.parent.take_profit
        ):
            return "Take Profit"
        if (
            self.parent.max_holding_period is not None
            and holding_period >= self.parent.max_holding_period
        ):
            return "Maximum Holding Period"
        if holding_period < self.parent.min_holding_period:
            return None
        if abs(z_score) <= self.parent.exit_z:
            return "Mean Reversion"
        if not self.parent.allow_reversals:
            return None
        if current_direction == 1 and z_score > self.parent.entry_z:
            return "Reversal"
        if current_direction == -1 and z_score < -self.parent.entry_z:
            return "Reversal"
        return None

    def _open_position_cost(
        self,
        direction: int,
        beta: float,
        z_score: float,
    ) -> Tuple[float, float]:
        """Return entry cost and turnover for a new position."""

        entry_weights = _position_weights(
            direction,
            beta,
            self.parent.normalize_exposure,
        )
        entry_turnover = _portfolio_turnover((0.0, 0.0), entry_weights)
        entry_cost = self.parent.calculate_transaction_costs(
            turnover=entry_turnover,
            z_score=z_score,
            event_type="entry",
        )
        return entry_cost, entry_turnover

    @staticmethod
    def _make_trade(
        ticker1: str,
        ticker2: str,
        entry_date: Optional[pd.Timestamp],
        exit_date: pd.Timestamp,
        entry_spread: float,
        exit_spread: float,
        direction: int,
        holding_period: int,
        exit_reason: str,
        entry_beta: float,
        exit_beta: float,
        entry_z: float,
        exit_z: float,
        gross_return: float,
        net_return: float,
        transaction_cost: float,
    ) -> Trade:
        """Create a normalized trade record."""

        direction_name = "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD"
        return Trade(
            ticker1=ticker1,
            ticker2=ticker2,
            entry_date=pd.Timestamp(entry_date),
            exit_date=pd.Timestamp(exit_date),
            entry_price=_as_float(entry_spread),
            exit_price=_as_float(exit_spread),
            direction=direction_name,
            holding_period=max(int(holding_period), 0),
            pnl=_as_float(net_return),
            exit_reason=exit_reason,
            entry_beta=_as_float(entry_beta),
            exit_beta=_as_float(exit_beta),
            entry_z=_as_float(entry_z),
            exit_z=_as_float(exit_z),
            gross_return=_as_float(gross_return),
            net_return=_as_float(net_return),
            transaction_cost=_as_float(transaction_cost),
        )




class StatisticalArbitrageEngine:
    """Object-oriented statistical arbitrage backtesting engine.

    The engine estimates rolling hedge ratios, constructs lagged z-score
    signals, executes spread trades with transaction costs, and reports
    pair-level and portfolio-level performance metrics.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        fee_bps: float = 15.0,
        lookback: int = 20,
        market: str = "NIFTY",
        stop_loss: Optional[float] = -0.05,
        take_profit: Optional[float] = 0.15,
        min_trades: int = 5,
        beta_window: int = 60,
        beta_min_obs: Optional[int] = None,
        beta_clip: float = 10.0,
        min_holding_period: int = 1,
        max_holding_period: Optional[int] = None,
        bid_ask_bps: float = 5.0,
        market_impact_bps: float = 2.0,
        slippage_bps: float = 0.0,
        initial_capital: float = 1.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        allow_reversals: bool = True,
        normalize_exposure: bool = True,
        max_abs_daily_return: Optional[float] = 0.50,
        z_score_clip: Optional[float] = 10.0,
    ) -> None:
        """Initialize the backtesting engine.

        Args:
            prices: Wide price DataFrame indexed by date and columned by ticker.
            entry_z: Absolute z-score threshold for opening a spread trade.
            exit_z: Absolute z-score threshold for mean-reversion exits.
            fee_bps: Explicit broker/exchange fee in basis points.
            lookback: Rolling z-score window; kept for notebook compatibility.
            market: Market name used to select the annual risk-free rate.
            stop_loss: Per-trade loss threshold in return space.
            take_profit: Per-trade profit threshold in return space.
            min_trades: Minimum completed trades required for a valid result.
            beta_window: Rolling OLS window for hedge-ratio estimation.
            beta_min_obs: Minimum observations required for rolling OLS.
            beta_clip: Absolute hedge-ratio cap for numerical stability.
            min_holding_period: Minimum bars before z-score/reversal exits.
            max_holding_period: Maximum bars before forced exit.
            bid_ask_bps: Bid-ask approximation in basis points.
            market_impact_bps: Impact coefficient per absolute z-score.
            slippage_bps: Additional slippage in basis points.
            initial_capital: Initial capital used for equity curves.
            periods_per_year: Trading periods used for annualization.
            allow_reversals: Whether an opposite entry signal can reverse.
            normalize_exposure: Normalize legs to one unit of gross exposure.
            max_abs_daily_return: Optional cap for one-period strategy returns.
            z_score_clip: Optional cap for unstable z-scores and impact costs.
        """

        self.prices = _clean_price_frame(prices)
        self.entry_z = abs(float(entry_z))
        self.exit_z = abs(float(exit_z))
        self.lookback = max(int(lookback), 2)
        self.zscore_window = self.lookback
        self.market = str(market).upper()
        self.fee_bps_input = float(fee_bps)
        self.fee_bps = float(fee_bps) / 10000.0
        self.fee_rate = self.fee_bps
        self.bid_ask_rate = float(bid_ask_bps) / 10000.0
        self.market_impact_rate = float(market_impact_bps) / 10000.0
        self.slippage_rate = float(slippage_bps) / 10000.0
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.min_trades = max(int(min_trades), 0)
        self.beta_window = max(int(beta_window), 2)
        self.beta_min_obs = (
            max(int(beta_min_obs), 2)
            if beta_min_obs is not None
            else self.beta_window
        )
        self.beta_min_obs = min(self.beta_min_obs, self.beta_window)
        self.beta_clip = abs(float(beta_clip))
        self.min_holding_period = max(int(min_holding_period), 0)
        self.max_holding_period = (
            max(int(max_holding_period), 1)
            if max_holding_period is not None
            else None
        )
        self.initial_capital = max(float(initial_capital), EPSILON)
        self.periods_per_year = max(int(periods_per_year), 1)
        self.allow_reversals = bool(allow_reversals)
        self.normalize_exposure = bool(normalize_exposure)
        self.max_abs_daily_return = max_abs_daily_return
        self.z_score_clip = z_score_clip
        self.annual_rfr = _market_annual_rfr(self.market)
        self.daily_rfr = self.annual_rfr / self.periods_per_year
        self.execution_engine = TradeExecutionEngine(self)

    def estimate_rolling_beta(
        self,
        ticker1: str,
        ticker2: str,
        window: Optional[int] = None,
    ) -> pd.Series:
        """Estimate a lagged rolling OLS hedge ratio.

        The hedge ratio is beta in ``log(P1) = alpha + beta * log(P2)``.
        Rolling covariance and variance are shifted by one bar so the beta
        used at time t is estimated only from information available before t.
        """

        pair_prices = self._get_pair_prices(ticker1, ticker2)
        if pair_prices.empty:
            return pd.Series(dtype="float64", name="rolling_beta")

        beta_window = max(int(window or self.beta_window), 2)
        min_obs = min(self.beta_min_obs, beta_window)
        log_price1 = np.log(pair_prices[ticker1])
        log_price2 = np.log(pair_prices[ticker2])
        covariance = log_price1.rolling(
            beta_window,
            min_periods=min_obs,
        ).cov(log_price2)
        variance = log_price2.rolling(
            beta_window,
            min_periods=min_obs,
        ).var()
        shifted_variance = variance.shift(1)
        beta = covariance.shift(1) / shifted_variance.where(
            shifted_variance.abs() > EPSILON
        )
        beta = beta.replace([np.inf, -np.inf], np.nan)
        beta = beta.clip(lower=-self.beta_clip, upper=self.beta_clip)
        beta = beta.ffill()
        beta.name = "rolling_beta"
        return beta

    def calculate_spread(
        self,
        ticker1: str,
        ticker2: str,
        rolling_beta: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Construct hedge-ratio adjusted log-price spread."""

        pair_prices = self._get_pair_prices(ticker1, ticker2)
        if pair_prices.empty:
            return pd.DataFrame()
        beta = (
            rolling_beta.copy()
            if rolling_beta is not None
            else self.estimate_rolling_beta(ticker1, ticker2)
        )
        beta = beta.reindex(pair_prices.index)
        log_price1 = np.log(pair_prices[ticker1])
        log_price2 = np.log(pair_prices[ticker2])
        spread = log_price1 - beta * log_price2
        result = pd.DataFrame(
            {
                "price1": pair_prices[ticker1],
                "price2": pair_prices[ticker2],
                "s1": pair_prices[ticker1],
                "s2": pair_prices[ticker2],
                "log_price1": log_price1,
                "log_price2": log_price2,
                "beta": beta,
                "spread": spread,
            },
            index=pair_prices.index,
        )
        result = result.replace([np.inf, -np.inf], np.nan)
        return result.dropna(subset=["price1", "price2", "beta", "spread"])

    def calculate_zscore(
        self,
        spread: Union[pd.Series, pd.DataFrame],
        window: Optional[int] = None,
    ) -> pd.Series:
        """Calculate lagged rolling spread z-score without look-ahead bias."""

        if isinstance(spread, pd.DataFrame):
            spread_series = spread["spread"] if "spread" in spread else spread.iloc[:, 0]
        else:
            spread_series = spread
        z_window = max(int(window or self.zscore_window), 2)
        spread_series = pd.to_numeric(spread_series, errors="coerce")
        spread_series = spread_series.replace([np.inf, -np.inf], np.nan)
        rolling_mean = spread_series.rolling(
            z_window,
            min_periods=z_window,
        ).mean().shift(1)
        rolling_std = spread_series.rolling(
            z_window,
            min_periods=z_window,
        ).std().shift(1)
        z_score = (spread_series - rolling_mean) / rolling_std.where(
            rolling_std.abs() > EPSILON
        )
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        if self.z_score_clip is not None:
            z_score = z_score.clip(
                lower=-abs(float(self.z_score_clip)),
                upper=abs(float(self.z_score_clip)),
            )
        z_score.name = "z_score"
        return z_score

    def generate_signals(self, z_score: pd.Series) -> pd.DataFrame:
        """Generate raw entry and exit signals from a z-score series."""

        clean_z = pd.to_numeric(z_score, errors="coerce")
        signals = pd.DataFrame(index=clean_z.index)
        signals["z_score"] = clean_z
        signals["entry_signal"] = 0
        signals.loc[clean_z < -self.entry_z, "entry_signal"] = 1
        signals.loc[clean_z > self.entry_z, "entry_signal"] = -1
        signals["long_entry"] = signals["entry_signal"] == 1
        signals["short_entry"] = signals["entry_signal"] == -1
        signals["exit_signal"] = clean_z.abs() <= self.exit_z
        return signals

    def execute_trades(
        self,
        data: pd.DataFrame,
        ticker1: str = "",
        ticker2: str = "",
    ) -> Dict[str, Any]:
        """Execute trades for a prepared pair data frame."""

        return self.execution_engine.execute(data, ticker1, ticker2)

    def calculate_transaction_costs(
        self,
        turnover: Union[float, pd.Series],
        z_score: Union[float, pd.Series] = 0.0,
        event_type: str = "trade",
    ) -> Union[float, pd.Series]:
        """Calculate transaction costs from turnover and market state.

        Costs include explicit fees, bid-ask spread approximation, slippage,
        and a z-score scaled market-impact term. Reversals naturally cost more
        because their turnover is larger.
        """

        del event_type
        impact_z = np.abs(z_score)
        if self.z_score_clip is not None:
            impact_z = np.minimum(impact_z, abs(float(self.z_score_clip)))
        cost_rate = (
            self.fee_rate
            + self.bid_ask_rate
            + self.slippage_rate
            + self.market_impact_rate * impact_z
        )
        costs = turnover * cost_rate
        if isinstance(costs, pd.Series):
            return costs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _as_float(costs)

    def build_equity_curve(
        self,
        net_returns: pd.Series,
        initial_capital: Optional[float] = None,
    ) -> pd.Series:
        """Build a compounded equity curve from strategy returns."""

        starting_capital = (
            self.initial_capital
            if initial_capital is None
            else max(float(initial_capital), EPSILON)
        )
        returns = _clean_return_series(net_returns)
        equity_curve = starting_capital * (1.0 + returns).cumprod()
        equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan)
        return equity_curve.ffill().fillna(starting_capital)

    def calculate_metrics(
        self,
        ticker1: str,
        ticker2: str,
        execution: pd.DataFrame,
        trade_log: Sequence[Trade],
        rolling_beta: pd.Series,
        spread: pd.Series,
        z_score: pd.Series,
    ) -> BacktestResult:
        """Calculate a complete ``BacktestResult`` for one pair."""

        returns = _clean_return_series(execution["net_return"])
        equity_curve = (
            execution["equity_curve"]
            if "equity_curve" in execution
            else self.build_equity_curve(returns)
        )
        turnover = _as_float(execution["turnover"].sum())
        stats = _performance_statistics(
            returns=returns,
            equity_curve=equity_curve,
            trades=trade_log,
            annual_rfr=self.annual_rfr,
            periods_per_year=self.periods_per_year,
            turnover=turnover,
        )
        position_columns = [
            "direction",
            "position_beta",
            "weight1",
            "weight2",
            "turnover",
            "transaction_cost",
        ]
        positions = execution[position_columns].copy()
        return BacktestResult(
            ticker1=ticker1,
            ticker2=ticker2,
            rolling_beta=rolling_beta,
            roi=stats["roi"],
            cagr=stats["cagr"],
            annual_return=stats["annual_return"],
            annual_volatility=stats["annual_volatility"],
            sharpe=stats["sharpe"],
            sortino=stats["sortino"],
            calmar=stats["calmar"],
            maximum_drawdown=stats["maximum_drawdown"],
            profit_factor=stats["profit_factor"],
            win_rate=stats["win_rate"],
            average_trade_return=stats["average_trade_return"],
            average_holding_period=stats["average_holding_period"],
            trade_count=stats["trade_count"],
            turnover=stats["turnover"],
            trade_log=list(trade_log),
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            spread=spread,
            z_score=z_score,
        )

    def run_pair_backtest(self, ticker1: str, ticker2: str) -> Dict[str, Any]:
        """Run a complete backtest for one ticker pair."""

        spread_frame = self.calculate_spread(ticker1, ticker2)

        if spread_frame.empty:
                return {}

        z_score = self.calculate_zscore(spread_frame["spread"])

        data = spread_frame.join(z_score, how="left")
        data = data.dropna(subset=["price1","price2","beta","spread","z_score"])

       
        minimum_required = max(self.min_holding_period + 2, 3)

        if len(data) < minimum_required:
            return {}

        execution_result = self.execute_trades(data, ticker1, ticker2)

        execution = execution_result["execution"]
        trade_log = execution_result["trades"]
     
        if len(trade_log) < self.min_trades:
                return {}
        result = self.calculate_metrics(
            ticker1=ticker1,
            ticker2=ticker2,
            execution=execution,
            trade_log=trade_log,
            rolling_beta=data["beta"],
            spread=data["spread"],
            z_score=data["z_score"],
        )
        return result.to_dict()

    def calculate_spread_series(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """Backward-compatible spread/z-score helper used by old notebooks."""

        spread_frame = self.calculate_spread(ticker1, ticker2)
        if spread_frame.empty:
            return pd.DataFrame()
        z_score = self.calculate_zscore(spread_frame["spread"])
        result = spread_frame.join(z_score, how="left")
        result = result.dropna(subset=["s1", "s2", "z_score"])
        return result

    def test_cluster(self, cluster_tickers: List[str]) -> pd.DataFrame:
        """Backtest every unique pair in a cluster and rank by Sharpe."""

        tickers = [
            ticker
            for ticker in dict.fromkeys(cluster_tickers)
            if ticker in self.prices.columns
        ]
        if len(tickers) < 2:
            return pd.DataFrame()
        results: List[Dict[str, Any]] = []
        for ticker1, ticker2 in combinations(tickers, 2):
            result = self.run_pair_backtest(ticker1, ticker2)
            if result:
                results.append(result)
        if not results:
            return pd.DataFrame()
        result_frame = pd.DataFrame(results)
        return result_frame.sort_values(
            "Sharpe",
            ascending=False,
        ).reset_index(drop=True)

    def portfolio_backtest(
        self,
        pairs: Union[pd.DataFrame, Sequence[Tuple[str, str]]],
        weighting: str = "equal",
        risk_lookback: int = 60,
    ) -> Dict[str, Any]:
        """Run a simultaneous multi-pair portfolio backtest."""

        portfolio_engine = PortfolioEngine(self)
        return portfolio_engine.run(
            pairs=pairs,
            weighting=weighting,
            risk_lookback=risk_lookback,
        )

    def _get_pair_prices(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """Return clean pair prices, or an empty frame when unavailable."""

        if ticker1 not in self.prices.columns or ticker2 not in self.prices.columns:
            return pd.DataFrame()
        pair_prices = self.prices[[ticker1, ticker2]].copy()
        pair_prices = pair_prices.replace([np.inf, -np.inf], np.nan)
        pair_prices = pair_prices.dropna(how="any")
        valid = (pair_prices[ticker1] > 0.0) & (pair_prices[ticker2] > 0.0)
        return pair_prices.loc[valid]

    def _clip_daily_return(self, value: float) -> float:
        """Clip unstable one-period returns while preserving sign."""

        result = _as_float(value)
        if self.max_abs_daily_return is None:
            return max(result, -0.999999)
        cap = abs(float(self.max_abs_daily_return))
        return float(np.clip(result, -min(cap, 0.999999), cap))


class PortfolioEngine:
    """Portfolio-level engine for simultaneous statistical-arbitrage pairs."""

    def __init__(self, pair_engine: StatisticalArbitrageEngine) -> None:
        self.pair_engine = pair_engine

    def run(
        self,
        pairs: Union[pd.DataFrame, Sequence[Tuple[str, str]]],
        weighting: str = "equal",
        risk_lookback: int = 60,
    ) -> Dict[str, Any]:
        """Backtest a portfolio of pairs with equal or inverse-risk weights."""

        pair_list = self._normalize_pairs(pairs)
        if not pair_list:
            return {}
        pair_results: List[Dict[str, Any]] = []
        return_series: Dict[str, pd.Series] = {}
        all_trades: List[Trade] = []
        for ticker1, ticker2 in pair_list:
            result = self.pair_engine.run_pair_backtest(ticker1, ticker2)
            if not result:
                continue
            pair_name = f"{ticker1}-{ticker2}"
            pair_results.append(result)
            returns = _clean_return_series(result["returns"])
            returns.name = pair_name
            return_series[pair_name] = returns
            all_trades.extend(
                self._trade_from_record(record)
                for record in result.get("trade_log", [])
            )
        if not return_series:
            return {}
        returns_frame = pd.DataFrame(return_series).fillna(0.0)
        weights = self._build_weights(
            returns_frame=returns_frame,
            weighting=weighting,
            risk_lookback=risk_lookback,
        )
        portfolio_returns = (returns_frame * weights).sum(axis=1)
        portfolio_returns.name = "portfolio_return"
        equity_curve = self.pair_engine.build_equity_curve(portfolio_returns)
        stats = _performance_statistics(
            returns=portfolio_returns,
            equity_curve=equity_curve,
            trades=all_trades,
            annual_rfr=self.pair_engine.annual_rfr,
            periods_per_year=self.pair_engine.periods_per_year,
            turnover=0.0,
        )
        return {
            "Portfolio": True,
            "Pairs": pair_list,
            "Pair_Count": len(return_series),
            "Weighting": weighting,
            "ROI": stats["roi"],
            "CAGR": stats["cagr"] * 100.0,
            "CAGR_Decimal": stats["cagr"],
            "Annual_Return": stats["annual_return"] * 100.0,
            "Annual_Return_Decimal": stats["annual_return"],
            "Annual_Volatility": stats["annual_volatility"] * 100.0,
            "Annual_Volatility_Decimal": stats["annual_volatility"],
            "Sharpe": stats["sharpe"],
            "Sortino": stats["sortino"],
            "Calmar": stats["calmar"],
            "Max_Drawdown": stats["maximum_drawdown"] * 100.0,
            "Maximum_Drawdown": stats["maximum_drawdown"],
            "Profit_Factor": stats["profit_factor"],
            "Win_Rate": stats["win_rate"],
            "Average_Trade_Return": stats["average_trade_return"],
            "Average_Holding_Period": stats["average_holding_period"],
            "Trades": stats["trade_count"],
            "Trade_Count": stats["trade_count"],
            "equity_curve": equity_curve,
            "portfolio_returns": portfolio_returns,
            "pair_returns": returns_frame,
            "weights": weights,
            "pair_results": pair_results,
        }

    @staticmethod
    def _normalize_pairs(
        pairs: Union[pd.DataFrame, Sequence[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        """Convert supported pair inputs to a unique list of ticker tuples."""

        normalized: List[Tuple[str, str]] = []
        if isinstance(pairs, pd.DataFrame):
            lower_columns = {column.lower(): column for column in pairs.columns}
            first_column = lower_columns.get("ticker1") or lower_columns.get("s1")
            second_column = lower_columns.get("ticker2") or lower_columns.get("s2")
            if first_column is None or second_column is None:
                raise ValueError(
                    "pairs DataFrame must contain Ticker1/Ticker2 columns"
                )
            iterator: Iterable[Tuple[Any, Any]] = zip(
                pairs[first_column],
                pairs[second_column],
            )
        else:
            iterator = pairs
        seen = set()
        for pair in iterator:
            if len(pair) != 2:
                continue
            ticker1, ticker2 = str(pair[0]), str(pair[1])
            key = tuple(sorted((ticker1, ticker2)))
            if ticker1 == ticker2 or key in seen:
                continue
            seen.add(key)
            normalized.append((ticker1, ticker2))
        return normalized

    @staticmethod
    def _build_weights(
        returns_frame: pd.DataFrame,
        weighting: str,
        risk_lookback: int,
    ) -> pd.DataFrame:
        """Build equal or inverse-volatility portfolio weights."""

        if returns_frame.empty:
            return returns_frame.copy()
        method = str(weighting).lower().replace("-", "_")
        if method in {"equal", "equal_weight", "equal_weighted"}:
            return pd.DataFrame(
                1.0 / returns_frame.shape[1],
                index=returns_frame.index,
                columns=returns_frame.columns,
            )
        if method not in {"risk", "risk_weight", "risk_weighted", "inverse_vol"}:
            raise ValueError("weighting must be 'equal' or 'risk'")
        lookback = max(int(risk_lookback), 2)
        rolling_volatility = returns_frame.rolling(
            lookback,
            min_periods=max(2, lookback // 2),
        ).std().shift(1)
        inverse_volatility = 1.0 / rolling_volatility.where(
            rolling_volatility > EPSILON
        )
        inverse_volatility = inverse_volatility.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)
        equal_weight = 1.0 / returns_frame.shape[1]
        return weights.fillna(equal_weight)

    @staticmethod
    def _trade_from_record(record: Dict[str, Any]) -> Trade:
        """Reconstruct a Trade dataclass from a dictionary record."""

        return Trade(
            ticker1=str(record.get("Ticker1", "")),
            ticker2=str(record.get("Ticker2", "")),
            entry_date=pd.Timestamp(record.get("Entry Date")),
            exit_date=pd.Timestamp(record.get("Exit Date")),
            entry_price=_as_float(record.get("Entry Price")),
            exit_price=_as_float(record.get("Exit Price")),
            direction=str(record.get("Direction", "")),
            holding_period=int(record.get("Holding Period", 0)),
            pnl=_as_float(record.get("PnL")),
            exit_reason=str(record.get("Exit Reason", "")),
            entry_beta=_as_float(record.get("Entry Beta")),
            exit_beta=_as_float(record.get("Exit Beta")),
            entry_z=_as_float(record.get("Entry Z")),
            exit_z=_as_float(record.get("Exit Z")),
            gross_return=_as_float(record.get("Gross Return")),
            net_return=_as_float(record.get("Net Return")),
            transaction_cost=_as_float(record.get("Transaction Cost")),
        )


VectorizedBacktester = StatisticalArbitrageEngine

