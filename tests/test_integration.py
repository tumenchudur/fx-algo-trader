"""Integration tests for the FX Trading System."""

import pytest
from pathlib import Path
import pandas as pd

from fx_trading.config.models import (
    BacktestConfig,
    StrategyConfig,
    CostConfig,
    RiskConfig,
    WalkForwardConfig,
)
from fx_trading.data.synthetic import SyntheticDataGenerator
from fx_trading.data.ingestion import DataIngestor
from fx_trading.data.validation import DataQualityValidator
from fx_trading.backtesting.engine import Backtester
from fx_trading.backtesting.walkforward import WalkForwardAnalysis
from fx_trading.monitoring.reports import ReportGenerator


class TestEndToEndBacktest:
    """End-to-end backtest tests."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate larger sample data."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate(
            symbol="EURUSD",
            timeframe="M5",
            num_bars=1000,
        )

    @pytest.fixture
    def full_config(self, tmp_path: Path) -> BacktestConfig:
        """Create full backtest config."""
        return BacktestConfig(
            run_id="integration_test",
            output_dir=tmp_path / "runs",
            data_path=tmp_path / "data.parquet",
            symbols=["EURUSD"],
            initial_capital=10000.0,
            strategy=StrategyConfig(
                name="integration_test_strategy",
                strategy_type="volatility_breakout",
                symbols=["EURUSD"],
                use_stop_loss=True,
                use_take_profit=True,
                risk_reward_ratio=2.0,
                params={
                    "lookback": 20,
                    "atr_period": 14,
                    "atr_threshold": 1.0,
                    "sl_atr_multiplier": 2.0,
                },
            ),
            costs=CostConfig(
                commission_per_lot=7.0,
                slippage_pips=0.5,
            ),
            risk=RiskConfig(
                max_risk_per_trade_pct=1.0,
                max_open_positions=3,
                daily_loss_limit_pct=3.0,
                max_drawdown_pct=15.0,
            ),
            random_seed=42,
        )

    def test_full_backtest_workflow(self, full_config, sample_data, tmp_path):
        """Test complete backtest workflow."""
        # 1. Save data
        data_path = tmp_path / "EURUSD_M5.parquet"
        sample_data.to_parquet(data_path)
        full_config.data_path = data_path

        # 2. Run backtest
        backtester = Backtester(full_config)
        result = backtester.run(data=sample_data)

        # 3. Verify result
        assert result is not None
        assert result.final_equity > 0
        assert result.initial_capital == 10000.0

        # 4. Generate report
        report_gen = ReportGenerator()
        report_path = report_gen.save_full_results(result, tmp_path / "results")

        assert report_path.exists()
        assert (report_path / "report.html").exists()
        assert (report_path / "summary.json").exists()

    def test_data_pipeline_integration(self, tmp_path):
        """Test data ingestion and validation pipeline."""
        # 1. Generate synthetic data and save as CSV
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(symbol="EURUSD", timeframe="M5", num_bars=500)

        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path)

        # 2. Ingest data
        ingestor = DataIngestor()
        ingested, report = ingestor.ingest_csv(csv_path, "EURUSD", "M5")

        # 3. Validate
        assert report.quality_score > 90
        assert report.total_rows == len(df)

        # 4. Save to parquet
        parquet_path = tmp_path / "EURUSD_M5.parquet"
        ingestor.save_parquet(ingested, parquet_path)

        # 5. Load and verify
        loaded = ingestor.load_parquet(parquet_path)
        assert len(loaded) == len(ingested)

    def test_risk_engine_integration(self, full_config, sample_data):
        """Test risk engine integration with backtester."""
        # Set tight risk limits
        full_config.risk.daily_loss_limit_pct = 1.0
        full_config.risk.max_drawdown_pct = 5.0

        backtester = Backtester(full_config)
        result = backtester.run(data=sample_data)

        # Risk engine should have controlled losses
        assert result.max_drawdown_pct <= full_config.risk.max_drawdown_pct + 1.0

    def test_multiple_strategies(self, sample_data, tmp_path):
        """Test running different strategies."""
        strategies = ["volatility_breakout", "mean_reversion"]
        results = {}

        for strategy_type in strategies:
            config = BacktestConfig(
                run_id=f"test_{strategy_type}",
                output_dir=tmp_path,
                data_path=tmp_path / "data.parquet",
                symbols=["EURUSD"],
                initial_capital=10000.0,
                strategy=StrategyConfig(
                    name=strategy_type,
                    strategy_type=strategy_type,
                    symbols=["EURUSD"],
                    params={} if strategy_type == "mean_reversion" else {
                        "lookback": 20,
                        "atr_period": 14,
                    },
                ),
                costs=CostConfig(),
                risk=RiskConfig(),
                random_seed=42,
            )

            backtester = Backtester(config)
            results[strategy_type] = backtester.run(data=sample_data.copy())

        # Both should complete
        assert all(r is not None for r in results.values())


class TestWalkForwardIntegration:
    """Integration tests for walk-forward analysis."""

    def test_walk_forward_workflow(self, tmp_path):
        """Test complete walk-forward workflow."""
        # Generate data
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate(symbol="EURUSD", timeframe="M5", num_bars=2000)

        base_config = BacktestConfig(
            run_id="wf_test",
            output_dir=tmp_path,
            data_path=tmp_path / "data.parquet",
            symbols=["EURUSD"],
            initial_capital=10000.0,
            strategy=StrategyConfig(
                name="wf_strategy",
                strategy_type="mean_reversion",
                symbols=["EURUSD"],
            ),
            costs=CostConfig(),
            risk=RiskConfig(),
            random_seed=42,
        )

        wf_config = WalkForwardConfig(
            base_config=base_config,
            num_windows=3,
            train_pct=0.7,
        )

        wf = WalkForwardAnalysis(wf_config)
        result = wf.run(data=data)

        assert len(result.windows) == 3
        assert result.window_consistency >= 0
        assert result.window_consistency <= 1


class TestCostModelIntegration:
    """Integration tests for cost models."""

    def test_costs_impact_returns(self, sample_data, tmp_path):
        """Higher costs should reduce returns."""
        # Low cost config
        low_cost_config = BacktestConfig(
            run_id="low_cost",
            output_dir=tmp_path,
            data_path=tmp_path / "data.parquet",
            symbols=["EURUSD"],
            initial_capital=10000.0,
            strategy=StrategyConfig(
                name="test",
                strategy_type="volatility_breakout",
                symbols=["EURUSD"],
                params={"lookback": 10, "atr_period": 7, "atr_threshold": 0.5},
            ),
            costs=CostConfig(
                commission_per_lot=0.0,
                slippage_pips=0.0,
            ),
            risk=RiskConfig(),
            random_seed=42,
        )

        # High cost config
        high_cost_config = BacktestConfig(
            run_id="high_cost",
            output_dir=tmp_path,
            data_path=tmp_path / "data.parquet",
            symbols=["EURUSD"],
            initial_capital=10000.0,
            strategy=StrategyConfig(
                name="test",
                strategy_type="volatility_breakout",
                symbols=["EURUSD"],
                params={"lookback": 10, "atr_period": 7, "atr_threshold": 0.5},
            ),
            costs=CostConfig(
                commission_per_lot=15.0,
                slippage_pips=2.0,
            ),
            risk=RiskConfig(),
            random_seed=42,
        )

        bt_low = Backtester(low_cost_config)
        result_low = bt_low.run(data=sample_data.copy())

        bt_high = Backtester(high_cost_config)
        result_high = bt_high.run(data=sample_data.copy())

        if result_low.total_trades > 0 and result_high.total_trades > 0:
            # Higher costs should result in lower returns
            # (or equal if no trades happened)
            assert result_high.total_return_pct <= result_low.total_return_pct + 0.1


class TestReproducibility:
    """Tests for result reproducibility."""

    def test_same_seed_same_results(self, sample_data, tmp_path):
        """Same seed should produce identical results."""
        config = BacktestConfig(
            run_id="repro_test",
            output_dir=tmp_path,
            data_path=tmp_path / "data.parquet",
            symbols=["EURUSD"],
            initial_capital=10000.0,
            strategy=StrategyConfig(
                name="test",
                strategy_type="volatility_breakout",
                symbols=["EURUSD"],
                params={"lookback": 10, "atr_period": 7, "atr_threshold": 0.5},
            ),
            costs=CostConfig(slippage_pips=0.5),
            risk=RiskConfig(),
            random_seed=12345,
        )

        results = []
        for _ in range(3):
            bt = Backtester(config)
            result = bt.run(data=sample_data.copy())
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].total_trades == results[0].total_trades
            assert abs(results[i].final_equity - results[0].final_equity) < 0.01
            assert abs(results[i].total_return_pct - results[0].total_return_pct) < 0.01
