# Added AFML Features

This document summarizes the newly added AFML-style utilities and where to look for more research or extension.

## Core Backtesting (`afml/core`)
- `VectorizedBacktester`: Simple rebalanced portfolio backtester with metric tracking.
- `StrategyBacktester`: Single-series backtester for position-based strategies.
- `StreamingBacktester`: Chunk-based retraining workflow for streaming data.
- `KalmanFilterBacktester`: Lightweight Kalman-style regression backtest.

## Data Utilities (`afml/data`)
- `SyntheticData`: Generates synthetic price series.
- `BacktestSimulator`: Basic backtest overfitting simulation.
- `BarSampler`: Time/tick/volume/dollar bars plus imbalance/run bars.
- `etf_trick`, `futures_roll`, `pca_hedge_weights`: Multi-product processing helpers.
- `DailyVolatility`, `TripleBarrierLabeling`, `MetaLabeling`: Labeling utilities.
- `SampleWeights`, `UniquenessSampling`: Sample weighting and uniqueness tools.
- `SequentialBootstrap`, `FeatureSampling`: Sampling and feature alignment helpers.
- `HighFrequencyDataSimulator`: Synthetic trade generator.

## Features (`afml/features`)
- `FractionalDifferentiation`: Fractional differencing (FFD).
- `StationarityTests`: Simple variance-based optimal-d search.
- `EntropyFeatures`: Plug-in entropy, Lempel-Ziv complexity, rolling entropy.

## Machine Learning (`afml/machine_learning`)
- `FeatureImportance`: MDI/MDA and clustering helpers.
- `TimeSeriesFeatureImportance`: Rolling MDA.
- `FeatureInteractionImportance`: Pairwise interaction scoring.
- `DisjointFeatureEnsemble`, `DiversityEnsemble`, `StackedGeneralizationEnsemble`, `BetSizingEnsemble`.

## Portfolio (`afml/portfolio`)
- `MetaLabeler`, `KellyBetSizing`, `BetSizingStrategies`.
- `equal_weight_strategy`, `minimum_variance_strategy`, `momentum_strategy`.
- `PortfolioOptimizer`: Minimum variance, maximum Sharpe, efficient frontier.
- `PortfolioAnalytics`: Performance summary, risk contributions, concentration.

## Microstructure (`afml/microstructure`)
- `TickDataProcessor`: Clean trades and create tick bars.
- `MarketMicrostructureAnalyzer`: Kyle's Lambda estimation.

## Validation (`afml/validation`)
- `EfficiencyAnalyzer`: Approximation error analysis.
- `DangerDetector`: Look-ahead and leakage checks.
- `RobustnessChecker`: Walk-forward, subsample, stability tests.
- `PerformanceStatistics`: Sharpe ratio and drawdown helpers.
- `PurgedKFold`, `CombinatorialPurgedKFold`, `WalkForwardAnalysis`.
- `StructuralBreaks`: CUSUM mean/volatility shift detection.

## Utilities (`afml/utils`)
- `VisualizationTools`: Common plotting helpers.
- `mp_pandas_obj`: Parallel apply helper.
